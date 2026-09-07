"""R7-O0 Phase 4.1 D0-only PPO smoke-training driver.

This entry point validates plumbing only. It never evaluates task performance,
never restores legacy optimizer state, and never writes a trained checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


PROTOCOL_ID = "R7_O0_CONTEXT_VALUE_V1"
SMOKE_ID = "R7_O0_PHASE41C_CLEAN_SMOKE_V1"
EXPECTED_GIT_HEAD = "3bcb4ea1c77e9c6beb199bd09d30ce05bc7eea2f"
EXPECTED_SOURCE_SHA256 = "46675eecc4e51372ec2c336d9637ef588a61b80e24aeb796de263bd1e75ce364"
EXPECTED_EXPANDED_SHA256 = "45fd281ac8bf96fa699f21be49eff301cf7f40f0a9c3b0bb3cf0e359b3e879e0"
EXPECTED_TRAINABLE_PARAMETERS = 572_705
EXPECTED_STATE_TENSOR_ELEMENTS = 572_907

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ROOT = Path(r"D:\Research\LEAP\LEAP_Hand_Isaac_Lab")
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "logs/r7_o0_smoke_phase41c"
SOURCE_CHECKPOINT_RELATIVE = Path("logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth")
EXPANDED_CHECKPOINT_RELATIVE = Path("logs/r7_o0_context_value/checkpoints/r7_o0_initialization_99d.pth")
AGENT_YAML_RELATIVE = Path(
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml"
)
TASK_ID = "Isaac-Reorient-Cube-Leap-CRI"
BRANCHES = ("history_only", "oracle")
NUM_ENVS = 1024
SEED = 7100
HORIZON_LENGTH = 32
MINIBATCH_SIZE = 32768
MINI_EPOCHS = 5
SEQ_LENGTH = 4
OBSERVATION_DIM = 96
CONTEXT_DIM = 3
POLICY_INPUT_DIM = 99
ACTION_DIM = 16
PHYSICS_DT = 1.0 / 120.0
DECIMATION = 4
EXPECTED_D0_CONTEXT = (-1.0, 1.0, 1.0)
GRU_INPUT_PARAMETER = "a2c_network.rnn.rnn.weight_ih_l0"
ALLOCATION_STAGES = {  # 资源边界编号与用户冻结定义一致。
    0: "before_app_launcher",
    1: "after_app_launcher",
    2: "after_gym_make",
    3: "after_env_wrapper_initialization",
    4: "after_99d_adapter_creation",
    5: "after_rl_games_runner_load",
    6: "after_agent_model_construction",
    7: "after_expanded_checkpoint_load",
    8: "after_agent_init_tensors",
    9: "after_first_env_reset",
    10: "after_first_forward",
    11: "after_first_backward",
    12: "after_first_optimizer_step",
}

PROTECTED_RELATIVE_PATHS = (
    Path("source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/reorientation_cri_env.py"),
    Path("source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/factor_conditions.py"),
    Path("source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/leap_hand_cri_env_cfg.py"),
    Path("source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py"),
    Path("scripts/rl_games/train.py"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _nvidia_smi_snapshot() -> dict[str, Any]:  # 读取系统级 GPU 显存，不初始化 Torch CUDA。
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8").stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    devices: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        devices.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "driver_version": fields[5],
            }
        )
    return {"available": bool(devices), "devices": devices}


def _windows_commit_snapshot() -> dict[str, Any]:  # 使用只读 Win32 API 获取系统 commit 边界。
    if os.name != "nt":
        return {"available": False, "reason": "non_windows"}
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):  # Win32 MEMORYSTATUSEX 的字段布局。
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        state = MemoryStatusEx()
        state.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
        return {
            "available": True,
            "physical_total_bytes": int(state.ullTotalPhys),
            "physical_available_bytes": int(state.ullAvailPhys),
            "commit_limit_bytes": int(state.ullTotalPageFile),
            "commit_available_bytes": int(state.ullAvailPageFile),
            "commit_used_bytes": int(state.ullTotalPageFile - state.ullAvailPageFile),
            "virtual_total_bytes": int(state.ullTotalVirtual),
            "virtual_available_bytes": int(state.ullAvailVirtual),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_resource_snapshot(torch_module: Any | None = None) -> dict[str, Any]:  # 汇总阶段资源但不强制 CUDA 初始化。
    snapshot: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "gpu_system": _nvidia_smi_snapshot(),
        "windows_memory": _windows_commit_snapshot(),
    }
    try:
        import psutil

        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()._asdict()
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        snapshot["system_memory"] = {
            "available": True,
            "total_bytes": int(virtual_memory.total),
            "available_bytes": int(virtual_memory.available),
            "used_bytes": int(virtual_memory.used),
            "percent": float(virtual_memory.percent),
            "pagefile_total_bytes": int(swap_memory.total),
            "pagefile_used_bytes": int(swap_memory.used),
            "pagefile_free_bytes": int(swap_memory.free),
        }
        snapshot["process_memory"] = {
            "available": True,
            **{name: int(value) for name, value in process_memory.items()},
        }
    except Exception as exc:
        snapshot["system_memory"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        snapshot["process_memory"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    cuda_initialized = bool(torch_module is not None and torch_module.cuda.is_initialized())
    snapshot["torch_cuda"] = {"initialized": cuda_initialized}
    if cuda_initialized:
        snapshot["torch_cuda"].update(
            {
                "device": str(torch_module.cuda.current_device()),
                "allocated_bytes": int(torch_module.cuda.memory_allocated()),
                "reserved_bytes": int(torch_module.cuda.memory_reserved()),
                "max_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
            }
        )
    return snapshot


class AllocationTimeline:  # 将每个资源边界立即原子落盘。
    def __init__(
        self,
        path: Path,
        branch: str,
        snapshot_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self.branch = branch
        self.snapshot_provider = snapshot_provider or collect_resource_snapshot
        self.stages: list[dict[str, Any]] = []
        self._completed: set[int] = set()

    @property
    def completed_stage_ids(self) -> set[int]:  # 返回副本，防止调用方改变内部状态。
        return set(self._completed)

    def _persist(self) -> None:  # 每次阶段更新都保留 crash-safe 证据。
        atomic_write_json(
            self.path,
            {
                "protocol_id": PROTOCOL_ID,
                "smoke_id": SMOKE_ID,
                "branch": self.branch,
                "schema_version": 1,
                "stages": self.stages,
                "scientific_conclusion_generated": False,
            },
        )

    def record(self, stage_id: int, details: dict[str, Any] | None = None) -> None:  # 一个边界只记录首次完成状态。
        if stage_id not in ALLOCATION_STAGES:
            raise ValueError(f"Unknown allocation stage: {stage_id}")
        if stage_id in self._completed:
            return
        entry = {
            "stage_id": stage_id,
            "stage_name": ALLOCATION_STAGES[stage_id],
            "status": "COMPLETE",
            "resources": self.snapshot_provider(),
        }
        if details:
            entry["details"] = details
        self.stages.append(entry)
        self._completed.add(stage_id)
        self._persist()

    def fail_first_incomplete(self, exc: BaseException) -> None:  # 把异常绑定到首个未完成边界。
        stage_id = next((index for index in ALLOCATION_STAGES if index not in self._completed), max(ALLOCATION_STAGES))
        self.stages.append(
            {
                "stage_id": stage_id,
                "stage_name": ALLOCATION_STAGES[stage_id],
                "status": "FAILED",
                "resources": self.snapshot_provider(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        self._persist()


def classify_runtime_failure(completed_stages: set[int], exc: BaseException) -> str:  # 按冻结首次失败边界分类。
    message = str(exc).lower()
    resource_terms = ("memory", "alloc", "cuda", "physx", "resource", "pinned")
    if 2 not in completed_stages and ("articulation" in message or "physics" in message or "gym" in message):
        return "TRANSIENT_ENVIRONMENT_LIFECYCLE_FAILURE_REPRODUCED"
    if 2 in completed_stages and 8 not in completed_stages:
        if 7 in completed_stages or any(term in message for term in resource_terms):
            return "PPO_BUFFER_ALLOCATION_LIMIT_IDENTIFIED" if 7 in completed_stages else "FULL_STACK_RESOURCE_LIMIT_IDENTIFIED"
    if 8 in completed_stages:
        return "PPO_PIPELINE_CONTRACT_FAILURE"
    return "ROOT_CAUSE_UNRESOLVED"


def git_bytes(repo_root: Path, *arguments: str) -> bytes:
    command = [
        "git",
        "-c",
        f"safe.directory={repo_root.as_posix()}",
        "-C",
        str(repo_root),
        *arguments,
    ]
    return subprocess.run(command, check=True, capture_output=True).stdout


def git_text(repo_root: Path, *arguments: str) -> str:
    return git_bytes(repo_root, *arguments).decode("utf-8", errors="strict").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated R7-O0 D0 PPO plumbing smoke")
    parser.add_argument("--branch", required=True, choices=BRANCHES)
    parser.add_argument("--max_iterations", type=int, default=1)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def validate_cli_contract(args: argparse.Namespace) -> None:
    if args.max_iterations < 1 or args.max_iterations > 100:
        raise ValueError("--max_iterations must be in [1,100]")
    if args.branch not in BRANCHES:
        raise ValueError(f"unsupported branch: {args.branch}")


def nested_value(mapping: dict[str, Any], dotted: str) -> Any:
    current: Any = mapping
    for component in dotted.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(f"Missing frozen config field: {dotted}")
        current = current[component]
    return current


def normalize_committed_yaml_scalars(agent_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Canonicalize PyYAML's string representation of the frozen scientific notation scalar."""
    path = "params.config.learning_rate"
    value = nested_value(agent_cfg, path)
    if not isinstance(value, str):
        return []
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RuntimeError(f"Non-finite frozen learning rate: {value!r}")
    agent_cfg["params"]["config"]["learning_rate"] = parsed
    return [{"path": path, "source_value": value, "runtime_value": parsed, "semantic_change": False}]


def validate_frozen_agent_config(agent_cfg: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "params.algo.name": "a2c_continuous",
        "params.model.name": "continuous_a2c_logstd",
        "params.network.name": "actor_critic",
        "params.network.separate": False,
        "params.network.space.continuous.fixed_sigma": True,
        "params.network.mlp.units": [512, 256, 128],
        "params.network.mlp.activation": "elu",
        "params.network.rnn.name": "gru",
        "params.network.rnn.units": 256,
        "params.network.rnn.layers": 1,
        "params.network.rnn.before_mlp": True,
        "params.network.rnn.concat_input": True,
        "params.network.rnn.layer_norm": True,
        "params.config.env_name": "rlgpu",
        "params.config.ppo": True,
        "params.config.mixed_precision": False,
        "params.config.normalize_input": True,
        "params.config.normalize_value": True,
        "params.config.normalize_advantage": True,
        "params.config.gamma": 0.99,
        "params.config.tau": 0.95,
        "params.config.learning_rate": 5e-4,
        "params.config.lr_schedule": "adaptive",
        "params.config.grad_norm": 1.0,
        "params.config.horizon_length": HORIZON_LENGTH,
        "params.config.minibatch_size": MINIBATCH_SIZE,
        "params.config.mini_epochs": MINI_EPOCHS,
        "params.config.seq_length": SEQ_LENGTH,
        "params.env.clip_observations": 5.0,
        "params.env.clip_actions": 1.0,
    }
    actual = {path: nested_value(agent_cfg, path) for path in expected}
    mismatches = {
        path: {"expected": expected[path], "actual": actual[path]}
        for path in expected
        if actual[path] != expected[path]
    }
    if mismatches:
        raise RuntimeError(f"Frozen committed PPO config mismatch: {mismatches}")
    return {"status": "PASS", "fields": actual, "mismatches": {}}


def normalized_context(mass_scale: float, friction: float, motor_scale: float) -> tuple[float, float, float]:
    values = (
        2.0 * (mass_scale - 1.0) / 0.35 - 1.0,
        2.0 * (friction - 0.45) / 0.35 - 1.0,
        2.0 * (motor_scale - 0.80) / 0.20 - 1.0,
    )
    if any(not math.isfinite(value) or value < -1.00001 or value > 1.00001 for value in values):
        raise ValueError(f"Context outside frozen normalized support: {values}")
    canonical: list[float] = []
    for value in values:
        if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            canonical.append(1.0)
        elif math.isclose(value, -1.0, rel_tol=0.0, abs_tol=1.0e-12):
            canonical.append(-1.0)
        else:
            canonical.append(min(1.0, max(-1.0, value)))
    return tuple(canonical)


def load_committed_agent_config(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import yaml

    head = git_text(repo_root, "rev-parse", "HEAD")
    if head != EXPECTED_GIT_HEAD:
        raise RuntimeError(f"Unexpected LEAP HEAD: expected={EXPECTED_GIT_HEAD}, actual={head}")
    object_name = f"HEAD:{AGENT_YAML_RELATIVE.as_posix()}"
    committed_bytes = git_bytes(repo_root, "show", object_name)
    parsed = yaml.safe_load(committed_bytes.decode("utf-8", errors="strict"))
    if not isinstance(parsed, dict):
        raise RuntimeError("Committed agent YAML did not parse to a mapping")
    scalar_normalization = normalize_committed_yaml_scalars(parsed)
    contract = validate_frozen_agent_config(parsed)
    working_path = repo_root / AGENT_YAML_RELATIVE
    working_hash = sha256_file(working_path)
    return parsed, {
        "source": object_name,
        "git_head": head,
        "committed_sha256": sha256_bytes(committed_bytes),
        "working_tree_path": str(working_path),
        "working_tree_sha256_audit_only": working_hash,
        "working_tree_consumed_as_config": False,
        "scalar_normalization": scalar_normalization,
        "frozen_contract": contract,
    }


def audit_protected_files(repo_root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        working_path = repo_root / relative
        if not working_path.is_file():
            raise FileNotFoundError(working_path)
        committed = git_bytes(repo_root, "show", f"HEAD:{relative.as_posix()}")
        expected_blob = git_text(repo_root, "rev-parse", f"HEAD:{relative.as_posix()}")
        actual_blob = git_text(repo_root, "hash-object", "--", relative.as_posix())
        if actual_blob != expected_blob:
            raise RuntimeError(f"Protected file differs from HEAD: {relative.as_posix()}")
        records[relative.as_posix()] = {
            "expected_git_blob": expected_blob,
            "actual_git_blob": actual_blob,
            "committed_sha256": sha256_bytes(committed),
            "working_tree_raw_sha256": sha256_file(working_path),
            "match_after_git_worktree_normalization": True,
        }
    return records


def tracked_diff(repo_root: Path) -> list[str]:
    output = git_text(repo_root, "diff", "--name-only")
    return [] if not output else output.splitlines()


def preflight(repo_root: Path) -> dict[str, Any]:
    source = repo_root / SOURCE_CHECKPOINT_RELATIVE
    expanded = repo_root / EXPANDED_CHECKPOINT_RELATIVE
    if not source.is_file() or not expanded.is_file():
        raise FileNotFoundError("Frozen source or expanded checkpoint is missing")
    source_hash = sha256_file(source)
    expanded_hash = sha256_file(expanded)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Source checkpoint SHA mismatch: {source_hash}")
    if expanded_hash != EXPECTED_EXPANDED_SHA256:
        raise RuntimeError(f"Expanded checkpoint SHA mismatch: {expanded_hash}")
    protected = audit_protected_files(repo_root)
    diffs = tracked_diff(repo_root)
    allowed_dirty = AGENT_YAML_RELATIVE.as_posix()
    if diffs != [allowed_dirty]:
        raise RuntimeError(f"Unexpected tracked working-tree diff: {diffs}")
    return {
        "source_checkpoint": {"path": str(source), "sha256": source_hash},
        "expanded_checkpoint": {"path": str(expanded), "sha256": expanded_hash},
        "protected_files": protected,
        "tracked_diff": diffs,
        "only_preexisting_yaml_diff": True,
    }


def relevant_processes(repo_root: Path) -> list[dict[str, Any]]:  # 只读枚举可能占用 Isaac/Python 的进程。
    try:
        import psutil
    except Exception as exc:
        return [{"inspection_available": False, "error": f"{type(exc).__name__}: {exc}"}]
    records: list[dict[str, Any]] = []
    repo_token = str(repo_root).lower()
    smoke_token = str(Path(__file__).resolve()).lower()
    ancestor_pids = {parent.pid for parent in psutil.Process(os.getpid()).parents()}  # 当前 isaaclab.bat/kit 启动链。
    for process in psutil.process_iter(["pid", "name", "exe", "cmdline", "create_time"]):
        try:
            info = process.info
            name = str(info.get("name") or "")
            if not any(token in name.lower() for token in ("python", "kit", "isaac")):
                continue
            command = " ".join(str(value) for value in (info.get("cmdline") or []))
            executable = str(info.get("exe") or "")
            combined = f"{command} {executable}".lower()
            records.append(
                {
                    "inspection_available": True,
                    "pid": int(info["pid"]),
                    "name": name,
                    "executable": executable,
                    "command_line": command,
                    "create_time": float(info.get("create_time") or 0.0),
                    "is_current_worker": int(info["pid"]) == os.getpid(),
                    "is_current_launch_ancestor": int(info["pid"]) in ancestor_pids and smoke_token in combined,
                    "project_related": repo_token in combined or smoke_token in combined,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return records


def confirmed_project_residuals(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:  # 排除当前 worker 与其专属启动祖先。
    return [
        process
        for process in processes
        if process.get("inspection_available", True)
        and process.get("project_related")
        and not process.get("is_current_worker")
        and not process.get("is_current_launch_ancestor")
    ]


def update_preflight_system_state(
    output_root: Path,
    branch: str,
    repo_root: Path,
    checkpoint_record: dict[str, Any],
) -> dict[str, Any]:  # 聚合每个 fresh worker 的 Stage 0 clean gate。
    path = output_root / "preflight_system_state.json"
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "smoke_id": SMOKE_ID,
        "schema_version": 1,
        "branch_preflights": {},
        "scientific_conclusion_generated": False,
    }
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    processes = relevant_processes(repo_root)
    residuals = confirmed_project_residuals(processes)
    record = {
        "timestamp_utc": utc_now(),
        "branch": branch,
        "current_worker_pid": os.getpid(),
        "processes": processes,
        "confirmed_project_residuals": residuals,
        "clean_process_gate_pass": not residuals,
        "resources": collect_resource_snapshot(),
        "git_head": git_text(repo_root, "rev-parse", "HEAD"),
        "tracked_diff": tracked_diff(repo_root),
        "checkpoint_hashes": {
            "source": checkpoint_record["source_checkpoint"]["sha256"],
            "expanded_99d": checkpoint_record["expanded_checkpoint"]["sha256"],
        },
    }
    payload["branch_preflights"][branch] = record
    payload["overall_pass"] = all(
        bool(item.get("clean_process_gate_pass")) for item in payload["branch_preflights"].values()
    )
    atomic_write_json(path, payload)
    return record


def model_state_hash(state_dict: dict[str, Any], torch_module: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not torch_module.is_tensor(value):
            raise TypeError(f"Unexpected non-tensor model state: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compare_model_to_checkpoint(model: Any, checkpoint_model: dict[str, Any], torch_module: Any) -> None:
    actual = model.state_dict()
    if set(actual) != set(checkpoint_model):
        raise RuntimeError(
            f"Model/checkpoint key mismatch: missing={sorted(set(checkpoint_model)-set(actual))}, "
            f"unexpected={sorted(set(actual)-set(checkpoint_model))}"
        )
    mismatches = [
        name
        for name in actual
        if tuple(actual[name].shape) != tuple(checkpoint_model[name].shape)
        or actual[name].dtype != checkpoint_model[name].dtype
        or not torch_module.equal(actual[name].detach().cpu(), checkpoint_model[name].detach().cpu())
    ]
    if mismatches:
        raise RuntimeError(f"Strict model-only initialization mismatch: {mismatches}")


def make_context_adapter_class(torch_module: Any, gym_module: Any, numpy_module: Any):
    class O0ContextVecEnvAdapter:
        """Narrow policy-facing 96+3 adapter with direct-readback lineage."""

        def __init__(self, base_env: Any, branch: str, clip_observations: float):
            self.base_env = base_env
            self.branch = branch
            self._clip_observations = clip_observations
            self._mass_reference = None
            self._effort_reference = None
            self.reset_calls = 0
            self.step_calls = 0
            self.readback_calls = 0
            self.context_rows_checked = 0
            self.input_shapes: set[tuple[int, ...]] = set()
            self.first_raw_context: list[float] | None = None
            self.max_d0_context_error = 0.0
            self.factor_assignment_audit_pass = False
            self.actual_availability_audit_pass = False
            self.combine_mode_audit_pass = False

        @property
        def unwrapped(self):
            return self.base_env.unwrapped

        @property
        def observation_space(self):
            return gym_module.spaces.Box(
                -self._clip_observations,
                self._clip_observations,
                (POLICY_INPUT_DIM,),
                dtype=numpy_module.float32,
            )

        @property
        def action_space(self):
            return self.base_env.action_space

        @property
        def state_space(self):
            return self.base_env.state_space

        @property
        def num_envs(self):
            return self.base_env.num_envs

        def get_number_of_agents(self):
            return self.base_env.get_number_of_agents()

        def get_env_info(self):
            info = dict(self.base_env.get_env_info())
            info["observation_space"] = self.observation_space
            return info

        def seed(self, seed: int = -1):
            return self.base_env.seed(seed)

        def set_train_info(self, env_frames: int, *args: Any, **kwargs: Any):
            method = getattr(self.base_env, "set_train_info", None)
            return method(env_frames, *args, **kwargs) if callable(method) else None

        def get_env_state(self):
            method = getattr(self.base_env, "get_env_state", None)
            return method() if callable(method) else None

        def set_env_state(self, env_state: Any):
            method = getattr(self.base_env, "set_env_state", None)
            return method(env_state) if callable(method) else None

        def has_action_masks(self):
            method = getattr(self.base_env, "has_action_masks", None)
            return bool(method()) if callable(method) else False

        def reset(self):
            packet = self.base_env.reset()
            self.reset_calls += 1
            return self._adapt(packet)

        def step(self, actions: Any):
            packet, rewards, dones, extras = self.base_env.step(actions)
            self.step_calls += 1
            return self._adapt(packet), rewards, dones, extras

        def close(self):
            return self.base_env.close()

        def _direct_context(self):
            snapshot = self.unwrapped.get_factor_snapshot()
            self.readback_calls += 1
            required_availability = ("mass", "object_material", "effort_limit", "combine_mode")
            availability = snapshot["actual_availability"]
            if not all(bool(availability[name].all().item()) for name in required_availability):
                raise RuntimeError("Direct factor read-back is unavailable after reset/step")
            self.actual_availability_audit_pass = True

            actual_mass = snapshot["actual_object_mass"]
            actual_friction = snapshot["actual_object_static_friction"]
            actual_effort = snapshot["actual_effort_limit"]
            if tuple(actual_mass.shape) != (NUM_ENVS, 1):
                raise RuntimeError(f"Unexpected actual mass shape: {tuple(actual_mass.shape)}")
            if tuple(actual_friction.shape) != (NUM_ENVS,):
                raise RuntimeError(f"Unexpected actual friction shape: {tuple(actual_friction.shape)}")
            if tuple(actual_effort.shape) != (NUM_ENVS, ACTION_DIM):
                raise RuntimeError(f"Unexpected actual effort shape: {tuple(actual_effort.shape)}")
            for tensor, name in (
                (actual_mass, "actual_object_mass"),
                (actual_friction, "actual_object_static_friction"),
                (actual_effort, "actual_effort_limit"),
            ):
                if not bool(torch_module.isfinite(tensor).all().item()):
                    raise RuntimeError(f"Non-finite direct read-back: {name}")

            torch_module.testing.assert_close(
                actual_mass, snapshot["requested_object_mass"], atol=1.0e-6, rtol=1.0e-5
            )
            torch_module.testing.assert_close(
                actual_friction, snapshot["requested_object_static_friction"], atol=1.0e-6, rtol=0.0
            )
            torch_module.testing.assert_close(
                actual_effort, snapshot["requested_effort_limit"], atol=1.0e-6, rtol=0.0
            )

            codes = tuple(snapshot["factor_codes"])
            self.factor_assignment_audit_pass = len(codes) == NUM_ENVS and set(codes) == {"000"}
            if not self.factor_assignment_audit_pass:
                raise RuntimeError("D0 smoke requires all factor assignments to be 000")
            self.combine_mode_audit_pass = (
                set(snapshot["actual_hand_combine_mode"]) == {"min"}
                and set(snapshot["actual_object_combine_mode"]) == {"min"}
            )
            if not self.combine_mode_audit_pass:
                raise RuntimeError("D0 smoke requires direct combine-mode read-back min/min")

            if self._mass_reference is None:
                reference = actual_mass[:, 0].mean().detach()
                if float((actual_mass[:, 0] - reference).abs().max().item()) > 1.0e-6:
                    raise RuntimeError("D0 actual mass is not uniform across environments")
                effort_reference = actual_effort.mean(dim=0).detach()
                if float((actual_effort - effort_reference).abs().max().item()) > 1.0e-6:
                    raise RuntimeError("D0 actual effort limits are not uniform across environments")
                self._mass_reference = reference
                self._effort_reference = effort_reference

            mass_scale = actual_mass[:, 0] / self._mass_reference
            motor_ratios = actual_effort / self._effort_reference
            if float((motor_ratios - motor_ratios.mean(dim=1, keepdim=True)).abs().max().item()) > 1.0e-6:
                raise RuntimeError("Per-joint motor-capacity scale is not uniform")
            motor_scale = motor_ratios.mean(dim=1)
            context = torch_module.stack(
                (
                    2.0 * (mass_scale - 1.0) / 0.35 - 1.0,
                    2.0 * (actual_friction - 0.45) / 0.35 - 1.0,
                    2.0 * (motor_scale - 0.80) / 0.20 - 1.0,
                ),
                dim=-1,
            ).clamp(-1.0, 1.0)
            expected = torch_module.tensor(
                EXPECTED_D0_CONTEXT, dtype=context.dtype, device=context.device
            ).expand_as(context)
            error = float((context - expected).abs().max().item())
            self.max_d0_context_error = max(self.max_d0_context_error, error)
            if error > 1.0e-5:
                raise RuntimeError(f"Unexpected D0 direct-readback context; max error={error}")
            self.context_rows_checked += int(context.shape[0])
            return context

        def _adapt(self, packet: Any):
            if not isinstance(packet, dict) or set(packet) - {"obs"}:
                raise RuntimeError(f"O0 smoke forbids asymmetric/extra observation keys: {list(packet)}")
            observation = packet.get("obs")
            if not torch_module.is_tensor(observation) or tuple(observation.shape) != (NUM_ENVS, OBSERVATION_DIM):
                raise RuntimeError(f"Expected legal observation [{NUM_ENVS},{OBSERVATION_DIM}], got {getattr(observation, 'shape', None)}")
            direct = self._direct_context()
            if self.branch == "history_only":
                appended = torch_module.zeros_like(direct)
            elif self.branch == "oracle":
                appended = direct
            else:
                raise RuntimeError(f"Unsupported branch: {self.branch}")
            if self.first_raw_context is None:
                self.first_raw_context = [float(value) for value in appended[0].detach().cpu().tolist()]
            policy_input = torch_module.cat((observation, appended), dim=-1)
            self.input_shapes.add(tuple(policy_input.shape))
            if tuple(policy_input.shape) != (NUM_ENVS, POLICY_INPUT_DIM):
                raise RuntimeError(f"Policy input contract mismatch: {tuple(policy_input.shape)}")
            if self.branch == "history_only" and int(torch_module.count_nonzero(policy_input[:, 96:99]).item()) != 0:
                raise RuntimeError("History-only context slot is not exact zero")
            return {"obs": policy_input}

        def evidence(self) -> dict[str, Any]:
            return {
                "branch": self.branch,
                "source_fields": [
                    "actual_object_mass",
                    "actual_object_static_friction",
                    "actual_effort_limit",
                ],
                "prohibited_fields_consumed": [],
                "factor_code_used_for_policy_input": False,
                "requested_values_used_for_policy_input": False,
                "reset_calls": self.reset_calls,
                "step_calls": self.step_calls,
                "readback_calls": self.readback_calls,
                "context_rows_checked": self.context_rows_checked,
                "input_shapes": [list(shape) for shape in sorted(self.input_shapes)],
                "first_raw_context": self.first_raw_context,
                "expected_raw_context": [0.0, 0.0, 0.0]
                if self.branch == "history_only"
                else list(EXPECTED_D0_CONTEXT),
                "max_d0_direct_context_error": self.max_d0_context_error,
                "actual_availability_pass": self.actual_availability_audit_pass,
                "factor_assignment_audit_pass": self.factor_assignment_audit_pass,
                "combine_mode_audit_pass": self.combine_mode_audit_pass,
            }

    return O0ContextVecEnvAdapter


def run_runtime(
    args: argparse.Namespace,
    app_launcher: Any,
    preflight_record: dict[str, Any],
    timeline: AllocationTimeline,
) -> dict[str, Any]:  # 保持训练语义，仅在冻结边界读取资源。
    import gymnasium as gymnasium
    import numpy as np
    import torch
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    import LEAP_Isaaclab.tasks  # noqa: F401

    timeline.snapshot_provider = lambda: collect_resource_snapshot(torch)

    repo_root = args.repo_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    branch_dir = output_root / args.branch
    agent_cfg, config_source = load_committed_agent_config(repo_root)
    realized = copy.deepcopy(agent_cfg)
    realized["params"]["seed"] = SEED
    realized_config = realized["params"]["config"]
    realized_config["num_actors"] = NUM_ENVS
    realized_config["max_epochs"] = args.max_iterations
    realized_config["train_dir"] = str(branch_dir / "rl_games")
    realized_config["full_experiment_name"] = "run"
    realized["params"]["load_checkpoint"] = False
    realized["params"]["load_path"] = ""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    env_cfg = parse_env_cfg(TASK_ID, device=app_launcher.device, num_envs=NUM_ENVS)
    env_cfg.seed = SEED
    env_cfg.factor_codes = ("000",)
    if env_cfg.events is not None or env_cfg.enable_adr is not False:
        raise RuntimeError("D0 smoke requires events=None and ADR disabled")
    if not math.isclose(float(env_cfg.sim.dt), PHYSICS_DT, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"physics dt changed: {env_cfg.sim.dt}")
    if int(env_cfg.decimation) != DECIMATION:
        raise RuntimeError(f"decimation changed: {env_cfg.decimation}")

    raw_env = None
    wrapped_env = None
    adapter = None
    agent = None
    forward_handle = None
    gradient_handle = None
    counters = {
        "forward_calls": 0,
        "forward_99d_calls": 0,
        "gradient_hook_calls": 0,
        "finite_gradient_calls": 0,
        "nonzero_gradient_calls": 0,
        "context_column_nonzero_gradient_calls": 0,
        "max_gradient_norm": 0.0,
        "max_context_column_gradient_norm": 0.0,
        "optimizer_step_count": 0,
    }
    network_input_shapes: set[tuple[int, ...]] = set()
    start = time.perf_counter()
    try:
        raw_env = gymnasium.make(TASK_ID, cfg=env_cfg)
        timeline.record(2)
        clip_obs = float(realized["params"]["env"]["clip_observations"])
        clip_actions = float(realized["params"]["env"]["clip_actions"])
        wrapped_env = RlGamesVecEnvWrapper(raw_env, realized_config["device"], clip_obs, clip_actions)
        if tuple(wrapped_env.observation_space.shape) != (OBSERVATION_DIM,):
            raise RuntimeError(f"Base observation is not 96-D: {wrapped_env.observation_space.shape}")
        if tuple(wrapped_env.action_space.shape) != (ACTION_DIM,):
            raise RuntimeError(f"Action space is not 16-D: {wrapped_env.action_space.shape}")
        timeline.record(3)
        Adapter = make_context_adapter_class(torch, __import__("gym"), np)
        adapter = Adapter(wrapped_env, args.branch, clip_obs)
        if tuple(adapter.observation_space.shape) != (POLICY_INPUT_DIM,):
            raise RuntimeError("Adapter did not advertise 99-D observation space")
        timeline.record(4)

        vecenv.register(
            "R7O0SmokeIsaacWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "R7O0SmokeIsaacWrapper", "env_creator": lambda **kwargs: adapter},
        )
        runner = Runner(IsaacAlgoObserver())
        runner.load(realized)
        timeline.record(5)
        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        timeline.record(6)

        checkpoint_path = repo_root / EXPANDED_CHECKPOINT_RELATIVE
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent.model.load_state_dict(checkpoint["model"], strict=True)
        compare_model_to_checkpoint(agent.model, checkpoint["model"], torch)
        trainable_count = sum(int(parameter.numel()) for parameter in agent.model.parameters() if parameter.requires_grad)
        state_count = sum(int(tensor.numel()) for tensor in agent.model.state_dict().values())
        if trainable_count != EXPECTED_TRAINABLE_PARAMETERS:
            raise RuntimeError(f"Trainable parameter mismatch: {trainable_count}")
        if state_count != EXPECTED_STATE_TENSOR_ELEMENTS:
            raise RuntimeError(f"State tensor element mismatch: {state_count}")
        if agent.has_central_value or realized["params"]["network"]["separate"] is not False:
            raise RuntimeError("Smoke requires one shared actor-critic input and no central-value network")
        if len(agent.optimizer.state) != 0:
            raise RuntimeError("Fresh optimizer unexpectedly has state before smoke update")

        initial_model_hash = model_state_hash(agent.model.state_dict(), torch)
        timeline.record(
            7,
            {
                "strict_checkpoint_load": True,
                "trainable_parameter_count": trainable_count,
                "state_tensor_element_count": state_count,
                "model_hash_before": initial_model_hash,
            },
        )

        def forward_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:  # 首次 forward 返回后记录 Stage 10。
            del module
            del output
            counters["forward_calls"] += 1
            if not inputs or not isinstance(inputs[0], dict) or "obs" not in inputs[0]:
                return
            observation = inputs[0]["obs"]
            if torch.is_tensor(observation):
                network_input_shapes.add(tuple(observation.shape))
                if observation.shape[-1] == POLICY_INPUT_DIM:
                    counters["forward_99d_calls"] += 1
            timeline.record(10, {"forward_calls": counters["forward_calls"]})

        forward_handle = agent.model.register_forward_hook(forward_hook)
        named_parameters = dict(agent.model.named_parameters())
        if GRU_INPUT_PARAMETER not in named_parameters:
            raise RuntimeError(f"Gradient-hook parameter is missing: {GRU_INPUT_PARAMETER}")

        def gradient_hook(gradient: Any):
            counters["gradient_hook_calls"] += 1
            finite = bool(torch.isfinite(gradient).all().item())
            if finite:
                counters["finite_gradient_calls"] += 1
            norm = float(gradient.norm().item())
            context_norm = float(gradient[:, 96:99].norm().item())
            counters["max_gradient_norm"] = max(counters["max_gradient_norm"], norm)
            counters["max_context_column_gradient_norm"] = max(
                counters["max_context_column_gradient_norm"], context_norm
            )
            if finite and norm > 0.0:
                counters["nonzero_gradient_calls"] += 1
            if finite and context_norm > 0.0:
                counters["context_column_nonzero_gradient_calls"] += 1
            timeline.record(
                11,
                {
                    "gradient_hook_calls": counters["gradient_hook_calls"],
                    "finite": finite,
                    "gradient_norm": norm,
                    "context_column_gradient_norm": context_norm,
                },
            )
            return gradient

        gradient_handle = named_parameters[GRU_INPUT_PARAMETER].register_hook(gradient_hook)
        original_optimizer_step = agent.optimizer.step

        def counted_optimizer_step(*step_args: Any, **step_kwargs: Any):
            result = original_optimizer_step(*step_args, **step_kwargs)
            counters["optimizer_step_count"] += 1
            timeline.record(12, {"optimizer_step_count": counters["optimizer_step_count"]})
            return result

        agent.optimizer.step = counted_optimizer_step
        agent.init_tensors()
        timeline.record(8)
        agent.mean_rewards = agent.last_mean_rewards = -100500
        agent.obs = agent.env_reset()
        timeline.record(9, {"adapter_reset_calls": adapter.reset_calls})
        for _ in range(args.max_iterations):
            agent.update_epoch()
            agent.train_epoch()
            agent.dataset.update_values_dict(None)
            frames = int(agent.curr_frames * agent.world_size if agent.multi_gpu else agent.curr_frames)
            agent.frame += frames
        torch.cuda.synchronize()

        final_model_hash = model_state_hash(agent.model.state_dict(), torch)
        optimizer_state_after = len(agent.optimizer.state)
        expected_raw = [0.0, 0.0, 0.0] if args.branch == "history_only" else list(EXPECTED_D0_CONTEXT)
        context_evidence = adapter.evidence()
        context_matches = context_evidence["first_raw_context"] is not None and all(
            abs(actual - expected) <= 1.0e-6
            for actual, expected in zip(context_evidence["first_raw_context"], expected_raw)
        )
        pass_checks = {
            "policy_input_99d": counters["forward_99d_calls"] > 0
            and all(shape[-1] == POLICY_INPUT_DIM for shape in network_input_shapes),
            "forward_positive": counters["forward_calls"] > 0,
            "gradient_hook_positive": counters["gradient_hook_calls"] > 0,
            "finite_nonzero_gradient": counters["nonzero_gradient_calls"] > 0
            and counters["finite_gradient_calls"] == counters["gradient_hook_calls"],
            "oracle_context_column_gradient": args.branch != "oracle"
            or counters["context_column_nonzero_gradient_calls"] > 0,
            "optimizer_step_positive": counters["optimizer_step_count"] > 0,
            "optimizer_state_populated": optimizer_state_after > 0,
            "model_hash_changed": final_model_hash != initial_model_hash,
            "parameter_count_exact": trainable_count == EXPECTED_TRAINABLE_PARAMETERS,
            "state_tensor_count_exact": state_count == EXPECTED_STATE_TENSOR_ELEMENTS,
            "context_injection_exact": context_matches,
            "adapter_readback_positive": context_evidence["readback_calls"] > 0,
            "direct_readback_available": context_evidence["actual_availability_pass"],
            "d0_assignment_audited": context_evidence["factor_assignment_audit_pass"],
            "combine_mode_audited": context_evidence["combine_mode_audit_pass"],
            "requested_values_not_policy_input": not context_evidence["requested_values_used_for_policy_input"],
            "no_trained_checkpoint_written": not list(branch_dir.rglob("*.pth")),
        }
        status = "PASS" if all(pass_checks.values()) else "BLOCKED"
        return {
            "protocol_id": PROTOCOL_ID,
            "smoke_id": SMOKE_ID,
            "schema_version": 1,
            "status": status,
            "branch": args.branch,
            "process_id": os.getpid(),
            "timestamp_utc": utc_now(),
            "scope": "D0_ONLY_PPO_PIPELINE_SMOKE_NOT_PERFORMANCE",
            "scientific_conclusion_generated": False,
            "performance_metrics_computed": False,
            "h1_or_a0_executed": False,
            "seed": SEED,
            "condition": "D0",
            "factor_assignment": "000 for all environments; audit only, never policy input",
            "num_envs": NUM_ENVS,
            "iterations_requested": args.max_iterations,
            "iterations_completed": int(agent.epoch_num),
            "frames_completed": int(agent.frame),
            "physics_dt": PHYSICS_DT,
            "decimation": DECIMATION,
            "input_dimension": POLICY_INPUT_DIM,
            "legal_observation_dimension": OBSERVATION_DIM,
            "context_dimension": CONTEXT_DIM,
            "action_dimension": ACTION_DIM,
            "trainable_parameter_count": trainable_count,
            "state_tensor_element_count": state_count,
            "checkpoint": preflight_record,
            "config_source": config_source,
            "realized_training_contract": {
                "horizon_length": int(agent.horizon_length),
                "minibatch_size": int(agent.minibatch_size),
                "mini_epochs": int(agent.mini_epochs_num),
                "seq_length": int(agent.seq_length),
                "batch_size": int(agent.batch_size),
                "working_tree_yaml_consumed": False,
                "legacy_optimizer_state_loaded": False,
                "automatic_checkpoint_save_path_used": False,
            },
            "context_injection": context_evidence,
            "network_input_shapes": [list(shape) for shape in sorted(network_input_shapes)],
            "training_counters": counters,
            "optimizer_state_entries_before": 0,
            "optimizer_state_entries_after": optimizer_state_after,
            "model_hash_before": initial_model_hash,
            "model_hash_after": final_model_hash,
            "checks": pass_checks,
            "allocation_completed_stages": sorted(timeline.completed_stage_ids),
            "allocation_timeline_path": str(timeline.path),
            "elapsed_seconds": time.perf_counter() - start,
        }
    finally:
        if gradient_handle is not None:
            gradient_handle.remove()
        if forward_handle is not None:
            forward_handle.remove()
        if agent is not None and getattr(agent, "writer", None) is not None:
            agent.writer.close()
        if wrapped_env is not None:
            wrapped_env.close()
        elif raw_env is not None:
            raw_env.close()


def fairness_from_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def context_matches(actual: Sequence[float], expected: Sequence[float]) -> bool:
        return len(actual) == len(expected) and all(
            math.isclose(float(observed), float(target), rel_tol=0.0, abs_tol=1.0e-6)
            for observed, target in zip(actual, expected)
        )

    complete = all(branch in results for branch in BRANCHES)
    checks: dict[str, bool] = {
        "both_branches_present": complete,
        "both_branches_pass": complete and all(results[name].get("status") == "PASS" for name in BRANCHES),
    }
    if complete:
        history = results["history_only"]
        oracle = results["oracle"]
        equality_fields = (
            "seed",
            "condition",
            "num_envs",
            "iterations_requested",
            "physics_dt",
            "decimation",
            "input_dimension",
            "legal_observation_dimension",
            "context_dimension",
            "action_dimension",
            "trainable_parameter_count",
            "state_tensor_element_count",
            "model_hash_before",
        )
        checks.update({f"equal_{field}": history.get(field) == oracle.get(field) for field in equality_fields})
        checks["same_expanded_checkpoint"] = (
            history["checkpoint"]["expanded_checkpoint"]["sha256"]
            == oracle["checkpoint"]["expanded_checkpoint"]["sha256"]
            == EXPECTED_EXPANDED_SHA256
        )
        checks["same_committed_config"] = (
            history["config_source"]["committed_sha256"] == oracle["config_source"]["committed_sha256"]
        )
        checks["separate_processes"] = history.get("process_id") != oracle.get("process_id")
        checks["working_yaml_never_consumed"] = not history["config_source"]["working_tree_consumed_as_config"] and not oracle["config_source"]["working_tree_consumed_as_config"]
        checks["history_context_zero"] = context_matches(
            history["context_injection"]["first_raw_context"], (0.0, 0.0, 0.0)
        )
        checks["oracle_context_d0_actual"] = context_matches(
            oracle["context_injection"]["first_raw_context"], EXPECTED_D0_CONTEXT
        )
        checks["only_context_payload_differs"] = checks["history_context_zero"] and checks["oracle_context_d0_actual"]
    overall = complete and all(checks.values())
    return {
        "protocol_id": PROTOCOL_ID,
        "smoke_id": SMOKE_ID,
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "status": "PASS" if overall else ("INCOMPLETE" if not complete else "BLOCKED"),
        "overall_pass": overall,
        "checks": checks,
        "allowed_branch_difference": "indices 96:99 context values only",
        "scientific_fairness_claim": False,
        "scope": "smoke-pipeline parity only",
    }


def collect_artifact_hashes(output_root: Path, checksum_path: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path != checksum_path
    }


def refresh_root_artifacts(output_root: Path) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        path = output_root / branch / "branch_result.json"
        if path.is_file():
            results[branch] = json.loads(path.read_text(encoding="utf-8"))
    fairness = fairness_from_results(results)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "smoke_id": SMOKE_ID,
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "status": "PASS" if fairness["overall_pass"] else fairness["status"],
        "scope": "D0-only PPO training-path smoke; not O0 performance evidence",
        "branches_present": sorted(results),
        "branch_results": results,
        "scientific_conclusion_generated": False,
        "performance_comparison_permitted": False,
    }
    checkpoint_hash = {
        "protocol_id": PROTOCOL_ID,
        "smoke_id": SMOKE_ID,
        "timestamp_utc": utc_now(),
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "expanded_checkpoint_sha256": EXPECTED_EXPANDED_SHA256,
        "branches": {
            branch: {
                "checkpoint_sha256": result.get("checkpoint", {}).get("expanded_checkpoint", {}).get("sha256"),
                "model_hash_before": result.get("model_hash_before"),
                "model_hash_after": result.get("model_hash_after"),
            }
            for branch, result in results.items()
        },
        "source_or_expanded_checkpoint_modified": False,
        "trained_checkpoint_written": False,
    }
    atomic_write_json(output_root / "training_manifest.json", manifest)
    atomic_write_json(output_root / "checkpoint_hash.json", checkpoint_hash)
    atomic_write_json(output_root / "fairness_report.json", fairness)
    branch_timelines: dict[str, Any] = {}
    for branch in BRANCHES:
        timeline_path = output_root / branch / "allocation_timeline.json"
        if timeline_path.is_file():
            branch_timelines[branch] = json.loads(timeline_path.read_text(encoding="utf-8"))
    atomic_write_json(
        output_root / "allocation_timeline.json",
        {
            "protocol_id": PROTOCOL_ID,
            "smoke_id": SMOKE_ID,
            "schema_version": 1,
            "branches": branch_timelines,
            "scientific_conclusion_generated": False,
        },
    )
    checksum_path = output_root / "artifact_checksums.json"
    artifact_hashes = collect_artifact_hashes(output_root, checksum_path)
    atomic_write_json(
        checksum_path,
        {
            "protocol_id": PROTOCOL_ID,
            "smoke_id": SMOKE_ID,
            "schema_version": 1,
            "artifacts": artifact_hashes,
            "scientific_conclusion_generated": False,
        },
    )
    return fairness


def main(argv: Sequence[str] | None = None) -> int:
    # Importing this module is inert. Isaac is imported and launched only here.
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    try:
        validate_cli_contract(args)
        repo_root = args.repo_root.expanduser().resolve(strict=True)
        output_root = args.output_root.expanduser().resolve(strict=False)
        if output_root != DEFAULT_OUTPUT_ROOT.resolve(strict=False):
            raise RuntimeError(f"Output root must be exactly {DEFAULT_OUTPUT_ROOT.resolve(strict=False)}")
        branch_dir = output_root / args.branch
        if branch_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing branch artifact: {branch_dir}")
        branch_dir.mkdir(parents=True, exist_ok=False)
        preflight_record = preflight(repo_root)
        timeline = AllocationTimeline(branch_dir / "allocation_timeline.json", args.branch)
        timeline.record(0)
        clean_gate = update_preflight_system_state(output_root, args.branch, repo_root, preflight_record)
        if not clean_gate["clean_process_gate_pass"]:
            raise RuntimeError(f"Clean process gate found project residuals: {clean_gate['confirmed_project_residuals']}")
    except Exception as exc:
        print(f"BLOCKED before Isaac launch: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    app_launcher = None
    result: dict[str, Any]
    return_code = 2
    try:
        app_launcher = AppLauncher(args)
        timeline.record(1)
        try:
            result = run_runtime(args, app_launcher, preflight_record, timeline)
        except Exception as exc:
            timeline.fail_first_incomplete(exc)
            result = {
                "protocol_id": PROTOCOL_ID,
                "smoke_id": SMOKE_ID,
                "schema_version": 1,
                "status": "BLOCKED",
                "branch": args.branch,
                "process_id": os.getpid(),
                "timestamp_utc": utc_now(),
                "scope": "D0_ONLY_PPO_PIPELINE_SMOKE_NOT_PERFORMANCE",
                "scientific_conclusion_generated": False,
                "performance_metrics_computed": False,
                "h1_or_a0_executed": False,
                "failure_classification": classify_runtime_failure(timeline.completed_stage_ids, exc),
                "allocation_completed_stages": sorted(timeline.completed_stage_ids),
                "allocation_timeline_path": str(timeline.path),
                "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                "checkpoint": preflight_record,
            }

        # Persist before closing SimulationApp: Kit shutdown can terminate the host process
        # before Python regains control on some Windows runtime configurations.
        result["source_checkpoint_sha256_after"] = sha256_file(repo_root / SOURCE_CHECKPOINT_RELATIVE)
        result["expanded_checkpoint_sha256_after"] = sha256_file(repo_root / EXPANDED_CHECKPOINT_RELATIVE)
        result["protected_files_after"] = audit_protected_files(repo_root)
        if result["source_checkpoint_sha256_after"] != EXPECTED_SOURCE_SHA256:
            result["status"] = "BLOCKED"
        if result["expanded_checkpoint_sha256_after"] != EXPECTED_EXPANDED_SHA256:
            result["status"] = "BLOCKED"
        atomic_write_json(branch_dir / "branch_result.json", result)
        fairness = refresh_root_artifacts(output_root)
        print(
            json.dumps({"branch": args.branch, "status": result["status"], "aggregate": fairness["status"]}, indent=2),
            flush=True,
        )
        return_code = 0 if result["status"] == "PASS" else 2
    finally:
        if app_launcher is not None:
            app_launcher.app.close()
    return return_code


if __name__ == "__main__":
    sys.exit(main())
