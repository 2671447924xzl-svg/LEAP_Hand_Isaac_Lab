"""R7-O0 正式训练入口、6144 容量预检与混合条件完整性门。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent  # 只定位同目录的只读审计工具。
SMOKE_AUDIT_PATH = SCRIPT_DIR / "r7_o0_smoke_training.py"  # 复用范围限于审计与遥测。
_AUDIT_SPEC = importlib.util.spec_from_file_location("r7_o0_smoke_audit_utilities", SMOKE_AUDIT_PATH)
if _AUDIT_SPEC is None or _AUDIT_SPEC.loader is None:
    raise ImportError(f"无法加载审计工具: {SMOKE_AUDIT_PATH}")
audit = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(audit)


PROTOCOL_ID = "R7_O0_CONTEXT_VALUE_V1"  # 冻结科学协议。
FAIRNESS_SCHEMA = "R7_O0_FAIRNESS_V1"  # 冻结公平性协议。
IMPLEMENTATION_ID = "R7_O0_PHASE42_FORMAL_DRIVER_V1"  # 本轮工程实现标识。
BRANCHES = ("history_only", "oracle")  # 唯一合法训练分支。
TRAINING_SEEDS = (7100, 7101, 7102, 7103, 7104)  # 冻结训练种子。
FORMAL_NUM_ENVS = 6144  # 正式单进程环境总数。
ENVS_PER_CONDITION = 2048  # 每个冻结条件的环境数。
FORMAL_CONDITIONS = ("000", "010", "100")  # D0、D1、D2 的冻结次序。
EXPECTED_FACTOR_COUNTS = {"000": 2048, "010": 2048, "100": 2048}  # 精确平衡合同。
LEGAL_OBSERVATION_DIM = 96  # 合法历史观测宽度。
CONTEXT_DIM = 3  # 公共 context slot 宽度。
POLICY_INPUT_DIM = 99  # 两分支共同网络输入宽度。
ACTION_DIM = 16  # 冻结动作宽度。
HORIZON_LENGTH = 32  # PPO rollout 长度。
MINIBATCH_SIZE = 32768  # 冻结 minibatch，不读取工作树覆盖。
MAX_EPOCHS = 5000  # 正式训练预算。
PHYSICS_DT = 1.0 / 120.0  # 冻结物理步长。
DECIMATION = 4  # 冻结控制降采样。
NOMINAL_OBJECT_MASS_KG = 0.216  # manifest 冻结的 actual mass 归一化基准。
NOMINAL_EFFORT_LIMIT = 0.50  # manifest 冻结的 actual effort 归一化基准。
EVALUATION_CHECKPOINT_RULE = "epoch_5000_final_only"  # 唯一评估 checkpoint 规则。
TASK_ID = "Isaac-Reorient-Cube-Leap-CRI"  # 独立 CRI 任务入口。
DEFAULT_REPO_ROOT = Path(r"D:\Research\LEAP\LEAP_Hand_Isaac_Lab")  # 工程根目录。
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "logs/r7_o0_formal_training"  # 正式 artifact 根。
AGENT_YAML_RELATIVE = Path(
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml"
)  # 只从 Git HEAD 读取。
SOURCE_CHECKPOINT_RELATIVE = Path("logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth")
EXPANDED_CHECKPOINT_RELATIVE = Path("logs/r7_o0_context_value/checkpoints/r7_o0_initialization_99d.pth")
EXPECTED_SOURCE_SHA256 = audit.EXPECTED_SOURCE_SHA256  # 复用冻结哈希常量。
EXPECTED_EXPANDED_SHA256 = audit.EXPECTED_EXPANDED_SHA256  # 复用冻结哈希常量。
EXPECTED_TRAINABLE_PARAMETERS = 572_705  # 冻结参数规模。
EXPECTED_STATE_TENSOR_ELEMENTS = 572_907  # 冻结 state tensor 规模。
CONTEXT_SOURCE_FIELDS = (
    "actual_object_mass",
    "actual_object_static_friction",
    "actual_effort_limit",
)  # Oracle 仅可使用 post-reset actual read-back。
PROHIBITED_POLICY_FIELDS = (
    "factor_code",
    "condition_id",
    "env_id",
    "requested_value",
    "seed",
    "reward",
    "object_state_truth",
    "contact_force",
    "future_information",
)  # 不得进入 actor 或 critic。
EXPECTED_ORACLE_CONTEXTS = {
    "000": (-1.0, 1.0, 1.0),
    "010": (-1.0, -1.0, 1.0),
    "100": (1.0, 1.0, 1.0),
}  # D0、D1、D2 的冻结归一化 context。
FORMAL_STAGES = {
    0: "before_app_launcher",
    1: "after_app_launcher",
    2: "after_6144_env_gym_make",
    3: "after_reset",
    4: "after_factor_context_audit",
    5: "after_99d_adapter",
    6: "after_runner_model_construction",
    7: "after_checkpoint_load_and_rollout_allocation",
}  # F0–F7 边界不可重排。
MIXED_PROBE_SAMPLE_PER_CONDITION = 8  # 只抽取少量环境进行非科研物理完整性探针。
MIXED_PROBE_SETTLE_STEPS = 30  # 固定 settling 物理步数。
MIXED_PROBE_FORCE_LEVELS_N = (0.5, 1.0, 1.5, 2.0)  # 预先冻结的 world-x 力阶梯。
MIXED_PROBE_HOLD_STEPS = 12  # 每个力阶梯的物理步数。
MIXED_PROBE_DIRECTION_EPS_M = 1.0e-6  # 只判方向，不形成科学效应阈值。
AUTHORIZATION_SCOPE = "R7_O0_FORMAL_5000_EPOCH_TRAINING"  # 下一轮授权文件必须精确匹配。
FORMAL_RUN_REQUIRED_FIELDS = (
    "protocol_id",
    "branch",
    "seed",
    "git_commit",
    "checkpoint_sha256",
    "realized_config_sha256",
    "factor_counts",
    "actual_context_distribution",
    "optimizer_fresh_state",
    "epoch_count",
    "frame_count",
    "final_checkpoint_sha256",
    "resource_telemetry",
    "completion_status",
)  # 每个未来正式 run 的统一 artifact schema。


def build_balanced_factor_codes() -> tuple[str, ...]:  # 构造一次性固定 assignment。
    return tuple(code for code in FORMAL_CONDITIONS for _ in range(ENVS_PER_CONDITION))


def audit_factor_counts(codes: Sequence[str]) -> dict[str, Any]:  # 拒绝缺失、多余或不平衡条件。
    counts = dict(sorted(Counter(codes).items()))
    passed = len(codes) == FORMAL_NUM_ENVS and counts == EXPECTED_FACTOR_COUNTS
    return {"pass": passed, "counts": counts, "expected": EXPECTED_FACTOR_COUNTS, "total": len(codes)}


def normalized_context(mass_scale: float, friction: float, motor_scale: float) -> tuple[float, float, float]:  # 冻结仿射映射。
    values = (
        2.0 * (float(mass_scale) - 1.0) / 0.35 - 1.0,
        2.0 * (float(friction) - 0.45) / 0.35 - 1.0,
        2.0 * (float(motor_scale) - 0.80) / 0.20 - 1.0,
    )
    if any(not math.isfinite(value) or value < -1.00001 or value > 1.00001 for value in values):
        raise ValueError(f"context 超出冻结 support: {values}")
    return tuple(1.0 if abs(value - 1.0) <= 1.0e-6 else -1.0 if abs(value + 1.0) <= 1.0e-6 else value for value in values)


def policy_context(branch: str, code: str, mass_scale: float, friction: float, motor_scale: float) -> tuple[float, float, float]:  # 纯函数合约。
    if branch == "history_only":
        return (0.0, 0.0, 0.0)
    if branch != "oracle" or code not in EXPECTED_ORACLE_CONTEXTS:
        raise ValueError(f"非法 branch/code: {branch}/{code}")
    context = normalized_context(mass_scale, friction, motor_scale)
    expected = EXPECTED_ORACLE_CONTEXTS[code]
    if any(abs(actual - target) > 1.0e-6 for actual, target in zip(context, expected)):
        raise RuntimeError(f"Oracle context 不符合 {code}: actual={context}, expected={expected}")
    return context


def expected_frozen_config_fields() -> dict[str, Any]:  # 返回需要逐项核对的 committed PPO 合同。
    return {
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
        "params.config.mini_epochs": 5,
        "params.config.seq_length": 4,
        "params.config.max_epochs": MAX_EPOCHS,
        "params.config.save_frequency": 200,
        "params.env.clip_observations": 5.0,
        "params.env.clip_actions": 1.0,
    }


def validate_flat_config(actual: dict[str, Any]) -> None:  # 精确拒绝任一冻结字段漂移。
    expected = expected_frozen_config_fields()
    mismatches = {key: {"expected": expected[key], "actual": actual.get(key)} for key in expected if actual.get(key) != expected[key]}
    if mismatches:
        raise RuntimeError(f"冻结 PPO 合同不匹配: {mismatches}")


def nested_value(mapping: dict[str, Any], dotted: str) -> Any:  # 读取嵌套 YAML 字段。
    current: Any = mapping
    for component in dotted.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(f"缺少冻结配置字段: {dotted}")
        current = current[component]
    return current


def load_committed_agent_config(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:  # 永不解析 dirty YAML。
    import yaml

    object_name = f"HEAD:{AGENT_YAML_RELATIVE.as_posix()}"
    committed_bytes = audit.git_bytes(repo_root, "show", object_name)
    config = yaml.safe_load(committed_bytes.decode("utf-8", errors="strict"))
    if isinstance(nested_value(config, "params.config.learning_rate"), str):
        config["params"]["config"]["learning_rate"] = float(config["params"]["config"]["learning_rate"])
    flat = {key: nested_value(config, key) for key in expected_frozen_config_fields()}
    validate_flat_config(flat)
    working_path = repo_root / AGENT_YAML_RELATIVE
    return config, {
        "source": object_name,
        "git_commit": audit.git_text(repo_root, "rev-parse", "HEAD"),
        "committed_sha256": audit.sha256_bytes(committed_bytes),
        "working_tree_sha256_audit_only": audit.sha256_file(working_path),
        "working_tree_consumed_as_config": False,
        "validated_fields": flat,
    }


def require_fresh_optimizer(state: dict[Any, Any]) -> None:  # model-only 初始化必须留下空 optimizer state。
    if len(state) != 0:
        raise RuntimeError(f"optimizer 不是 fresh state: entries={len(state)}")


def validate_cli_contract(args: argparse.Namespace) -> None:  # 所有模式共享同一 branch/seed 合同。
    if args.mode not in ("capacity-preflight", "train"):
        raise ValueError(f"非法 mode: {args.mode}")
    if args.branch not in BRANCHES:
        raise ValueError(f"非法 branch: {args.branch}")
    if args.seed not in TRAINING_SEEDS:
        raise ValueError(f"非法 training seed: {args.seed}")


def validate_training_authorization(path: Path | None, branch: str, seed: int, git_commit: str) -> dict[str, Any]:  # 缺省拒绝 5000 epochs。
    if path is None or not path.is_file():
        raise PermissionError("FORMAL_TRAINING_NOT_AUTHORIZED")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_id": PROTOCOL_ID,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "approved": True,
        "branch": branch,
        "seed": seed,
        "git_commit": git_commit,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise PermissionError("FORMAL_TRAINING_NOT_AUTHORIZED")
    return payload


def initialize_artifact_schema(output_root: Path) -> list[str]:  # 预建完整 formal run 目录树。
    relative_paths = ["capacity_preflight", "mixed_factor_integrity"]
    relative_paths.extend(f"{branch}/seed_{seed}" for branch in BRANCHES for seed in TRAINING_SEEDS)
    for relative in relative_paths:
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    return relative_paths


def final_decision(capacity_pass: bool, mixed_integrity_pass: bool) -> list[str]:  # Gate M 不得被 capacity 状态覆盖。
    if not capacity_pass:
        return ["FORMAL_6144_CAPACITY_BLOCKED", "HARDWARE_DECISION_REQUIRED"]
    if not mixed_integrity_pass:
        return ["FORMAL_6144_CAPACITY_PASS", "MIXED_FACTOR_INTEGRITY_BLOCKED", "SCIENTIFIC_PROTOCOL_DECISION_REQUIRED"]
    return ["FORMAL_6144_CAPACITY_PASS", "MIXED_FACTOR_INTEGRITY_PASS", "READY_FOR_FORMAL_O0_TRAINING_AUTHORIZATION"]


class FormalTimeline:  # F0–F7 每完成一阶段即原子落盘。
    def __init__(self, path: Path, snapshot_provider: Any = audit.collect_resource_snapshot):
        self.path = path
        self.snapshot_provider = snapshot_provider
        self.started = time.perf_counter()
        self.stages: list[dict[str, Any]] = []
        self._write()

    @property
    def completed_ids(self) -> set[int]:  # 返回已经落盘的阶段集合。
        return {int(item["stage_id"]) for item in self.stages if item["status"] == "COMPLETE"}

    def _write(self) -> None:  # 写入可在 OOM 后恢复的证据。
        audit.atomic_write_json(
            self.path,
            {
                "protocol_id": PROTOCOL_ID,
                "implementation_id": IMPLEMENTATION_ID,
                "schema_version": 1,
                "stages": self.stages,
                "elapsed_seconds": time.perf_counter() - self.started,
                "scientific_conclusion_generated": False,
            },
        )

    def record(self, stage_id: int, details: dict[str, Any] | None = None) -> None:  # 同一阶段只允许记录一次。
        if stage_id not in FORMAL_STAGES:
            raise ValueError(f"非法 formal stage: {stage_id}")
        if stage_id in self.completed_ids:
            return
        self.stages.append(
            {
                "stage_id": stage_id,
                "stage_name": FORMAL_STAGES[stage_id],
                "status": "COMPLETE",
                "elapsed_seconds": time.perf_counter() - self.started,
                "resources": self.snapshot_provider(),
                "details": details or {},
            }
        )
        self._write()

    def fail(self, exc: BaseException) -> None:  # 把异常归于首个未完成边界。
        next_stage = next((stage for stage in FORMAL_STAGES if stage not in self.completed_ids), None)
        self.stages.append(
            {
                "stage_id": next_stage,
                "stage_name": FORMAL_STAGES.get(next_stage, "after_f7"),
                "status": "FAILED",
                "elapsed_seconds": time.perf_counter() - self.started,
                "resources": self.snapshot_provider(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        self._write()


def build_parser() -> argparse.ArgumentParser:  # 训练授权文件默认不存在。
    parser = argparse.ArgumentParser(description="R7-O0 正式训练与 6144 capacity preflight")
    parser.add_argument("--mode", required=True, choices=("capacity-preflight", "train"))
    parser.add_argument("--branch", required=True, choices=BRANCHES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--authorization-file", type=Path, default=None)
    return parser


def formal_process_records(repo_root: Path) -> list[dict[str, Any]]:  # 识别当前 formal worker 与链外残留进程。
    try:
        import psutil
    except Exception as exc:
        return [{"inspection_available": False, "error": f"{type(exc).__name__}: {exc}"}]
    records: list[dict[str, Any]] = []
    formal_token = str(Path(__file__).resolve()).lower()
    repo_token = str(repo_root).lower()
    ancestor_pids = {parent.pid for parent in psutil.Process(os.getpid()).parents()}
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
                    "command_line": command,
                    "executable": executable,
                    "create_time": float(info.get("create_time") or 0.0),
                    "is_current_worker": int(info["pid"]) == os.getpid(),
                    "is_current_launch_ancestor": int(info["pid"]) in ancestor_pids and formal_token in combined,
                    "project_related": repo_token in combined or formal_token in combined,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return records


def clean_process_gate(repo_root: Path) -> dict[str, Any]:  # 单 GPU 上只接受当前 launch chain。
    processes = formal_process_records(repo_root)
    residuals = [
        item
        for item in processes
        if item.get("inspection_available", True)
        and item.get("project_related")
        and not item.get("is_current_worker")
        and not item.get("is_current_launch_ancestor")
    ]
    return {"pass": not residuals, "processes": processes, "residuals": residuals}


def create_protocol_manifest(repo_root: Path, config_record: dict[str, Any]) -> dict[str, Any]:  # 只描述预注册协议，不含结果。
    return {
        "protocol_id": PROTOCOL_ID,
        "fairness_schema": FAIRNESS_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "protocol_gap": False,
        "branches": list(BRANCHES),
        "training_seeds": list(TRAINING_SEEDS),
        "factor_assignment": {"order": list(FORMAL_CONDITIONS), "counts": EXPECTED_FACTOR_COUNTS},
        "num_envs": FORMAL_NUM_ENVS,
        "policy_input": {"legal_observation_dim": LEGAL_OBSERVATION_DIM, "context_dim": CONTEXT_DIM, "total_dim": POLICY_INPUT_DIM},
        "ppo": {"horizon_length": HORIZON_LENGTH, "minibatch_size": MINIBATCH_SIZE, "max_epochs": MAX_EPOCHS},
        "checkpoint_selection": EVALUATION_CHECKPOINT_RULE,
        "git_commit": audit.git_text(repo_root, "rev-parse", "HEAD"),
        "realized_config_source": config_record,
        "mixed_probe": {
            "scope": "engineering_integrity_only_not_scientific_evidence",
            "sample_per_condition": MIXED_PROBE_SAMPLE_PER_CONDITION,
            "settle_steps": MIXED_PROBE_SETTLE_STEPS,
            "force_levels_n": list(MIXED_PROBE_FORCE_LEVELS_N),
            "hold_steps": MIXED_PROBE_HOLD_STEPS,
            "direction_epsilon_m": MIXED_PROBE_DIRECTION_EPS_M,
        },
        "evaluation_executed": False,
        "scientific_conclusion_generated": False,
    }


def checkpoint_preflight(repo_root: Path) -> dict[str, Any]:  # 两个冻结 checkpoint 只读核验。
    source = repo_root / SOURCE_CHECKPOINT_RELATIVE
    expanded = repo_root / EXPANDED_CHECKPOINT_RELATIVE
    source_hash = audit.sha256_file(source)
    expanded_hash = audit.sha256_file(expanded)
    if source_hash != EXPECTED_SOURCE_SHA256 or expanded_hash != EXPECTED_EXPANDED_SHA256:
        raise RuntimeError(f"checkpoint 哈希不匹配: source={source_hash}, expanded={expanded_hash}")
    tracked = audit.tracked_diff(repo_root)
    if tracked != [AGENT_YAML_RELATIVE.as_posix()]:
        raise RuntimeError(f"存在未批准 tracked diff: {tracked}")
    return {
        "source": {"path": str(source), "sha256": source_hash},
        "expanded": {"path": str(expanded), "sha256": expanded_hash},
        "tracked_diff": tracked,
        "only_preexisting_yaml_diff": True,
    }


def tensor_digest(tensor: Any) -> str:  # 压缩记录大型逐环境 read-back。
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def snapshot_stability(before: dict[str, Any], after: dict[str, Any], torch_module: Any) -> dict[str, Any]:  # 验证 reset 不漂移。
    tensor_fields = (
        "factor_code_bits",
        "requested_object_mass",
        "actual_object_mass",
        "requested_object_inertia",
        "actual_object_inertia",
        "requested_object_static_friction",
        "actual_object_static_friction",
        "requested_effort_limit",
        "actual_effort_limit",
    )
    checks = {field: bool(torch_module.allclose(before[field], after[field], atol=1.0e-6, rtol=1.0e-5)) for field in tensor_fields}
    checks["factor_codes"] = tuple(before["factor_codes"]) == tuple(after["factor_codes"])
    checks["hand_combine_mode"] = tuple(before["actual_hand_combine_mode"]) == tuple(after["actual_hand_combine_mode"])
    checks["object_combine_mode"] = tuple(before["actual_object_combine_mode"]) == tuple(after["actual_object_combine_mode"])
    return {"pass": all(checks.values()), "checks": checks}


def audit_mixed_snapshot(snapshot: dict[str, Any], core: Any, torch_module: Any) -> dict[str, Any]:  # 审计全部 6144 个 actual 值。
    from isaaclab.sim.utils.stage import get_current_stage

    codes = tuple(snapshot["factor_codes"])
    count_report = audit_factor_counts(codes)
    if not count_report["pass"]:
        raise RuntimeError(f"mixed factor counts 不匹配: {count_report}")
    required_availability = ("mass", "inertia", "object_material", "hand_material", "effort_limit", "combine_mode")
    availability = {field: bool(snapshot["actual_availability"][field].all().item()) for field in required_availability}
    if not all(availability.values()):
        raise RuntimeError(f"actual read-back 不完整: {availability}")
    requested_actual_fields = (
        ("requested_object_mass", "actual_object_mass", 1.0e-6, 1.0e-5),
        ("requested_object_inertia", "actual_object_inertia", 1.0e-6, 1.0e-5),
        ("requested_object_static_friction", "actual_object_static_friction", 1.0e-6, 0.0),
        ("requested_object_dynamic_friction", "actual_object_dynamic_friction", 1.0e-6, 0.0),
        ("requested_effort_limit", "actual_effort_limit", 1.0e-6, 0.0),
    )
    matches: dict[str, bool] = {}
    for requested, actual_name, atol, rtol in requested_actual_fields:
        matches[actual_name] = bool(torch_module.allclose(snapshot[requested], snapshot[actual_name], atol=atol, rtol=rtol))
    if not all(matches.values()):
        raise RuntimeError(f"requested/actual mismatch: {matches}")
    code_indices = {code: torch_module.tensor([index for index, label in enumerate(codes) if label == code], device=core.device) for code in FORMAL_CONDITIONS}
    actual_mass = snapshot["actual_object_mass"][:, 0]
    actual_friction = snapshot["actual_object_static_friction"]
    actual_effort = snapshot["actual_effort_limit"]
    nominal_mass = actual_mass[code_indices["000"]].mean()
    nominal_effort = actual_effort[code_indices["000"]].mean(dim=0)
    expected_values = {
        "000": (1.0, 0.80, 1.0),
        "010": (1.0, 0.45, 1.0),
        "100": (1.35, 0.80, 1.0),
    }
    group_summary: dict[str, Any] = {}
    max_context_error = 0.0
    for code, ids in code_indices.items():
        mass_scale = actual_mass[ids] / nominal_mass
        friction = actual_friction[ids]
        effort_scale = (actual_effort[ids] / nominal_effort).mean(dim=1)
        target_mass, target_friction, target_effort = expected_values[code]
        value_checks = {
            "mass": bool(torch_module.allclose(mass_scale, torch_module.full_like(mass_scale, target_mass), atol=1.0e-5, rtol=1.0e-5)),
            "friction": bool(torch_module.allclose(friction, torch_module.full_like(friction, target_friction), atol=1.0e-6, rtol=0.0)),
            "effort": bool(torch_module.allclose(effort_scale, torch_module.full_like(effort_scale, target_effort), atol=1.0e-6, rtol=0.0)),
        }
        context = torch_module.stack(
            (
                2.0 * (mass_scale - 1.0) / 0.35 - 1.0,
                2.0 * (friction - 0.45) / 0.35 - 1.0,
                2.0 * (effort_scale - 0.80) / 0.20 - 1.0,
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)
        expected_context = torch_module.tensor(EXPECTED_ORACLE_CONTEXTS[code], dtype=context.dtype, device=context.device).expand_as(context)
        context_error = float((context - expected_context).abs().max().item())
        max_context_error = max(max_context_error, context_error)
        value_checks["context"] = context_error <= 1.0e-5
        if not all(value_checks.values()):
            raise RuntimeError(f"{code} actual factor/context 不匹配: {value_checks}, error={context_error}")
        group_summary[code] = {
            "count": int(ids.numel()),
            "actual_mass_mean_kg": float(actual_mass[ids].mean().item()),
            "actual_static_friction_mean": float(friction.mean().item()),
            "actual_effort_mean": float(actual_effort[ids].mean().item()),
            "normalized_context": list(EXPECTED_ORACLE_CONTEXTS[code]),
            "checks": value_checks,
        }
    combine_pass = set(snapshot["actual_hand_combine_mode"]) == {"min"} and set(snapshot["actual_object_combine_mode"]) == {"min"}
    material_states = sorted({round(float(value), 6) for value in actual_friction.detach().cpu().tolist()})
    material_state_separation = material_states == [0.45, 0.8]
    stage = get_current_stage()
    sampled_material_paths: list[str] = []
    for group_index, code in enumerate(FORMAL_CONDITIONS):
        group_start = group_index * ENVS_PER_CONDITION
        for env_id in range(group_start, group_start + MIXED_PROBE_SAMPLE_PER_CONDITION):
            material_path = f"{core.scene.env_prim_paths[env_id]}/object/cri_object_physics_material"
            if not stage.GetPrimAtPath(material_path).IsValid():
                raise RuntimeError(f"mixed material prim 缺失: code={code}, path={material_path}")
            sampled_material_paths.append(material_path)
    material_prim_identity_pass = len(set(sampled_material_paths)) == len(sampled_material_paths)
    if not combine_pass or not material_state_separation or not material_prim_identity_pass:
        raise RuntimeError(
            f"material state collapse/overwrite: modes={combine_pass}, states={material_states}, "
            f"prim_identity={material_prim_identity_pass}"
        )
    return {
        "pass": True,
        "factor_counts": count_report,
        "actual_availability": availability,
        "requested_actual_matches": matches,
        "groups": group_summary,
        "combine_mode_pass": combine_pass,
        "material_state_values": material_states,
        "material_state_separation_pass": material_state_separation,
        "material_prim_identity_pass": material_prim_identity_pass,
        "sampled_material_prim_paths": sampled_material_paths,
        "max_context_error": max_context_error,
        "digests": {field: tensor_digest(snapshot[field]) for field in CONTEXT_SOURCE_FIELDS},
    }


def run_mixed_friction_probe(core: Any, torch_module: Any) -> dict[str, Any]:  # 只验证同 scene 摩擦方向，不计算 O0 指标。
    selected: dict[str, list[int]] = {}
    for group_index, code in enumerate(FORMAL_CONDITIONS):
        start = group_index * ENVS_PER_CONDITION
        selected[code] = list(range(start, start + MIXED_PROBE_SAMPLE_PER_CONDITION))
    flat_ids = [env_id for code in FORMAL_CONDITIONS for env_id in selected[code]]
    ids = torch_module.tensor(flat_ids, dtype=torch_module.long, device=core.device)
    zero_forces = torch_module.zeros((len(flat_ids), 1, 3), dtype=torch_module.float32, device=core.device)
    core.object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=zero_forces,
        torques=zero_forces,
        env_ids=ids,
        is_global=True,
    )

    def physics_step() -> None:  # 绕过 task step，避免 reward/termination 参与探针。
        core.scene.write_data_to_sim()
        core.sim.step(render=False)
        core.scene.update(float(core.cfg.sim.dt))

    for _ in range(MIXED_PROBE_SETTLE_STEPS):
        physics_step()
    x0 = core.object.data.root_pos_w.torch[ids, 0].clone()
    traces: list[dict[str, Any]] = []
    try:
        for force_n in MIXED_PROBE_FORCE_LEVELS_N:
            forces = zero_forces.clone()
            forces[:, 0, 0] = force_n
            core.object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=forces,
                torques=zero_forces,
                env_ids=ids,
                is_global=True,
            )
            for _ in range(MIXED_PROBE_HOLD_STEPS):
                physics_step()
            displacement = (core.object.data.root_pos_w.torch[ids, 0] - x0).abs()
            traces.append({"force_n": force_n, "displacement_m": [float(value) for value in displacement.detach().cpu().tolist()]})
    finally:
        core.object.permanent_wrench_composer.reset()
    final_displacement = torch_module.tensor(traces[-1]["displacement_m"], dtype=torch_module.float64)
    medians: dict[str, float] = {}
    cursor = 0
    for code in FORMAL_CONDITIONS:
        values = final_displacement[cursor : cursor + MIXED_PROBE_SAMPLE_PER_CONDITION]
        medians[code] = float(values.median().item())
        cursor += MIXED_PROBE_SAMPLE_PER_CONDITION
    direction_pass = medians["010"] > medians["000"] + MIXED_PROBE_DIRECTION_EPS_M and medians["010"] > medians["100"] + MIXED_PROBE_DIRECTION_EPS_M
    return {
        "status": "PASS" if direction_pass else "BLOCKED",
        "scope": "mixed_scene_engineering_integrity_only_not_p0_or_o0_evidence",
        "selected_env_ids": selected,
        "force_levels_n": list(MIXED_PROBE_FORCE_LEVELS_N),
        "settle_steps": MIXED_PROBE_SETTLE_STEPS,
        "hold_steps": MIXED_PROBE_HOLD_STEPS,
        "final_median_abs_x_displacement_m": medians,
        "criterion": "median(D1/010) > median(D0/000) and median(D1/010) > median(D2/100) by 1e-6 m",
        "direction_pass": direction_pass,
        "traces": traces,
        "scientific_conclusion_generated": False,
    }


def make_formal_adapter_class(torch_module: Any, gym_module: Any, numpy_module: Any):  # 独立于 smoke 状态机的 policy adapter。
    class FormalO0Adapter:
        def __init__(self, base_env: Any, branch: str, clip_observations: float):
            self.base_env = base_env
            self.branch = branch
            self.clip_observations = clip_observations
            self.audit_records: list[dict[str, Any]] = []

        @property
        def unwrapped(self):
            return self.base_env.unwrapped

        @property
        def observation_space(self):
            return gym_module.spaces.Box(-self.clip_observations, self.clip_observations, (POLICY_INPUT_DIM,), dtype=numpy_module.float32)

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

        def set_env_state(self, state: Any):
            method = getattr(self.base_env, "set_env_state", None)
            return method(state) if callable(method) else None

        def has_action_masks(self):
            method = getattr(self.base_env, "has_action_masks", None)
            return bool(method()) if callable(method) else False

        def reset(self):
            return self._adapt(self.base_env.reset())

        def step(self, actions: Any):
            packet, rewards, dones, extras = self.base_env.step(actions)
            return self._adapt(packet), rewards, dones, extras

        def close(self):
            return self.base_env.close()

        def _adapt(self, packet: Any):  # context 仅在 reset/step 后从 actual getter 构造。
            if not isinstance(packet, dict) or set(packet) - {"obs"}:
                raise RuntimeError(f"禁止非对称或额外 observation keys: {list(packet)}")
            observation = packet.get("obs")
            if not torch_module.is_tensor(observation) or tuple(observation.shape) != (FORMAL_NUM_ENVS, LEGAL_OBSERVATION_DIM):
                raise RuntimeError(f"96D observation shape 不匹配: {getattr(observation, 'shape', None)}")
            snapshot = self.unwrapped.get_factor_snapshot()
            audit_report = audit_mixed_snapshot(snapshot, self.unwrapped, torch_module)
            codes = tuple(snapshot["factor_codes"])
            mass = snapshot["actual_object_mass"][:, 0]
            friction = snapshot["actual_object_static_friction"]
            effort = snapshot["actual_effort_limit"]
            code_tensor = {code: torch_module.tensor([index for index, label in enumerate(codes) if label == code], device=mass.device) for code in FORMAL_CONDITIONS}
            mass_scale = mass / NOMINAL_OBJECT_MASS_KG
            motor_scale = effort.mean(dim=1) / NOMINAL_EFFORT_LIMIT
            actual_context = torch_module.stack(
                (
                    2.0 * (mass_scale - 1.0) / 0.35 - 1.0,
                    2.0 * (friction - 0.45) / 0.35 - 1.0,
                    2.0 * (motor_scale - 0.80) / 0.20 - 1.0,
                ),
                dim=-1,
            ).clamp(-1.0, 1.0)
            context = torch_module.zeros_like(actual_context) if self.branch == "history_only" else actual_context
            policy_input = torch_module.cat((observation, context), dim=-1)
            if tuple(policy_input.shape) != (FORMAL_NUM_ENVS, POLICY_INPUT_DIM):
                raise RuntimeError(f"99D policy input 不匹配: {tuple(policy_input.shape)}")
            if self.branch == "history_only" and int(torch_module.count_nonzero(context).item()) != 0:
                raise RuntimeError("History context 不是 exact zero")
            self.audit_records.append(
                {
                    "source_fields": list(CONTEXT_SOURCE_FIELDS),
                    "prohibited_fields_consumed": [],
                    "requested_values_used": False,
                    "factor_codes_used": False,
                    "branch": self.branch,
                    "input_shape": list(policy_input.shape),
                    "mixed_snapshot_pass": audit_report["pass"],
                    "context_digests_by_code": {
                        code: tensor_digest(context[ids]) for code, ids in code_tensor.items()
                    },
                }
            )
            return {"obs": policy_input}

    return FormalO0Adapter


def model_state_hash(state_dict: dict[str, Any]) -> str:  # 记录初始化与最终模型证据。
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_formal_stack(args: argparse.Namespace, app_launcher: Any, timeline: FormalTimeline, run_dir: Path, capacity_mode: bool) -> dict[str, Any]:  # 构建共同 formal stack，不调用训练。
    import gymnasium as gymnasium
    import numpy as np
    import torch
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    import LEAP_Isaaclab.tasks  # noqa: F401

    timeline.snapshot_provider = lambda: audit.collect_resource_snapshot(torch)
    repo_root = args.repo_root.resolve(strict=True)
    config, config_record = load_committed_agent_config(repo_root)
    realized = copy.deepcopy(config)
    realized["params"]["seed"] = args.seed
    realized_config = realized["params"]["config"]
    realized_config["num_actors"] = FORMAL_NUM_ENVS
    realized_config["max_epochs"] = MAX_EPOCHS
    realized_config["train_dir"] = str(run_dir / "rl_games")
    realized_config["full_experiment_name"] = "run"
    realized["params"]["load_checkpoint"] = False
    realized["params"]["load_path"] = ""
    realized_sha = audit.sha256_bytes(json.dumps(realized, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env_cfg = parse_env_cfg(TASK_ID, device=app_launcher.device, num_envs=FORMAL_NUM_ENVS)
    env_cfg.seed = args.seed
    env_cfg.factor_codes = build_balanced_factor_codes()
    if env_cfg.events is not None or env_cfg.enable_adr is not False:
        raise RuntimeError("formal O0 要求 events=None 且 ADR=false")
    if abs(float(env_cfg.sim.dt) - PHYSICS_DT) > 1.0e-12 or int(env_cfg.decimation) != DECIMATION:
        raise RuntimeError(f"physics/controller 合同漂移: dt={env_cfg.sim.dt}, decimation={env_cfg.decimation}")
    raw_env = wrapped_env = adapter = agent = None
    guards = {"train_epoch_calls": 0, "optimizer_step_calls": 0, "backward_gradient_calls": 0}
    try:
        raw_env = gymnasium.make(TASK_ID, cfg=env_cfg)
        timeline.record(2, {"num_envs": FORMAL_NUM_ENVS})
        raw_env.reset(seed=args.seed)
        timeline.record(3)
        first_snapshot = raw_env.unwrapped.get_factor_snapshot()
        first_audit = audit_mixed_snapshot(first_snapshot, raw_env.unwrapped, torch)
        raw_env.reset(seed=args.seed)
        second_snapshot = raw_env.unwrapped.get_factor_snapshot()
        second_audit = audit_mixed_snapshot(second_snapshot, raw_env.unwrapped, torch)
        reset_stability = snapshot_stability(first_snapshot, second_snapshot, torch)
        if not reset_stability["pass"]:
            raise RuntimeError(f"mixed factor assignment/read-back 在 reset 后漂移: {reset_stability}")
        physical_probe = run_mixed_friction_probe(raw_env.unwrapped, torch)
        raw_env.unwrapped.object.permanent_wrench_composer.reset()
        raw_env.reset(seed=args.seed)
        post_probe_snapshot = raw_env.unwrapped.get_factor_snapshot()
        post_probe_audit = audit_mixed_snapshot(post_probe_snapshot, raw_env.unwrapped, torch)
        post_probe_stability = snapshot_stability(second_snapshot, post_probe_snapshot, torch)
        mixed_integrity_pass = physical_probe["status"] == "PASS" and post_probe_stability["pass"]
        mixed_report = {
            "protocol_id": PROTOCOL_ID,
            "implementation_id": IMPLEMENTATION_ID,
            "status": "PASS" if mixed_integrity_pass else "MIXED_FACTOR_INTEGRITY_BLOCKED",
            "first_readback": first_audit,
            "second_readback": second_audit,
            "reset_stability": reset_stability,
            "physical_probe": physical_probe,
            "post_probe_readback": post_probe_audit,
            "post_probe_stability": post_probe_stability,
            "scientific_conclusion_generated": False,
        }
        audit.atomic_write_json(args.output_root / "mixed_factor_integrity" / "latest_result.json", mixed_report)
        if not mixed_integrity_pass:
            raise RuntimeError("MIXED_FACTOR_INTEGRITY_BLOCKED")
        timeline.record(4, {"mixed_factor_integrity": "PASS", "max_context_error": first_audit["max_context_error"]})
        clip_obs = float(realized["params"]["env"]["clip_observations"])
        clip_actions = float(realized["params"]["env"]["clip_actions"])
        wrapped_env = RlGamesVecEnvWrapper(raw_env, realized_config["device"], clip_obs, clip_actions)
        Adapter = make_formal_adapter_class(torch, __import__("gym"), np)
        adapter = Adapter(wrapped_env, args.branch, clip_obs)
        adapter.reset()
        if tuple(adapter.observation_space.shape) != (POLICY_INPUT_DIM,):
            raise RuntimeError("adapter 未公开 99D observation space")
        timeline.record(5, {"branch": args.branch, "context_audit": adapter.audit_records[-1]})
        wrapper_name = f"R7O0FormalIsaacWrapper{os.getpid()}"
        vecenv.register(wrapper_name, lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
        env_configurations.register("rlgpu", {"vecenv_type": wrapper_name, "env_creator": lambda **kwargs: adapter})
        runner = Runner(IsaacAlgoObserver())
        runner.load(realized)
        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        timeline.record(6)
        checkpoint_path = repo_root / EXPANDED_CHECKPOINT_RELATIVE
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent.model.load_state_dict(checkpoint["model"], strict=True)
        trainable_count = sum(int(parameter.numel()) for parameter in agent.model.parameters() if parameter.requires_grad)
        state_count = sum(int(tensor.numel()) for tensor in agent.model.state_dict().values())
        if trainable_count != EXPECTED_TRAINABLE_PARAMETERS or state_count != EXPECTED_STATE_TENSOR_ELEMENTS:
            raise RuntimeError(f"model capacity 漂移: params={trainable_count}, state={state_count}")
        if agent.has_central_value or realized["params"]["network"]["separate"] is not False:
            raise RuntimeError("formal O0 禁止 asymmetric critic")
        require_fresh_optimizer(agent.optimizer.state)
        model_hash_before = model_state_hash(agent.model.state_dict())
        if capacity_mode:
            def forbidden_train_epoch(*_args: Any, **_kwargs: Any):  # capacity 模式碰到训练即硬失败。
                guards["train_epoch_calls"] += 1
                raise RuntimeError("capacity-preflight 禁止 train_epoch")

            def forbidden_optimizer_step(*_args: Any, **_kwargs: Any):  # capacity 模式禁止参数更新。
                guards["optimizer_step_calls"] += 1
                raise RuntimeError("capacity-preflight 禁止 optimizer.step")

            agent.train_epoch = forbidden_train_epoch
            agent.optimizer.step = forbidden_optimizer_step
            for parameter in agent.model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(lambda gradient: guards.__setitem__("backward_gradient_calls", guards["backward_gradient_calls"] + 1) or gradient)
        agent.init_tensors()
        torch.cuda.synchronize()
        if capacity_mode:
            if any(guards.values()) or any(parameter.grad is not None for parameter in agent.model.parameters()):
                raise RuntimeError(f"capacity-preflight 触发了禁止训练路径: {guards}")
        timeline.record(
            7,
            {
                "strict_checkpoint_load": True,
                "rollout_allocation": True,
                "trainable_parameter_count": trainable_count,
                "state_tensor_element_count": state_count,
                "fresh_optimizer": len(agent.optimizer.state) == 0,
                "forbidden_training_counters": guards,
            },
        )
        return {
            "raw_env": raw_env,
            "wrapped_env": wrapped_env,
            "adapter": adapter,
            "agent": agent,
            "runner": runner,
            "realized_config": realized,
            "realized_config_sha256": realized_sha,
            "config_record": config_record,
            "mixed_report": mixed_report,
            "model_hash_before": model_hash_before,
            "trainable_parameter_count": trainable_count,
            "state_tensor_element_count": state_count,
            "guards": guards,
        }
    except Exception:
        if agent is not None and getattr(agent, "writer", None) is not None:
            agent.writer.close()
        if wrapped_env is not None:
            wrapped_env.close()
        elif raw_env is not None:
            raw_env.close()
        raise


def close_stack(stack: dict[str, Any] | None) -> None:  # 统一关闭 writer 与 PhysX env。
    if not stack:
        return
    agent = stack.get("agent")
    if agent is not None and getattr(agent, "writer", None) is not None:
        agent.writer.close()
    wrapped = stack.get("wrapped_env")
    raw = stack.get("raw_env")
    if wrapped is not None:
        wrapped.close()
    elif raw is not None:
        raw.close()


def run_capacity_preflight(args: argparse.Namespace, app_launcher: Any, timeline: FormalTimeline, run_dir: Path) -> dict[str, Any]:  # 仅初始化到 F7。
    stack: dict[str, Any] | None = None
    try:
        stack = build_formal_stack(args, app_launcher, timeline, run_dir, capacity_mode=True)
        return {
            "status": "PASS",
            "capacity_status": "FORMAL_6144_FULLSTACK_INIT_PASS",
            "mixed_factor_integrity_status": stack["mixed_report"]["status"],
            "branch": args.branch,
            "seed": args.seed,
            "num_envs": FORMAL_NUM_ENVS,
            "completed_stages": sorted(timeline.completed_ids),
            "realized_config_sha256": stack["realized_config_sha256"],
            "model_hash_before": stack["model_hash_before"],
            "trainable_parameter_count": stack["trainable_parameter_count"],
            "state_tensor_element_count": stack["state_tensor_element_count"],
            "fresh_optimizer": len(stack["agent"].optimizer.state) == 0,
            "forbidden_training_counters": stack["guards"],
            "train_epoch_executed": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "trained_checkpoint_written": False,
            "scientific_conclusion_generated": False,
        }
    finally:
        close_stack(stack)


def run_formal_training(args: argparse.Namespace, app_launcher: Any, timeline: FormalTimeline, run_dir: Path) -> dict[str, Any]:  # 仅在外部授权文件通过后可达。
    stack: dict[str, Any] | None = None
    try:
        stack = build_formal_stack(args, app_launcher, timeline, run_dir, capacity_mode=False)
        agent = stack["agent"]
        agent.train()
        if int(agent.epoch_num) != MAX_EPOCHS:
            raise RuntimeError(f"正式训练未达到 epoch 5000: {agent.epoch_num}")
        final_path = run_dir / "checkpoints" / "epoch_5000_final"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(final_path))
        final_file = final_path.with_suffix(".pth")
        if not final_file.is_file():
            raise RuntimeError(f"epoch-5000 final checkpoint 缺失: {final_file}")
        return {
            "status": "COMPLETE",
            "completion_status": "COMPLETE",
            "protocol_id": PROTOCOL_ID,
            "branch": args.branch,
            "seed": args.seed,
            "git_commit": audit.git_text(args.repo_root, "rev-parse", "HEAD"),
            "checkpoint_sha256": EXPECTED_EXPANDED_SHA256,
            "factor_counts": EXPECTED_FACTOR_COUNTS,
            "actual_context_distribution": {
                code: stack["mixed_report"]["first_readback"]["groups"][code]["normalized_context"]
                for code in FORMAL_CONDITIONS
            },
            "optimizer_fresh_state": True,
            "epoch_count": int(agent.epoch_num),
            "frame_count": int(agent.frame),
            "checkpoint_selection": EVALUATION_CHECKPOINT_RULE,
            "final_checkpoint": {"path": str(final_file), "sha256": audit.sha256_file(final_file)},
            "final_checkpoint_sha256": audit.sha256_file(final_file),
            "realized_config_sha256": stack["realized_config_sha256"],
            "factor_count": EXPECTED_FACTOR_COUNTS,
            "optimizer_fresh_at_initialization": True,
            "resource_telemetry": str(timeline.path),
        }
    finally:
        close_stack(stack)


def refresh_checksums(output_root: Path) -> None:  # 覆盖 JSON、console 与未来 checkpoint。
    checksum_path = output_root / "artifact_checksums.json"
    artifacts = {
        str(path.relative_to(output_root)).replace("\\", "/"): audit.sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path != checksum_path
    }
    audit.atomic_write_json(
        checksum_path,
        {"protocol_id": PROTOCOL_ID, "implementation_id": IMPLEMENTATION_ID, "artifacts": artifacts},
    )


def main(argv: Sequence[str] | None = None) -> int:  # 本轮只授权 capacity-preflight。
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    try:
        validate_cli_contract(args)
        args.repo_root = args.repo_root.expanduser().resolve(strict=True)
        args.output_root = args.output_root.expanduser().resolve(strict=False)
        git_commit = audit.git_text(args.repo_root, "rev-parse", "HEAD")
        if args.mode == "train":
            validate_training_authorization(args.authorization_file, args.branch, args.seed, git_commit)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    initialize_artifact_schema(args.output_root)
    run_id = f"run_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    run_dir = args.output_root / ("capacity_preflight" if args.mode == "capacity-preflight" else f"{args.branch}/seed_{args.seed}") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    timeline = FormalTimeline(run_dir / "allocation_timeline.json")
    app_launcher = None
    result: dict[str, Any]
    try:
        config, config_record = load_committed_agent_config(args.repo_root)
        del config
        checkpoint_record = checkpoint_preflight(args.repo_root)
        clean_record = clean_process_gate(args.repo_root)
        audit.atomic_write_json(args.output_root / "protocol_manifest.json", create_protocol_manifest(args.repo_root, config_record))
        audit.atomic_write_json(
            run_dir / "preflight_system_state.json",
            {"clean_process_gate": clean_record, "checkpoint": checkpoint_record, "resources": audit.collect_resource_snapshot()},
        )
        if not clean_record["pass"]:
            raise RuntimeError(f"clean process gate 发现残留: {clean_record['residuals']}")
        timeline.record(0)
        app_launcher = AppLauncher(args)
        timeline.record(1)
        if args.mode == "capacity-preflight":
            result = run_capacity_preflight(args, app_launcher, timeline, run_dir)
        else:
            result = run_formal_training(args, app_launcher, timeline, run_dir)
    except Exception as exc:
        timeline.fail(exc)
        result = {
            "status": "BLOCKED",
            "protocol_id": PROTOCOL_ID,
            "implementation_id": IMPLEMENTATION_ID,
            "mode": args.mode,
            "branch": args.branch,
            "seed": args.seed,
            "num_envs": FORMAL_NUM_ENVS,
            "completed_stages": sorted(timeline.completed_ids),
            "failure_stage": next((stage for stage in FORMAL_STAGES if stage not in timeline.completed_ids), None),
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "resource_state": audit.collect_resource_snapshot(),
            "scientific_protocol_must_be_revised": False,
            "execution_alternatives_not_applied": ["larger-memory GPU", "multi-GPU/server execution", "explicit protocol revision"],
            "scientific_conclusion_generated": False,
        }
    result["git_commit"] = audit.git_text(args.repo_root, "rev-parse", "HEAD")
    result["checkpoint_hashes_after"] = checkpoint_preflight(args.repo_root)
    result_path = run_dir / "result.json"
    audit.atomic_write_json(result_path, result)
    if args.mode == "capacity-preflight":
        audit.atomic_write_json(args.output_root / "capacity_preflight" / "latest_result.json", result)
        mixed_pass = result.get("mixed_factor_integrity_status") == "PASS"
        decision = final_decision(result.get("status") == "PASS", mixed_pass)
        audit.atomic_write_json(
            args.output_root / "capacity_decision.json",
            {"decision": decision, "result_path": str(result_path), "scientific_conclusion_generated": False},
        )
    else:
        decision = ["FORMAL_TRAINING_COMPLETE"] if result.get("status") == "COMPLETE" else ["BLOCKED_IMPLEMENTATION_CONTRACT"]
    refresh_checksums(args.output_root)
    print("\n".join(decision), flush=True)
    if app_launcher is not None:
        app_launcher.app.close()
    return 0 if result.get("status") in ("PASS", "COMPLETE") else 2


if __name__ == "__main__":
    sys.exit(main())
