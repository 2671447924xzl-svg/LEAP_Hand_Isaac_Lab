"""R7-O0 Phase 4.2b：6144 环境单 epoch backward 可行性门。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent  # 只定位同目录的冻结 formal driver。
FORMAL_DRIVER_PATH = SCRIPT_DIR / "r7_o0_formal_train.py"  # 复用 Phase 4.2 formal stack。
_FORMAL_SPEC = importlib.util.spec_from_file_location("r7_o0_formal_train_phase42b", FORMAL_DRIVER_PATH)
if _FORMAL_SPEC is None or _FORMAL_SPEC.loader is None:
    raise ImportError(f"无法加载 formal driver: {FORMAL_DRIVER_PATH}")
formal = importlib.util.module_from_spec(_FORMAL_SPEC)
_FORMAL_SPEC.loader.exec_module(formal)


PROTOCOL_ID = formal.PROTOCOL_ID  # 冻结科学协议。
FAIRNESS_SCHEMA = formal.FAIRNESS_SCHEMA  # 冻结公平性协议。
FEASIBILITY_ID = "R7_O0_PHASE42B_BACKWARD_FEASIBILITY_V1"  # 工程 gate 标识。
PARENT_ENGINEERING_COMMIT = "f868ecc2df0475a49f12451f0ad56b23fde75444"  # Phase 4.2 起点。
SEED = 7100  # 本 gate 唯一允许的训练种子。
EXPECTED_FRAMES = formal.FORMAL_NUM_ENVS * formal.HORIZON_LENGTH  # 单 epoch 环境帧数。
GRU_INPUT_PARAMETER = formal.audit.GRU_INPUT_PARAMETER  # 99D GRU 输入权重。
DEFAULT_OUTPUT_ROOT = formal.DEFAULT_OUTPUT_ROOT / "backward_feasibility"  # 与 Phase 4.2 artifact 根隔离。
DEFAULT_ISAACLAB_BAT = Path(r"D:\Isaac\IsaacLab_LEAP\isaaclab.bat")  # 当前验证主机的唯一 launcher。
BACKWARD_STAGES = {
    "B0": "before_rollout",
    "B1": "after_rollout",
    "B2": "before_first_backward",
    "B3": "immediately_after_first_backward",
    "B4": "after_first_optimizer_step",
}  # 用户冻结的资源边界。
RESOURCE_SIGNATURES = (
    "out of memory",
    "cuda error: out of memory",
    "cuda out of memory",
    "pinned memory",
    "memory allocation",
    "cudaerrormemoryallocation",
    "cublas_status_alloc_failed",
    "failed to allocate",
    "allocation failed",
    "std::bad_alloc",
    "bad allocation",
    "gpu heap",
    "not enough memory",
    "paging file",
)  # 只用于失败分类，不触发自动降档。
PROHIBITED_FAILURE_SIGNATURES = {
    "oom": ("out of memory", "cuda error: out of memory", "cuda out of memory"),
    "pinned_memory": ("pinned memory",),
    "articulation": ("failed to create articulation",),
}  # 必须显式记录的失败类型。


def utc_now() -> str:  # 返回 UTC artifact 时间戳。
    return formal.audit.utc_now()


def frozen_contract_record() -> dict[str, Any]:  # 生成逐分支相同的冻结合同记录。
    fields = formal.expected_frozen_config_fields()
    return {
        "num_envs": formal.FORMAL_NUM_ENVS,
        "factor_counts": dict(formal.EXPECTED_FACTOR_COUNTS),
        "horizon_length": fields["params.config.horizon_length"],
        "minibatch_size": fields["params.config.minibatch_size"],
        "mini_epochs": fields["params.config.mini_epochs"],
        "seq_length": fields["params.config.seq_length"],
        "mixed_precision": fields["params.config.mixed_precision"],
        "policy_input_dim": formal.POLICY_INPUT_DIM,
        "legal_observation_dim": formal.LEGAL_OBSERVATION_DIM,
        "context_dim": formal.CONTEXT_DIM,
        "action_dim": formal.ACTION_DIM,
        "formal_max_epochs_unchanged": fields["params.config.max_epochs"],
        "gate_epoch_calls": 1,
        "seed": SEED,
    }


def scientific_config_sha256() -> str:  # 排除 branch 日志路径后的冻结运行配置哈希。
    payload = json.dumps(frozen_contract_record(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_cli_contract(args: argparse.Namespace) -> None:  # 只允许两个冻结分支和 seed 7100。
    if args.branch not in formal.BRANCHES:
        raise ValueError(f"非法 branch: {args.branch}")
    if int(args.seed) != SEED:
        raise ValueError(f"Phase 4.2b 只允许 seed={SEED}: actual={args.seed}")


def classify_failure(stage_id: str | None, message: str) -> list[str]:  # 区分资源边界与软件问题。
    lowered = message.lower()
    resource_failure = any(signature in lowered for signature in RESOURCE_SIGNATURES)
    if stage_id in {"B0", "B1", "B2", "B3", "B4"} and resource_failure:
        return ["FORMAL_6144_BACKWARD_CAPACITY_BLOCKED", "HARDWARE_OR_PROTOCOL_REVISION_REQUIRED"]
    return ["FORMAL_TRAINING_PIPELINE_BLOCKED"]


def failure_signature_record(message: str) -> dict[str, bool]:  # 保存 OOM、pinned 与 articulation 签名。
    lowered = message.lower()
    record = {
        name: any(signature in lowered for signature in signatures)
        for name, signatures in PROHIBITED_FAILURE_SIGNATURES.items()
    }
    record["resource_allocation"] = any(signature in lowered for signature in RESOURCE_SIGNATURES)
    return record


class BackwardResourceTimeline:  # B0–B4 完成后立即原子落盘。
    def __init__(self, path: Path, branch: str, snapshot_provider: Any = formal.audit.collect_resource_snapshot):
        self.path = path
        self.branch = branch
        self.snapshot_provider = snapshot_provider
        self.started = time.perf_counter()
        self.stages: list[dict[str, Any]] = []
        self.active_stage_id: str | None = None
        self._write()

    @property
    def completed_ids(self) -> set[str]:  # 返回已完成边界。
        return {str(item["stage_id"]) for item in self.stages if item["status"] == "COMPLETE"}

    @property
    def next_stage_id(self) -> str | None:  # 返回首个未完成边界。
        return next((stage_id for stage_id in BACKWARD_STAGES if stage_id not in self.completed_ids), None)

    @property
    def failed_stage_id(self) -> str | None:  # 返回已落盘失败边界。
        failed = [item.get("stage_id") for item in self.stages if item.get("status") == "FAILED"]
        return failed[-1] if failed else None

    def _write(self) -> None:  # 保证 OOM 后仍有最近完整边界。
        formal.audit.atomic_write_json(
            self.path,
            {
                "protocol_id": PROTOCOL_ID,
                "feasibility_id": FEASIBILITY_ID,
                "schema_version": 1,
                "branch": self.branch,
                "active_stage_id": self.active_stage_id,
                "stages": self.stages,
                "elapsed_seconds": time.perf_counter() - self.started,
                "scientific_conclusion_generated": False,
            },
        )

    def begin(self, stage_id: str) -> None:  # 在可能失败的操作前先冻结活动边界。
        if stage_id not in BACKWARD_STAGES:
            raise ValueError(f"非法 backward stage: {stage_id}")
        if stage_id in self.completed_ids:
            self.active_stage_id = None
            return
        self.active_stage_id = stage_id
        self._write()

    def record(self, stage_id: str, details: dict[str, Any] | None = None) -> None:  # 每个边界只记录一次。
        if stage_id not in BACKWARD_STAGES:
            raise ValueError(f"非法 backward stage: {stage_id}")
        if stage_id in self.completed_ids:
            return
        self.active_stage_id = stage_id
        self.stages.append(
            {
                "stage_id": stage_id,
                "stage_name": BACKWARD_STAGES[stage_id],
                "status": "COMPLETE",
                "elapsed_seconds": time.perf_counter() - self.started,
                "resources": self.snapshot_provider(),
                "details": details or {},
            }
        )
        self.active_stage_id = None
        self._write()

    def fail(self, exc: BaseException) -> None:  # 将异常绑定到首个未完成边界。
        stage_id = self.active_stage_id
        self.stages.append(
            {
                "stage_id": stage_id,
                "stage_name": BACKWARD_STAGES.get(stage_id, "after_B4"),
                "status": "FAILED",
                "elapsed_seconds": time.perf_counter() - self.started,
                "resources": self.snapshot_provider(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        self.active_stage_id = None
        self._write()


def validate_oracle_precondition(output_root: Path, current_commit: str, current_pid: int | None = None) -> dict[str, Any]:  # History PASS 才允许 Oracle。
    history_path = output_root / "history_only" / "result.json"
    console_path = output_root / "history_only" / "console.log"
    preflight_path = output_root / "history_only" / "preflight_system_state.json"
    if not history_path.is_file():
        raise RuntimeError("ORACLE_BLOCKED_HISTORY_RESULT_MISSING")
    if not console_path.is_file():
        raise RuntimeError("ORACLE_BLOCKED_HISTORY_CONSOLE_MISSING")
    if not preflight_path.is_file():
        raise RuntimeError("ORACLE_BLOCKED_HISTORY_PREFLIGHT_MISSING")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    console_text = console_path.read_text(encoding="utf-8", errors="replace")
    pid = os.getpid() if current_pid is None else current_pid
    checksum_path = output_root / "artifact_checksums.json"
    checksum_record = json.loads(checksum_path.read_text(encoding="utf-8")) if checksum_path.is_file() else {}
    expected_result_hash = checksum_record.get("artifacts", {}).get("history_only/result.json")
    expected_console_hash = checksum_record.get("artifacts", {}).get("history_only/console.log")
    expected_preflight_hash = checksum_record.get("artifacts", {}).get("history_only/preflight_system_state.json")
    actual_result_hash = formal.audit.sha256_file(history_path)
    actual_console_hash = formal.audit.sha256_file(console_path)
    actual_preflight_hash = formal.audit.sha256_file(preflight_path)
    console_capture = history.get("console_capture", {})
    capture_admission = preflight.get("console_capture_admission", {})
    clean_process_gate = preflight.get("clean_process_gate", {})
    token_digest = str(console_capture.get("token_sha256", ""))
    process_exit_code = history.get("process_exit_code")
    start_sentinel = f"R7_PHASE42B_PARENT_CAPTURE_START token_sha256={token_digest}"
    completion_sentinel = (
        f"R7_PHASE42B_PARENT_CAPTURE_COMPLETE token_sha256={token_digest} "
        f"exit_code={process_exit_code}"
    )
    console_lines = console_text.splitlines()
    controlled_parent = clean_process_gate.get("controlled_parent") or {}
    before = history.get("checkpoint_hashes_before", {})
    after = history.get("checkpoint_hashes_after", {})
    checkpoint_identity = all(
        (
            record.get("source", {}).get("sha256") == formal.EXPECTED_SOURCE_SHA256
            and record.get("expanded", {}).get("sha256") == formal.EXPECTED_EXPANDED_SHA256
            and record.get("only_preexisting_yaml_diff") is True
        )
        for record in (before, after)
    )
    checks = {
        "history_pass": history.get("status") == "PASS",
        "history_branch": history.get("branch") == "history_only",
        "protocol_identity": history.get("protocol_id") == PROTOCOL_ID
        and history.get("fairness_schema") == FAIRNESS_SCHEMA
        and history.get("feasibility_id") == FEASIBILITY_ID,
        "same_git_commit": history.get("git_commit") == current_commit,
        "different_process": int(history.get("process_id", pid)) != int(pid),
        "frozen_contract": history.get("frozen_contract") == frozen_contract_record(),
        "checkpoint_identity": checkpoint_identity,
        "committed_config_only": history.get("config_source", {}).get("working_tree_consumed_as_config") is False
        and bool(history.get("config_source", {}).get("committed_sha256")),
        "model_capacity": history.get("trainable_parameter_count") == formal.EXPECTED_TRAINABLE_PARAMETERS
        and history.get("state_tensor_element_count") == formal.EXPECTED_STATE_TENSOR_ELEMENTS,
        "model_hashes_present": bool(history.get("model_hash_before")) and bool(history.get("model_hash_after")),
        "context_contract": history.get("context_contract", {}).get("pass") is True,
        "all_history_checks": bool(history.get("checks")) and all(history.get("checks", {}).values()),
        "console_finalized": history.get("console_evidence_finalized") is True,
        "no_failure_signatures": not any(history.get("failure_signatures", {}).values()),
        "no_checkpoint": history.get("no_trained_checkpoint_retained") is True,
        "artifact_hash": bool(expected_result_hash) and expected_result_hash == actual_result_hash,
        "console_artifact_hash": bool(expected_console_hash)
        and expected_console_hash == actual_console_hash
        and history.get("console_sha256") == actual_console_hash,
        "console_capture_provenance": console_capture.get("owner") == "phase42b_parent_launcher"
        and Path(str(console_capture.get("canonical_path", ""))).resolve() == console_path.resolve()
        and len(token_digest) == 64
        and all(character in "0123456789abcdef" for character in token_digest)
        and console_capture.get("start_sentinel_present") is True
        and console_capture.get("completion_sentinel_present") is True,
        "console_sentinel_content": console_lines.count(start_sentinel) == 1
        and console_lines.count(completion_sentinel) == 1
        and process_exit_code == 0,
        "preflight_artifact_hash": bool(expected_preflight_hash)
        and expected_preflight_hash == actual_preflight_hash,
        "parent_capture_binding": capture_admission.get("owner") == "phase42b_parent_launcher"
        and capture_admission.get("token_sha256") == token_digest
        and capture_admission.get("token_persisted_in_command_line") is False
        and clean_process_gate.get("pass") is True
        and clean_process_gate.get("controlled_parent_count") == 1
        and capture_admission.get("parent_pid") == clean_process_gate.get("expected_parent_pid")
        and controlled_parent.get("pid") == capture_admission.get("parent_pid")
        and controlled_parent.get("is_controlled_capture_parent") is True,
        "no_scientific_conclusion": history.get("scientific_conclusion_generated") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"ORACLE_BLOCKED_HISTORY_PRECONDITION: {checks}")
    return {"pass": True, "checks": checks, "history_result": str(history_path)}


def audit_controlled_parent(
    records: list[dict[str, Any]],
    expected_parent_pid: int,
    ancestor_pids: set[int],
) -> dict[str, Any]:  # 仅豁免且强制存在唯一、PID 绑定的受控 launcher。
    script_token = str(Path(__file__).resolve()).lower()
    controlled: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    for item in records:
        command = str(item.get("command_line") or "").lower()
        is_controlled_parent = (
            int(item.get("pid", -1)) == int(expected_parent_pid)
            and int(item.get("pid", -1)) in ancestor_pids
            and script_token in command
            and "--launch-branch" in command
        )
        item["is_controlled_capture_parent"] = is_controlled_parent
        if is_controlled_parent:
            controlled.append(item)
        if (
            item.get("inspection_available", True)
            and item.get("project_related")
            and not item.get("is_current_worker")
            and not item.get("is_current_launch_ancestor")
            and not is_controlled_parent
        ):
            residuals.append(item)
    parent_identity_pass = len(controlled) == 1
    return {
        "pass": parent_identity_pass and not residuals,
        "expected_parent_pid": int(expected_parent_pid),
        "controlled_parent_count": len(controlled),
        "controlled_parent": controlled[0] if parent_identity_pass else None,
        "processes": records,
        "residuals": residuals,
    }


def feasibility_clean_process_gate(repo_root: Path, expected_parent_pid: int) -> dict[str, Any]:  # 允许当前受控 parent launcher 祖先进程。
    records = formal.formal_process_records(repo_root)
    try:
        import psutil

        ancestor_pids = {parent.pid for parent in psutil.Process(os.getpid()).parents()}
    except Exception:
        ancestor_pids = set()
    return audit_controlled_parent(records, expected_parent_pid, ancestor_pids)


def _synchronize_and_record(torch_module: Any, timeline: BackwardResourceTimeline, stage_id: str, details: dict[str, Any] | None = None) -> None:  # 对齐异步 CUDA 边界。
    if stage_id in timeline.completed_ids:
        return
    timeline.begin(stage_id)
    if torch_module.cuda.is_initialized():
        torch_module.cuda.synchronize()
    timeline.record(stage_id, details)


def run_branch_epoch(
    args: argparse.Namespace,
    app_launcher: Any,
    allocation_timeline: Any,
    resource_timeline: BackwardResourceTimeline,
    branch_dir: Path,
) -> dict[str, Any]:  # 运行且仅运行一个完整 PPO epoch。
    import torch

    stack: dict[str, Any] | None = None
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
        "train_epoch_calls": 0,
    }
    network_input_shapes: set[tuple[int, ...]] = set()
    started = time.perf_counter()
    try:
        stack = formal.build_formal_stack(args, app_launcher, allocation_timeline, branch_dir, capacity_mode=False)
        agent = stack["agent"]
        resource_timeline.snapshot_provider = lambda: formal.audit.collect_resource_snapshot(torch)
        optimizer_entries_before = len(agent.optimizer.state)
        formal.require_fresh_optimizer(agent.optimizer.state)
        model_hash_before = stack["model_hash_before"]

        def forward_hook(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:  # 只读取网络输入形状。
            counters["forward_calls"] += 1
            if not inputs or not isinstance(inputs[0], dict) or "obs" not in inputs[0]:
                return
            observation = inputs[0]["obs"]
            if torch.is_tensor(observation):
                shape = tuple(observation.shape)
                network_input_shapes.add(shape)
                if shape[-1] == formal.POLICY_INPUT_DIM:
                    counters["forward_99d_calls"] += 1

        forward_handle = agent.model.register_forward_hook(forward_hook)
        named_parameters = dict(agent.model.named_parameters())
        if GRU_INPUT_PARAMETER not in named_parameters:
            raise RuntimeError(f"缺少 gradient-hook parameter: {GRU_INPUT_PARAMETER}")

        def gradient_hook(gradient: Any):  # 只读审计 GRU 输入权重梯度。
            counters["gradient_hook_calls"] += 1
            finite = bool(torch.isfinite(gradient).all().item())
            norm = float(gradient.norm().item())
            context_norm = float(gradient[:, 96:99].norm().item())
            if finite:
                counters["finite_gradient_calls"] += 1
            if finite and norm > 0.0:
                counters["nonzero_gradient_calls"] += 1
            if finite and context_norm > 0.0:
                counters["context_column_nonzero_gradient_calls"] += 1
            counters["max_gradient_norm"] = max(counters["max_gradient_norm"], norm)
            counters["max_context_column_gradient_norm"] = max(
                counters["max_context_column_gradient_norm"], context_norm
            )
            return gradient

        gradient_handle = named_parameters[GRU_INPUT_PARAMETER].register_hook(gradient_hook)
        original_play_steps = agent.play_steps
        original_play_steps_rnn = agent.play_steps_rnn
        original_calc_gradients = agent.calc_gradients
        original_truncate_and_step = agent.trancate_gradients_and_step
        original_optimizer_step = agent.optimizer.step

        def wrap_rollout(original: Any):  # 同时覆盖 recurrent 与 non-recurrent rollout。
            def measured_rollout(*call_args: Any, **call_kwargs: Any):
                if "B0" not in resource_timeline.completed_ids:
                    torch.cuda.reset_peak_memory_stats()
                    _synchronize_and_record(torch, resource_timeline, "B0")
                resource_timeline.begin("B1")
                output = original(*call_args, **call_kwargs)
                _synchronize_and_record(torch, resource_timeline, "B1")
                return output

            return measured_rollout

        def measured_calc_gradients(*call_args: Any, **call_kwargs: Any):  # 首个 loss/backward 入口。
            _synchronize_and_record(torch, resource_timeline, "B2")
            if "B3" not in resource_timeline.completed_ids:
                resource_timeline.begin("B3")
            return original_calc_gradients(*call_args, **call_kwargs)

        def measured_truncate_and_step(*call_args: Any, **call_kwargs: Any):  # 入口位于 backward 返回之后。
            _synchronize_and_record(
                torch,
                resource_timeline,
                "B3",
                {
                    "gradient_hook_calls": counters["gradient_hook_calls"],
                    "finite_gradient_calls": counters["finite_gradient_calls"],
                    "nonzero_gradient_calls": counters["nonzero_gradient_calls"],
                },
            )
            if "B4" not in resource_timeline.completed_ids:
                resource_timeline.begin("B4")
            return original_truncate_and_step(*call_args, **call_kwargs)

        def measured_optimizer_step(*call_args: Any, **call_kwargs: Any):  # 真实 optimizer 返回后计数。
            result = original_optimizer_step(*call_args, **call_kwargs)
            counters["optimizer_step_count"] += 1
            _synchronize_and_record(
                torch,
                resource_timeline,
                "B4",
                {"optimizer_step_count": counters["optimizer_step_count"]},
            )
            return result

        agent.play_steps = wrap_rollout(original_play_steps)
        agent.play_steps_rnn = wrap_rollout(original_play_steps_rnn)
        agent.calc_gradients = measured_calc_gradients
        agent.trancate_gradients_and_step = measured_truncate_and_step
        agent.optimizer.step = measured_optimizer_step
        agent.mean_rewards = agent.last_mean_rewards = -100500
        agent.obs = agent.env_reset()
        agent.update_epoch()
        counters["train_epoch_calls"] += 1
        agent.train_epoch()
        agent.dataset.update_values_dict(None)
        frames = int(agent.curr_frames * agent.world_size if agent.multi_gpu else agent.curr_frames)
        agent.frame += frames
        torch.cuda.synchronize()

        model_hash_after = formal.model_state_hash(agent.model.state_dict())
        optimizer_entries_after = len(agent.optimizer.state)
        context_audit = stack["adapter"].audit_records[-1]
        expected_context_digests: dict[str, str] = {}
        for code in formal.FORMAL_CONDITIONS:
            values = (0.0, 0.0, 0.0) if args.branch == "history_only" else formal.EXPECTED_ORACLE_CONTEXTS[code]
            expected_tensor = torch.tensor(values, dtype=torch.float32).repeat(formal.ENVS_PER_CONDITION, 1)
            expected_context_digests[code] = formal.tensor_digest(expected_tensor)
        context_contract = {
            "pass": context_audit.get("context_digests_by_code") == expected_context_digests
            and context_audit.get("source_fields") == list(formal.CONTEXT_SOURCE_FIELDS)
            and context_audit.get("prohibited_fields_consumed") == []
            and context_audit.get("requested_values_used") is False
            and context_audit.get("factor_codes_used") is False
            and context_audit.get("input_shape") == [formal.FORMAL_NUM_ENVS, formal.POLICY_INPUT_DIM],
            "source_fields": context_audit.get("source_fields"),
            "prohibited_fields_consumed": context_audit.get("prohibited_fields_consumed"),
            "requested_values_used": context_audit.get("requested_values_used"),
            "factor_codes_used": context_audit.get("factor_codes_used"),
            "input_shape": context_audit.get("input_shape"),
            "observed_digests_by_code": context_audit.get("context_digests_by_code"),
            "expected_digests_by_code": expected_context_digests,
            "branch": args.branch,
        }
        no_checkpoint = not list(branch_dir.rglob("*.pth"))
        checks = {
            "environment_creation_pass": 2 in allocation_timeline.completed_ids,
            "mixed_factor_integrity_pass": stack["mixed_report"]["status"] == "PASS",
            "policy_input_99d_pass": counters["forward_99d_calls"] > 0
            and all(shape[-1] == formal.POLICY_INPUT_DIM for shape in network_input_shapes),
            "strict_checkpoint_load_pass": 7 in allocation_timeline.completed_ids,
            "fresh_optimizer_before_update": optimizer_entries_before == 0,
            "forward_calls_positive": counters["forward_calls"] > 0,
            "finite_gradient_calls_positive": counters["finite_gradient_calls"] > 0,
            "nonzero_gradient_calls_positive": counters["nonzero_gradient_calls"] > 0,
            "optimizer_step_count_positive": counters["optimizer_step_count"] > 0,
            "optimizer_state_populated": optimizer_entries_after > 0,
            "model_hash_changed": model_hash_after != model_hash_before,
            "oracle_context_column_gradient_positive": args.branch != "oracle"
            or counters["context_column_nonzero_gradient_calls"] > 0,
            "exactly_one_epoch": counters["train_epoch_calls"] == 1 and int(agent.epoch_num) == 1,
            "expected_frames": int(agent.frame) == EXPECTED_FRAMES,
            "all_resource_stages_complete": resource_timeline.completed_ids == set(BACKWARD_STAGES),
            "context_contract_pass": context_contract["pass"],
            "no_trained_checkpoint_retained": no_checkpoint,
        }
        status = "PENDING_CONSOLE_FINALIZATION" if all(checks.values()) else "BLOCKED"
        return {
            "protocol_id": PROTOCOL_ID,
            "fairness_schema": FAIRNESS_SCHEMA,
            "feasibility_id": FEASIBILITY_ID,
            "schema_version": 1,
            "status": status,
            "branch": args.branch,
            "seed": args.seed,
            "process_id": os.getpid(),
            "timestamp_utc": utc_now(),
            "git_commit": formal.audit.git_text(args.repo_root, "rev-parse", "HEAD"),
            "parent_engineering_commit": PARENT_ENGINEERING_COMMIT,
            "scope": "ONE_FORMAL_PPO_EPOCH_BACKWARD_FEASIBILITY_NOT_SCIENTIFIC_EXPERIMENT",
            "num_envs": formal.FORMAL_NUM_ENVS,
            "factor_counts": dict(formal.EXPECTED_FACTOR_COUNTS),
            "frozen_contract": frozen_contract_record(),
            "scientific_config_sha256": scientific_config_sha256(),
            "realized_config_sha256": stack["realized_config_sha256"],
            "config_source": stack["config_record"],
            "context_audit": context_audit,
            "context_contract": context_contract,
            "mixed_factor_integrity": stack["mixed_report"],
            "trainable_parameter_count": stack["trainable_parameter_count"],
            "state_tensor_element_count": stack["state_tensor_element_count"],
            "epochs_completed": int(agent.epoch_num),
            "frames_completed": int(agent.frame),
            "optimizer_state_entries_before": optimizer_entries_before,
            "optimizer_state_entries_after": optimizer_entries_after,
            "training_counters": counters,
            "network_input_shapes": [list(shape) for shape in sorted(network_input_shapes)],
            "model_hash_before": model_hash_before,
            "model_hash_after": model_hash_after,
            "resource_timeline": str(resource_timeline.path),
            "allocation_timeline": str(allocation_timeline.path),
            "checks": checks,
            "failure_signatures": None,
            "console_evidence_finalized": False,
            "no_trained_checkpoint_retained": no_checkpoint,
            "elapsed_seconds": time.perf_counter() - started,
            "performance_metrics_computed": False,
            "evaluation_executed": False,
            "statistical_testing_executed": False,
            "h1_or_a0_executed": False,
            "adaptive_control_executed": False,
            "scientific_conclusion_generated": False,
        }
    finally:
        if gradient_handle is not None:
            gradient_handle.remove()
        if forward_handle is not None:
            forward_handle.remove()
        formal.close_stack(stack)


def build_fairness_report(results: dict[str, dict[str, Any]]) -> dict[str, Any]:  # 汇总两个进程的工程公平性。
    complete = all(branch in results for branch in formal.BRANCHES)
    history = results.get("history_only", {})
    oracle = results.get("oracle", {})
    checks = {
        "both_branches_present": complete,
        "both_branches_pass": complete and all(results[branch].get("status") == "PASS" for branch in formal.BRANCHES),
        "history_first": complete and history.get("branch") == "history_only" and oracle.get("branch") == "oracle",
        "fresh_processes": complete and history.get("process_id") != oracle.get("process_id"),
        "same_seed": complete and history.get("seed") == oracle.get("seed") == SEED,
        "same_git_commit": complete and history.get("git_commit") == oracle.get("git_commit"),
        "same_frozen_contract": complete and history.get("frozen_contract") == oracle.get("frozen_contract") == frozen_contract_record(),
        "same_factor_counts": complete
        and history.get("factor_counts") == oracle.get("factor_counts") == formal.EXPECTED_FACTOR_COUNTS,
        "same_initial_model_hash": complete and history.get("model_hash_before") == oracle.get("model_hash_before"),
        "checkpoint_hashes_exact": complete
        and all(
            result.get("checkpoint_hashes_before", {}).get("source", {}).get("sha256") == formal.EXPECTED_SOURCE_SHA256
            and result.get("checkpoint_hashes_before", {}).get("expanded", {}).get("sha256") == formal.EXPECTED_EXPANDED_SHA256
            and result.get("checkpoint_hashes_after", {}).get("source", {}).get("sha256") == formal.EXPECTED_SOURCE_SHA256
            and result.get("checkpoint_hashes_after", {}).get("expanded", {}).get("sha256") == formal.EXPECTED_EXPANDED_SHA256
            for result in (history, oracle)
        ),
        "same_committed_config": complete
        and history.get("config_source", {}).get("committed_sha256")
        == oracle.get("config_source", {}).get("committed_sha256")
        and history.get("config_source", {}).get("working_tree_consumed_as_config") is False
        and oracle.get("config_source", {}).get("working_tree_consumed_as_config") is False,
        "same_model_capacity": complete
        and history.get("trainable_parameter_count") == oracle.get("trainable_parameter_count") == formal.EXPECTED_TRAINABLE_PARAMETERS
        and history.get("state_tensor_element_count") == oracle.get("state_tensor_element_count") == formal.EXPECTED_STATE_TENSOR_ELEMENTS,
        "policy_input_difference_limited_to_context_slot": complete
        and history.get("context_contract", {}).get("pass") is True
        and oracle.get("context_contract", {}).get("pass") is True
        and history.get("context_contract", {}).get("source_fields") == oracle.get("context_contract", {}).get("source_fields")
        and history.get("context_contract", {}).get("prohibited_fields_consumed") == []
        and oracle.get("context_contract", {}).get("prohibited_fields_consumed") == []
        and history.get("context_contract", {}).get("branch") == "history_only"
        and oracle.get("context_contract", {}).get("branch") == "oracle"
        and history.get("context_contract", {}).get("observed_digests_by_code")
        != oracle.get("context_contract", {}).get("observed_digests_by_code"),
        "same_scientific_config_hash": complete
        and history.get("scientific_config_sha256")
        == oracle.get("scientific_config_sha256")
        == scientific_config_sha256(),
        "one_epoch_each": complete and history.get("epochs_completed") == oracle.get("epochs_completed") == 1,
        "expected_frames_each": complete and history.get("frames_completed") == oracle.get("frames_completed") == EXPECTED_FRAMES,
        "fresh_optimizer_each": complete
        and history.get("optimizer_state_entries_before") == oracle.get("optimizer_state_entries_before") == 0,
        "optimizer_populated_each": complete
        and int(history.get("optimizer_state_entries_after", 0)) > 0
        and int(oracle.get("optimizer_state_entries_after", 0)) > 0,
        "real_update_each": complete
        and int(history.get("training_counters", {}).get("optimizer_step_count", 0)) > 0
        and int(oracle.get("training_counters", {}).get("optimizer_step_count", 0)) > 0,
        "oracle_context_gradient": complete
        and int(oracle.get("training_counters", {}).get("context_column_nonzero_gradient_calls", 0)) > 0,
        "no_checkpoint_each": complete
        and history.get("no_trained_checkpoint_retained") is True
        and oracle.get("no_trained_checkpoint_retained") is True,
        "no_scientific_conclusion": complete
        and history.get("scientific_conclusion_generated") is False
        and oracle.get("scientific_conclusion_generated") is False,
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "fairness_schema": FAIRNESS_SCHEMA,
        "feasibility_id": FEASIBILITY_ID,
        "schema_version": 1,
        "overall_pass": all(checks.values()),
        "checks": checks,
        "branch_result_paths": {branch: f"{branch}/result.json" for branch in results},
        "scientific_conclusion_generated": False,
    }


def _console_failure_excerpt(text: str) -> list[str]:  # 保留原生 CUDA/PhysX/内存错误行。
    tokens = ("error", "failed", "failure", "cuda", "cublas", "physx", "memory", "alloc", "exception")
    return [line for line in text.splitlines() if any(token in line.lower() for token in tokens)][-200:]


def finalize_console_evidence(
    output_root: Path,
    branch: str,
    console_path: Path,
    exit_code: int,
    capture_token: str,
) -> dict[str, Any]:  # 进程退出后绑定父进程拥有的原生 console 证据。
    if branch not in formal.BRANCHES:
        raise ValueError(f"非法 branch: {branch}")
    output_root = output_root.resolve(strict=True)
    console_path = console_path.resolve(strict=True)
    branch_dir = output_root / branch
    canonical_console = (branch_dir / "console.log").resolve(strict=False)
    if console_path != canonical_console:
        raise RuntimeError(f"NONCANONICAL_CONSOLE_LOG: actual={console_path}, expected={canonical_console}")
    result_path = branch_dir / "result.json"
    console_text = console_path.read_text(encoding="utf-8", errors="replace")
    token_sha256 = hashlib.sha256(capture_token.encode("utf-8")).hexdigest()
    start_sentinel = f"R7_PHASE42B_PARENT_CAPTURE_START token_sha256={token_sha256}"
    completion_sentinel = (
        f"R7_PHASE42B_PARENT_CAPTURE_COMPLETE token_sha256={token_sha256} exit_code={int(exit_code)}"
    )
    if start_sentinel not in console_text or completion_sentinel not in console_text:
        raise RuntimeError("CONSOLE_CAPTURE_SENTINEL_MISSING_OR_MISMATCHED")
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        timeline_path = branch_dir / "resource_timeline.json"
        timeline = json.loads(timeline_path.read_text(encoding="utf-8")) if timeline_path.is_file() else {}
        result = {
            "protocol_id": PROTOCOL_ID,
            "fairness_schema": FAIRNESS_SCHEMA,
            "feasibility_id": FEASIBILITY_ID,
            "schema_version": 1,
            "status": "BLOCKED",
            "branch": branch,
            "seed": SEED,
            "failure_stage": timeline.get("active_stage_id"),
            "error": {"type": "PROCESS_EXIT_WITHOUT_RESULT", "message": f"exit_code={exit_code}"},
            "resource_state": formal.audit.collect_resource_snapshot(),
            "resource_state_provenance": "finalizer_process_after_worker_exit",
            "scientific_conclusion_generated": False,
        }
    result_error_text = json.dumps(result.get("error") or {}, sort_keys=True, default=str)
    combined_error = f"{result_error_text}\n{console_text}"
    signatures = failure_signature_record(combined_error)
    result["post_exit_resource_state"] = formal.audit.collect_resource_snapshot()
    result["post_exit_resource_state_provenance"] = "finalizer_process_after_worker_exit"
    failure_stage = result.get("failure_stage")
    if failure_stage is None:
        timeline_path = branch_dir / "resource_timeline.json"
        if timeline_path.is_file():
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            failure_stage = timeline.get("active_stage_id")
            failed = [item for item in timeline.get("stages", []) if item.get("status") == "FAILED"]
            if failed:
                failure_stage = failed[-1].get("stage_id")
    checks = dict(result.get("checks") or {})
    checks.update(
        {
            "console_evidence_finalized": True,
            "process_exit_zero": int(exit_code) == 0,
            "no_oom": not signatures["oom"],
            "no_pinned_memory_failure": not signatures["pinned_memory"],
            "no_articulation_failure": not signatures["articulation"],
            "no_resource_allocation_failure": not signatures["resource_allocation"],
            "no_trained_checkpoint_retained": not list(branch_dir.rglob("*.pth")),
        }
    )
    runtime_eligible = result.get("status") in ("PASS", "PENDING_CONSOLE_FINALIZATION")
    passed = runtime_eligible and all(checks.values())
    decision = ["BRANCH_BACKWARD_PASS"] if passed else classify_failure(failure_stage, combined_error)
    result.update(
        {
            "status": "PASS" if passed else "BLOCKED",
            "failure_stage": None if passed else failure_stage,
            "failure_signatures": signatures,
            "console_evidence_finalized": True,
            "console_log": str(console_path),
            "console_sha256": formal.audit.sha256_file(console_path),
            "console_capture": {
                "owner": "phase42b_parent_launcher",
                "canonical_path": str(canonical_console),
                "token_sha256": token_sha256,
                "start_sentinel_present": True,
                "completion_sentinel_present": True,
            },
            "console_failure_excerpt": _console_failure_excerpt(console_text),
            "process_exit_code": int(exit_code),
            "checks": checks,
            "decision": decision,
            "no_trained_checkpoint_retained": checks["no_trained_checkpoint_retained"],
            "scientific_conclusion_generated": False,
        }
    )
    formal.audit.atomic_write_json(result_path, result)
    refresh_aggregate_artifacts(output_root)
    return result


def final_decision(results: dict[str, dict[str, Any]]) -> list[str]:  # 输出用户冻结的三类终态。
    if all(branch in results for branch in formal.BRANCHES):
        fairness = build_fairness_report(results)
        if fairness["overall_pass"]:
            return ["FORMAL_6144_BACKWARD_PASS", "READY_FOR_FORMAL_O0_TRAINING"]
    for branch in formal.BRANCHES:
        result = results.get(branch)
        if result and result.get("status") != "PASS":
            return list(result.get("decision", ["FORMAL_TRAINING_PIPELINE_BLOCKED"]))
    return ["FORMAL_TRAINING_PIPELINE_BLOCKED"]


def load_results(output_root: Path) -> dict[str, dict[str, Any]]:  # 只加载已完成的 branch result。
    results: dict[str, dict[str, Any]] = {}
    for branch in formal.BRANCHES:
        path = output_root / branch / "result.json"
        if path.is_file():
            results[branch] = json.loads(path.read_text(encoding="utf-8"))
    return results


def refresh_aggregate_artifacts(output_root: Path) -> list[str]:  # 每次 branch 退出前原子刷新汇总证据。
    results = load_results(output_root)
    fairness = build_fairness_report(results)
    resource_payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "feasibility_id": FEASIBILITY_ID,
        "branches": {},
        "scientific_conclusion_generated": False,
    }
    checkpoint_payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "feasibility_id": FEASIBILITY_ID,
        "branches": {},
        "scientific_conclusion_generated": False,
    }
    for branch, result in results.items():
        timeline_path = output_root / branch / "resource_timeline.json"
        if timeline_path.is_file():
            resource_payload["branches"][branch] = json.loads(timeline_path.read_text(encoding="utf-8"))
        checkpoint_payload["branches"][branch] = {
            "checkpoint_hashes_before": result.get("checkpoint_hashes_before"),
            "checkpoint_hashes_after": result.get("checkpoint_hashes_after"),
            "model_hash_before": result.get("model_hash_before"),
            "model_hash_after": result.get("model_hash_after"),
            "no_trained_checkpoint_retained": result.get("no_trained_checkpoint_retained"),
        }
    decision = final_decision(results)
    formal.audit.atomic_write_json(output_root / "resource_timeline.json", resource_payload)
    formal.audit.atomic_write_json(output_root / "fairness_report.json", fairness)
    formal.audit.atomic_write_json(output_root / "checkpoint_hash.json", checkpoint_payload)
    formal.audit.atomic_write_json(
        output_root / "decision.json",
        {
            "protocol_id": PROTOCOL_ID,
            "feasibility_id": FEASIBILITY_ID,
            "decision": decision,
            "formal_training_automatically_authorized": False,
            "scientific_conclusion_generated": False,
        },
    )
    checksum_path = output_root / "artifact_checksums.json"
    artifacts = {
        str(path.relative_to(output_root)).replace("\\", "/"): formal.audit.sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path != checksum_path
    }
    formal.audit.atomic_write_json(
        checksum_path,
        {
            "protocol_id": PROTOCOL_ID,
            "feasibility_id": FEASIBILITY_ID,
            "artifacts": artifacts,
            "scientific_conclusion_generated": False,
        },
    )
    return decision


def build_parser() -> argparse.ArgumentParser:  # 构建单 branch fresh-process CLI。
    parser = argparse.ArgumentParser(description="R7-O0 6144-env formal backward feasibility gate")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--branch", required=True, choices=formal.BRANCHES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repo-root", type=Path, default=formal.DEFAULT_REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def build_launch_parser() -> argparse.ArgumentParser:  # 构建拥有 stdout/stderr 的 parent launcher CLI。
    parser = argparse.ArgumentParser(description="Launch one isolated R7-O0 Phase 4.2b worker")
    parser.add_argument("--launch-branch", action="store_true", required=True)
    parser.add_argument("--branch", required=True, choices=formal.BRANCHES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repo-root", type=Path, default=formal.DEFAULT_REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--isaaclab-bat", type=Path, default=DEFAULT_ISAACLAB_BAT)
    return parser


def launch_branch_process(args: argparse.Namespace) -> int:  # 父进程独占 worker、console 与退出码。
    validate_cli_contract(args)
    repo_root = args.repo_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    isaaclab_bat = args.isaaclab_bat.expanduser().resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    branch_dir = output_root / args.branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    result_path = branch_dir / "result.json"
    console_path = branch_dir / "console.log"
    if result_path.exists() or console_path.exists():
        raise RuntimeError(f"拒绝覆盖既有 branch artifact: {branch_dir}")
    current_commit = formal.audit.git_text(repo_root, "rev-parse", "HEAD")
    if args.branch == "oracle":
        validate_oracle_precondition(output_root, current_commit)
    capture_token = secrets.token_hex(32)
    token_sha256 = hashlib.sha256(capture_token.encode("utf-8")).hexdigest()
    worker_command = [
        str(isaaclab_bat),
        "-p",
        str(Path(__file__).resolve()),
        "--worker",
        "--branch",
        args.branch,
        "--seed",
        str(args.seed),
        "--repo-root",
        str(repo_root),
        "--output-root",
        str(output_root),
        "--headless",
    ]
    environment = os.environ.copy()
    environment["R7_PHASE42B_CAPTURE_TOKEN"] = capture_token
    environment["R7_PHASE42B_CAPTURE_PARENT_PID"] = str(os.getpid())
    with console_path.open("w", encoding="utf-8", errors="replace") as console_stream:
        console_stream.write(f"R7_PHASE42B_PARENT_CAPTURE_START token_sha256={token_sha256}\n")
        console_stream.flush()
        completed = subprocess.run(
            worker_command,
            cwd=repo_root,
            stdout=console_stream,
            stderr=subprocess.STDOUT,
            env=environment,
            shell=False,
            check=False,
        )
        console_stream.write(
            f"\nR7_PHASE42B_PARENT_CAPTURE_COMPLETE token_sha256={token_sha256} "
            f"exit_code={int(completed.returncode)}\n"
        )
        console_stream.flush()
    finalized = finalize_console_evidence(
        output_root,
        args.branch,
        console_path,
        int(completed.returncode),
        capture_token,
    )
    print("\n".join(finalized["decision"]), flush=True)
    return 0 if finalized["status"] == "PASS" else 2


def main(argv: Sequence[str] | None = None) -> int:  # 每个进程只执行一个 branch。
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--launch-branch" in raw_argv:
        return launch_branch_process(build_launch_parser().parse_args(raw_argv))

    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(raw_argv)
    app_launcher = None
    allocation_timeline = None
    resource_timeline = None
    result: dict[str, Any] | None = None
    try:
        validate_cli_contract(args)
        capture_token = os.environ.get("R7_PHASE42B_CAPTURE_TOKEN")
        capture_parent_pid_text = os.environ.get("R7_PHASE42B_CAPTURE_PARENT_PID")
        if not capture_token or not capture_parent_pid_text:
            raise RuntimeError("WORKER_CONTROLLED_PARENT_ENVIRONMENT_MISSING")
        try:
            capture_parent_pid = int(capture_parent_pid_text)
        except ValueError as exc:
            raise RuntimeError("WORKER_CONTROLLED_PARENT_PID_INVALID") from exc
        args.repo_root = args.repo_root.expanduser().resolve(strict=True)
        args.output_root = args.output_root.expanduser().resolve(strict=False)
        args.output_root.mkdir(parents=True, exist_ok=True)
        branch_dir = args.output_root / args.branch
        branch_dir.mkdir(parents=True, exist_ok=True)
        result_path = branch_dir / "result.json"
        if result_path.exists():
            raise RuntimeError(f"branch result 已存在，拒绝覆盖: {result_path}")
        console_path = branch_dir / "console.log"
        if not console_path.is_file():
            raise RuntimeError(f"EXTERNAL_CONSOLE_CAPTURE_REQUIRED: {console_path}")
        current_commit = formal.audit.git_text(args.repo_root, "rev-parse", "HEAD")
        if args.branch == "oracle":
            validate_oracle_precondition(args.output_root, current_commit)
        config, config_record = formal.load_committed_agent_config(args.repo_root)
        del config
        checkpoint_before = formal.checkpoint_preflight(args.repo_root)
        clean_record = feasibility_clean_process_gate(args.repo_root, capture_parent_pid)
        if not clean_record["pass"]:
            raise RuntimeError(f"clean process gate 发现残留: {clean_record['residuals']}")
        formal.audit.atomic_write_json(
            branch_dir / "preflight_system_state.json",
            {
                "protocol_id": PROTOCOL_ID,
                "feasibility_id": FEASIBILITY_ID,
                "branch": args.branch,
                "git_commit": current_commit,
                "parent_engineering_commit": PARENT_ENGINEERING_COMMIT,
                "config_source": config_record,
                "checkpoint": checkpoint_before,
                "clean_process_gate": clean_record,
                "console_capture_admission": {
                    "owner": "phase42b_parent_launcher",
                    "parent_pid": capture_parent_pid,
                    "token_sha256": hashlib.sha256(capture_token.encode("utf-8")).hexdigest(),
                    "token_persisted_in_command_line": False,
                },
                "resources": formal.audit.collect_resource_snapshot(),
                "scientific_conclusion_generated": False,
            },
        )
        allocation_timeline = formal.FormalTimeline(branch_dir / "allocation_timeline.json")
        resource_timeline = BackwardResourceTimeline(branch_dir / "resource_timeline.json", args.branch)
        allocation_timeline.record(0)
        app_launcher = AppLauncher(args)
        allocation_timeline.record(1)
        result = run_branch_epoch(args, app_launcher, allocation_timeline, resource_timeline, branch_dir)
        result["checkpoint_hashes_before"] = checkpoint_before
        result["checkpoint_hashes_after"] = formal.checkpoint_preflight(args.repo_root)
        result["decision"] = (
            ["PENDING_CONSOLE_FINALIZATION"]
            if result["status"] == "PENDING_CONSOLE_FINALIZATION"
            else ["FORMAL_TRAINING_PIPELINE_BLOCKED"]
        )
    except Exception as exc:
        if allocation_timeline is not None:
            allocation_timeline.fail(exc)
        if resource_timeline is not None:
            resource_timeline.fail(exc)
        branch = getattr(args, "branch", "unknown")
        branch_dir = getattr(args, "output_root", DEFAULT_OUTPUT_ROOT) / branch
        branch_dir.mkdir(parents=True, exist_ok=True)
        full_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        failed_stage = resource_timeline.failed_stage_id if resource_timeline is not None else None
        decision = classify_failure(failed_stage, full_error)
        try:
            failure_commit = formal.audit.git_text(args.repo_root, "rev-parse", "HEAD")
        except Exception as git_exc:
            failure_commit = None
            full_error = f"{full_error}\nGit audit failed: {type(git_exc).__name__}: {git_exc}"
        result = {
            "protocol_id": PROTOCOL_ID,
            "fairness_schema": FAIRNESS_SCHEMA,
            "feasibility_id": FEASIBILITY_ID,
            "schema_version": 1,
            "status": "BLOCKED",
            "branch": branch,
            "seed": getattr(args, "seed", None),
            "process_id": os.getpid(),
            "timestamp_utc": utc_now(),
            "git_commit": failure_commit,
            "parent_engineering_commit": PARENT_ENGINEERING_COMMIT,
            "failure_stage": failed_stage,
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "failure_signatures": failure_signature_record(full_error),
            "resource_state": formal.audit.collect_resource_snapshot(),
            "decision": decision,
            "automatic_fallback_applied": False,
            "execution_alternatives_not_applied": [
                "lower_num_envs",
                "lower_minibatch_size",
                "lower_horizon",
                "change_mixed_precision",
                "change_network",
            ],
            "no_trained_checkpoint_retained": not list(branch_dir.rglob("*.pth")),
            "scientific_conclusion_generated": False,
        }
        try:
            result["checkpoint_hashes_after"] = formal.checkpoint_preflight(args.repo_root)
        except Exception as checkpoint_exc:
            result["checkpoint_postfailure_audit_error"] = f"{type(checkpoint_exc).__name__}: {checkpoint_exc}"
    finally:
        if app_launcher is not None:
            try:
                app_launcher.app.close()
            except Exception as shutdown_exc:
                if result is not None:
                    result["shutdown_error"] = f"{type(shutdown_exc).__name__}: {shutdown_exc}"
                    result["status"] = "BLOCKED"
                    result["decision"] = ["FORMAL_TRAINING_PIPELINE_BLOCKED"]
    result_path = args.output_root / args.branch / "result.json"
    formal.audit.atomic_write_json(result_path, result)
    decision = refresh_aggregate_artifacts(args.output_root)
    process_status = result["status"]
    pending = process_status == "PENDING_CONSOLE_FINALIZATION"
    print("\n".join(["BRANCH_EXECUTION_COMPLETE_PENDING_CONSOLE_FINALIZATION"] if pending else decision), flush=True)
    return 0 if process_status in ("PASS", "PENDING_CONSOLE_FINALIZATION") else 2


if __name__ == "__main__":
    sys.exit(main())
