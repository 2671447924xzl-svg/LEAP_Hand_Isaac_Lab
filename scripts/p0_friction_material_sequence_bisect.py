"""P0-7F0.2 anomalous material-sequence exact replay and conditional differential bisect."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
import types
from datetime import datetime
from pathlib import Path
from typing import Any


TASK_ID = "Isaac-Reorient-Cube-Leap-CRI"  # 复用冻结CRI任务。
FACTOR_CODES = ("000", "100", "010", "001")  # 显式四环境assignment。
TARGET_ENV_ID = 2  # 只检查friction factor环境。
DIRECT_PREFIX = (0.45, 0.40, 0.35)  # 精确重放旧direct-case前缀。
EXPECTED_RESET_FRICTION = 0.45  # reset应恢复的canonical值。
ANOMALOUS_FRICTION = 0.35  # 历史异常残留值。
BASE_SEED = 42  # 与旧诊断一致。
ATOL = 1.0e-6  # 与CRI material断言一致。
SAME_PROCESS_LIFECYCLES = 5  # 冻结的同进程fresh lifecycle数量。
FRESH_PROCESS_RUNS = 5  # 冻结的全新simulator process数量。
EVIDENCE_LABEL = "P0_7F0_2_DIAGNOSIS_ONLY_NOT_FOR_QUALIFICATION"  # 禁止用于qualification。
INSTRUMENTATION_MODES = ("NO_MONKEYPATCH", "OLD_TRACE_RECORDER_EQUIVALENT", "NEW_TRACE_RECORDER_EQUIVALENT")
ROOT_CAUSE_CLASSES = (
    "DIAGNOSTIC_INSTRUMENTATION_ARTIFACT",
    "SEQUENCE_DEPENDENT_MATERIAL_STATE_DEFECT",
    "INTERMITTENT_PHYSX_MATERIAL_BEHAVIOR",
    "HISTORICAL_ANOMALY_NOT_REPRODUCED",
    "ROOT_CAUSE_UNRESOLVED",
)  # 只允许科研审核批准的分类。
PROTECTED_PATHS = (
    "scripts/p0_physics_validation.py",
    "scripts/p0_timing_audit.py",
    "scripts/p0_friction_coupling_diagnosis.py",
    "scripts/p0_friction_material_diagnosis.py",
    "scripts/p0_friction_material_sync_diagnosis.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/factor_conditions.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/leap_hand_cri_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_IsaacLab/tasks/leap_hand_reorient_cri/reorientation_cri_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml",
    "logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth",
)  # 新脚本运行前后必须保持字节不变。


def build_pre_parser() -> argparse.ArgumentParser:  # 父编排器不启动AppLauncher。
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--static_only", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("logs/p0_friction_material_sequence_bisect"))
    parser.add_argument(
        "--worker_mode", choices=("none", "lifecycle_batch", "process_exact", "differential"), default="none"
    )
    parser.add_argument("--worker_output", type=Path)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser


PRE_ARGS, _ = build_pre_parser().parse_known_args()  # 先区分父进程与仿真worker。


def validate_constants() -> None:  # 启动任何进程前拒绝规格漂移。
    assert FACTOR_CODES == ("000", "100", "010", "001")
    assert DIRECT_PREFIX == (0.45, 0.40, 0.35)
    assert SAME_PROCESS_LIFECYCLES == 5
    assert FRESH_PROCESS_RUNS == 5
    assert INSTRUMENTATION_MODES == (
        "NO_MONKEYPATCH",
        "OLD_TRACE_RECORDER_EQUIVALENT",
        "NEW_TRACE_RECORDER_EQUIVALENT",
    )
    assert len(ROOT_CAUSE_CLASSES) == 5


validate_constants()
if PRE_ARGS.static_only:
    print("P0_7F0_2_STATIC_CONTRACT_PASS")
    raise SystemExit(0)


APP_LAUNCHER = None  # 父进程保持无SimulationApp状态。
SIMULATION_APP = None
ARGS = PRE_ARGS
if PRE_ARGS.worker_mode != "none":
    from isaaclab.app import AppLauncher

    def build_worker_parser() -> argparse.ArgumentParser:  # worker独立解析Isaac启动参数。
        parser = argparse.ArgumentParser(description="P0-7F0.2 isolated simulator worker.")
        parser.add_argument("--static_only", action="store_true")
        parser.add_argument("--output_dir", type=Path, default=PRE_ARGS.output_dir)
        parser.add_argument(
            "--worker_mode", choices=("lifecycle_batch", "process_exact", "differential"), required=True
        )
        parser.add_argument("--worker_output", type=Path, required=True)
        parser.add_argument("--worker_index", type=int, default=0)
        AppLauncher.add_app_launcher_args(parser)
        return parser

    ARGS = build_worker_parser().parse_args()
    APP_LAUNCHER = AppLauncher(ARGS)  # 每个worker拥有独立SimulationApp。
    SIMULATION_APP = APP_LAUNCHER.app

    import gymnasium as gym
    import torch

    import LEAP_Isaaclab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg


def sha256_file(path: Path) -> str:  # 流式计算证据与保护文件哈希。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes(repo_root: Path) -> dict[str, str | None]:  # 不存在项显式记为null。
    return {path: sha256_file(repo_root / path) if (repo_root / path).is_file() else None for path in PROTECTED_PATHS}


def git_text(repo_root: Path, *args: str) -> str:  # 只读记录代码身份。
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *args), capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else f"UNAVAILABLE:{completed.stderr.strip()}"


def json_safe(value: Any) -> Any:  # 严格JSON转换，未初始化NaN写为null。
    if PRE_ARGS.worker_mode != "none" and isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "values": json_safe(value.detach().cpu().tolist()),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:  # create-only保存证据。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> dict[str, Any]:  # 父进程只读聚合worker结果。
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def close_enough(actual: float, expected: float) -> bool:  # 使用冻结容差判断reset结果。
    return abs(actual - expected) <= ATOL


def create_env() -> Any:  # 保持原task、dt、decimation、material API和reset流程。
    cfg = parse_env_cfg(TASK_ID, device=ARGS.device, num_envs=len(FACTOR_CODES))
    cfg.factor_codes = FACTOR_CODES
    cfg.events = None
    cfg.enable_adr = False
    env = gym.make(TASK_ID, cfg=cfg)
    core = env.unwrapped
    if tuple(core.factor_code_labels) != FACTOR_CODES:
        env.close()
        raise RuntimeError("Explicit factor assignment mismatch")
    if core.cfg.events is not None or core.cfg.enable_adr:
        env.close()
        raise RuntimeError("P0-7F0.2 requires events=None and ADR disabled")
    return env


def material_state(core: Any, raw_read: Any) -> dict[str, Any]:  # 分离getter和科研buffer状态。
    materials = raw_read(core.object, "object")
    return {
        "material_tensor": materials,
        "getter_static_all_envs": materials[:, 0, 0],
        "getter_dynamic_all_envs": materials[:, 0, 1],
        "getter_static_target": float(materials[TARGET_ENV_ID, 0, 0]),
        "getter_dynamic_target": float(materials[TARGET_ENV_ID, 0, 1]),
        "requested_static_all_envs": core.requested_object_static_friction.clone(),
        "actual_static_buffer_all_envs": core.actual_object_static_friction.clone(),
        "physics_step_count": int(core.sim.get_physics_step_count()),
        "simulation_time": float(core.sim.physics_manager.get_simulation_time()),
    }


class CoreCallProfiler:  # 使用sys profile计数，不改变method binding。
    def __init__(self, core: Any):  # 锁定原始CRI方法code object。
        self.counts = {
            "_write_object_material": 0,
            "_read_materials": 0,
            "_read_back_factor_values": 0,
            "_assert_requested_matches_actual": 0,
            "reset": 0,
        }
        self.codes = {
            name: getattr(core, name).__func__.__code__
            for name in (
                "_write_object_material",
                "_read_materials",
                "_read_back_factor_values",
                "_assert_requested_matches_actual",
            )
        }
        self.setter_calls: list[dict[str, Any]] = []

    def callback(self, frame: Any, event: str, arg: Any) -> None:  # 只观察函数call事件。
        if event != "call":
            return
        for name, code in self.codes.items():
            if frame.f_code is code:
                self.counts[name] += 1
                if name == "_write_object_material":
                    caller = frame.f_back.f_code.co_name if frame.f_back is not None else "UNAVAILABLE"
                    self.setter_calls.append(
                        {
                            "caller": caller,
                            "env_ids": json_safe(frame.f_locals.get("env_ids")),
                            "requested_static_friction": json_safe(frame.f_locals.get("static_friction")),
                            "getter_before": "NOT_OBSERVED_BY_PROFILE",
                            "getter_after": "NOT_OBSERVED_BY_PROFILE",
                        }
                    )
                break

    def start(self) -> None:  # 启用无binding变更的调用计数。
        sys.setprofile(self.callback)

    def stop(self) -> None:  # 立即移除profile hook。
        sys.setprofile(None)


class TraceRecorderBase:  # 保存旧/新wrapper共同的原始方法与事件。
    def __init__(self, core: Any, mode: str):  # 每个fresh lifecycle使用独立recorder。
        self.core = core
        self.mode = mode
        self.phase = "UNSET"
        self.events: list[dict[str, Any]] = []
        self.original_reset_idx = core._reset_idx
        self.original_apply = core._apply_factor_conditions
        self.original_read = core._read_materials
        self.original_write = core._write_object_material
        self.original_read_back = core._read_back_factor_values
        self.original_assert = core._assert_requested_matches_actual

    def append(self, function: str, payload: dict[str, Any]) -> None:  # wrapper记录不触发额外仿真操作。
        self.events.append({"phase": self.phase, "function": function, **json_safe(payload)})

    def install_old_equivalent(self) -> None:  # 逐句保持旧TraceRecorder的调用行为。
        recorder = self

        def traced_read(this: Any, asset: Any, asset_name: str) -> Any:
            result = recorder.original_read(asset, asset_name)
            recorder.append("_read_materials", {"asset_name": asset_name, "material_tensor": result})
            return result

        def traced_write(this: Any, env_ids: Any, static_friction: Any, dynamic_friction: Any) -> None:
            before = recorder.original_read(this.object, "object")
            cpu_ids = env_ids.to(device="cpu", dtype=torch.int64)
            selected_before = before.cpu()[cpu_ids]
            recorder.original_write(env_ids, static_friction, dynamic_friction)
            after = recorder.original_read(this.object, "object")
            caller = sys._getframe(1).f_code.co_name
            recorder.append(
                "_write_object_material",
                {
                    "caller": caller,
                    "env_ids": env_ids,
                    "requested_static_friction": static_friction,
                    "requested_dynamic_friction": dynamic_friction,
                    "getter_before": float(before[TARGET_ENV_ID, 0, 0]),
                    "getter_after": float(after[TARGET_ENV_ID, 0, 0]),
                    "selected_shape": list(selected_before.shape),
                },
            )

        def traced_read_back(this: Any, env_ids: Any) -> None:
            before = this.actual_object_static_friction.clone()
            recorder.original_read_back(env_ids)
            recorder.append(
                "_read_back_factor_values",
                {"env_ids": env_ids, "actual_before": before, "actual_after": this.actual_object_static_friction},
            )

        def traced_assert(this: Any, env_ids: Any) -> None:
            payload = {
                "env_ids": env_ids,
                "requested": this.requested_object_static_friction[env_ids],
                "actual": this.actual_object_static_friction[env_ids],
            }
            try:
                recorder.original_assert(env_ids)
            except Exception as exc:
                recorder.append("_assert_requested_matches_actual", {**payload, "pass": False, "error": str(exc)})
                raise
            recorder.append("_assert_requested_matches_actual", {**payload, "pass": True, "error": None})

        self.core._read_materials = types.MethodType(traced_read, self.core)
        self.core._write_object_material = types.MethodType(traced_write, self.core)
        self.core._read_back_factor_values = types.MethodType(traced_read_back, self.core)
        self.core._assert_requested_matches_actual = types.MethodType(traced_assert, self.core)

    def install_new_equivalent(self) -> None:  # 在旧等价wrapper上增加F0.1的reset/apply边界包装。
        self.install_old_equivalent()
        recorder = self

        def traced_reset_idx(this: Any, env_ids: Any) -> Any:
            recorder.append("_reset_idx_enter", {"env_ids": env_ids})
            try:
                result = recorder.original_reset_idx(env_ids)
            except Exception as exc:
                recorder.append("_reset_idx_error", {"env_ids": env_ids, "error": str(exc)})
                raise
            recorder.append("_reset_idx_exit", {"env_ids": env_ids})
            return result

        def traced_apply(this: Any, env_ids: Any) -> Any:
            recorder.append("_apply_factor_conditions_enter", {"env_ids": env_ids})
            try:
                result = recorder.original_apply(env_ids)
            except Exception as exc:
                recorder.append("_apply_factor_conditions_error", {"env_ids": env_ids, "error": str(exc)})
                raise
            recorder.append("_apply_factor_conditions_exit", {"env_ids": env_ids})
            return result

        self.core._reset_idx = types.MethodType(traced_reset_idx, self.core)
        self.core._apply_factor_conditions = types.MethodType(traced_apply, self.core)


def configure_instrumentation(core: Any, mode: str) -> TraceRecorderBase:  # 明确三种binding条件。
    recorder = TraceRecorderBase(core, mode)
    if mode == "OLD_TRACE_RECORDER_EQUIVALENT":
        recorder.install_old_equivalent()
    elif mode == "NEW_TRACE_RECORDER_EQUIVALENT":
        recorder.install_new_equivalent()
    elif mode != "NO_MONKEYPATCH":
        raise ValueError(f"Unknown instrumentation mode: {mode}")
    return recorder


def attempt_reset(env: Any, recorder: TraceRecorderBase, phase: str) -> dict[str, Any]:  # 原样调用reset并捕获断言。
    recorder.phase = phase
    error = None
    try:
        env.reset(seed=BASE_SEED)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    return {"pass": error is None, "error": error}


def write_then_read(core: Any, recorder: TraceRecorderBase, friction: float, phase: str) -> dict[str, Any]:  # 复制旧direct case。
    recorder.phase = phase
    env_ids = torch.tensor((TARGET_ENV_ID,), dtype=torch.long, device=core.device)
    values = torch.tensor((friction,), dtype=torch.float32, device=core.device)
    before = material_state(core, recorder.original_read)
    core._write_object_material(env_ids, values, values)
    after_write = material_state(core, recorder.original_read)
    core._read_back_factor_values(env_ids)
    after_readback = material_state(core, recorder.original_read)
    return {
        "requested": friction,
        "before": before,
        "after_write": after_write,
        "after_readback": after_readback,
        "match": close_enough(after_readback["getter_static_target"], friction),
    }


def enrich_setter_calls(profiler: CoreCallProfiler, recorder: TraceRecorderBase) -> list[dict[str, Any]]:  # 优先使用wrapper真实前后getter。
    wrapper_calls = [event for event in recorder.events if event["function"] == "_write_object_material"]
    if not wrapper_calls:
        return profiler.setter_calls
    return [
        {
            "caller": event.get("caller"),
            "env_ids": event.get("env_ids"),
            "requested_static_friction": event.get("requested_static_friction"),
            "getter_before": event.get("getter_before"),
            "getter_after": event.get("getter_after"),
        }
        for event in wrapper_calls
    ]


def classify_reset_value(value: float) -> str:  # 只区分历史异常、正常值和其他值。
    if close_enough(value, ANOMALOUS_FRICTION):
        return "OLD_ANOMALY_REPRODUCED"
    if close_enough(value, EXPECTED_RESET_FRICTION):
        return "OLD_ANOMALY_NOT_REPRODUCED"
    return "UNEXPECTED_RESET_VALUE"


def run_exact_lifecycle(run_index: int, lifecycle_id: str, instrumentation_mode: str) -> dict[str, Any]:  # 精确重放后立即停止。
    env = create_env()
    core = env.unwrapped
    profiler = CoreCallProfiler(core)
    recorder = configure_instrumentation(core, instrumentation_mode)
    profiler.start()
    fatal_error = None
    result: dict[str, Any] = {}
    try:
        initial_reset = attempt_reset(env, recorder, "INITIAL_PARENT_RESET")
        initial_state = material_state(core, recorder.original_read)
        direct_cases = [
            write_then_read(core, recorder, friction, f"DIRECT_CASE_{friction:.2f}") for friction in DIRECT_PREFIX
        ]
        recorder.phase = "CANDIDATE_0.45_BEFORE_RESET"
        before_reset = material_state(core, recorder.original_read)
        reset = attempt_reset(env, recorder, "CANDIDATE_0.45_RESET")
        after_reset = material_state(core, recorder.original_read)
        result = {
            "process_id": os.getpid(),
            "lifecycle_id": lifecycle_id,
            "run_index": run_index,
            "instrumentation_mode": instrumentation_mode,
            "initial_reset": initial_reset,
            "initial_state": initial_state,
            "direct_cases": direct_cases,
            "before_reset": before_reset,
            "reset": reset,
            "after_reset": after_reset,
            "after_reset_env2_getter": after_reset["getter_static_target"],
            "outcome": classify_reset_value(after_reset["getter_static_target"]),
            "physics_step_count": after_reset["physics_step_count"],
            "simulation_time": after_reset["simulation_time"],
        }
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        profiler.stop()
        profiler.counts["reset"] = 2
        result["call_counts"] = profiler.counts
        result["setter_calls"] = enrich_setter_calls(profiler, recorder)
        result["trace_events"] = recorder.events
        result["fatal_error"] = fatal_error
        env.close()
    return result


def direct_write(core: Any, friction: float, refresh: bool) -> dict[str, Any]:  # sequence bisect使用的单值setter。
    env_ids = torch.tensor((TARGET_ENV_ID,), dtype=torch.long, device=core.device)
    values = torch.tensor((friction,), dtype=torch.float32, device=core.device)
    core._write_object_material(env_ids, values, values)
    if refresh:
        core._read_back_factor_values(env_ids)
    materials = core._read_materials(core.object, "object")
    return {"requested": friction, "refresh": refresh, "getter": float(materials[TARGET_ENV_ID, 0, 0])}


def run_bisect_sequence(sequence: str) -> dict[str, Any]:  # 每个S序列使用fresh env lifecycle。
    env = create_env()
    core = env.unwrapped
    mode = "OLD_TRACE_RECORDER_EQUIVALENT" if sequence == "S5" else "NO_MONKEYPATCH"
    profiler = CoreCallProfiler(core)
    recorder = configure_instrumentation(core, mode)
    profiler.start()
    fatal_error = None
    result: dict[str, Any] = {"sequence": sequence, "instrumentation_mode": mode, "process_id": os.getpid()}
    try:
        initial_reset = attempt_reset(env, recorder, f"{sequence}_INITIAL_RESET")
        writes: list[dict[str, Any]] = []
        if sequence == "S0":
            writes.append(direct_write(core, 0.35, refresh=False))
        elif sequence == "S1":
            writes.extend((direct_write(core, 0.45, False), direct_write(core, 0.35, False)))
        elif sequence == "S2":
            writes.extend((direct_write(core, 0.40, False), direct_write(core, 0.35, False)))
        elif sequence == "S3":
            writes.extend(
                (direct_write(core, 0.45, False), direct_write(core, 0.40, False), direct_write(core, 0.35, False))
            )
        elif sequence == "S4":
            writes.extend(
                (direct_write(core, 0.45, True), direct_write(core, 0.40, True), direct_write(core, 0.35, True))
            )
        elif sequence == "S5":
            material_state(core, recorder.original_read)
            writes.extend(
                write_then_read(core, recorder, friction, f"S5_DIRECT_CASE_{friction:.2f}") for friction in DIRECT_PREFIX
            )
            material_state(core, recorder.original_read)
        else:
            raise ValueError(f"Unknown sequence: {sequence}")
        before_reset = material_state(core, recorder.original_read)
        reset = attempt_reset(env, recorder, f"{sequence}_RESET")
        after_reset = material_state(core, recorder.original_read)
        result.update(
            {
                "initial_reset": initial_reset,
                "writes": writes,
                "before_reset": before_reset,
                "reset": reset,
                "after_reset": after_reset,
                "after_reset_env2_getter": after_reset["getter_static_target"],
                "status": "FAIL" if close_enough(after_reset["getter_static_target"], ANOMALOUS_FRICTION) else "PASS",
                "outcome": classify_reset_value(after_reset["getter_static_target"]),
            }
        )
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        profiler.stop()
        profiler.counts["reset"] = 2
        result["call_counts"] = profiler.counts
        result["setter_calls"] = enrich_setter_calls(profiler, recorder)
        result["trace_events"] = recorder.events
        result["fatal_error"] = fatal_error
        env.close()
    return result


def instrumentation_effect(parity: dict[str, dict[str, Any]]) -> str:  # 返回冻结的四种effect标签。
    anomalous = {mode: result["outcome"] == "OLD_ANOMALY_REPRODUCED" for mode, result in parity.items()}
    if anomalous["NO_MONKEYPATCH"]:
        return "none"
    old = anomalous["OLD_TRACE_RECORDER_EQUIVALENT"]
    new = anomalous["NEW_TRACE_RECORDER_EQUIVALENT"]
    if old and new:
        return "both"
    if old:
        return "old wrapper only"
    if new:
        return "new wrapper only"
    return "none"


def run_worker() -> int:  # worker只执行指定仿真分支并写一个结果文件。
    output_path = ARGS.worker_output.resolve()
    protected_before = protected_hashes(Path(__file__).resolve().parents[1])
    fatal_error = None
    payload: dict[str, Any] = {
        "evidence_label": EVIDENCE_LABEL,
        "qualification_eligible": False,
        "worker_mode": ARGS.worker_mode,
        "worker_index": ARGS.worker_index,
        "process_id": os.getpid(),
    }
    try:
        if ARGS.worker_mode == "lifecycle_batch":
            payload["results"] = [
                run_exact_lifecycle(index, f"same_process_lifecycle_{index}", "OLD_TRACE_RECORDER_EQUIVALENT")
                for index in range(SAME_PROCESS_LIFECYCLES)
            ]
        elif ARGS.worker_mode == "process_exact":
            payload["result"] = run_exact_lifecycle(
                ARGS.worker_index, f"fresh_process_{ARGS.worker_index}", "OLD_TRACE_RECORDER_EQUIVALENT"
            )
        elif ARGS.worker_mode == "differential":
            parity = {
                mode: run_exact_lifecycle(index, f"instrumentation_{index}", mode)
                for index, mode in enumerate(INSTRUMENTATION_MODES)
            }
            effect = instrumentation_effect(parity)
            payload["instrumentation_parity"] = parity
            payload["instrumentation_effect"] = effect
            wrapper_artifact = (
                parity["NO_MONKEYPATCH"]["outcome"] == "OLD_ANOMALY_NOT_REPRODUCED"
                and effect in ("old wrapper only", "new wrapper only", "both")
            )
            if wrapper_artifact:
                payload["stop_reason"] = "DIAGNOSTIC_INSTRUMENTATION_ARTIFACT"
                payload["sequence_bisect"] = []
            else:
                payload["sequence_bisect"] = [run_bisect_sequence(f"S{index}") for index in range(6)]
                payload["stop_reason"] = None
        else:
            raise ValueError(f"Unknown worker mode: {ARGS.worker_mode}")
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    protected_after = protected_hashes(Path(__file__).resolve().parents[1])
    payload.update(
        {
            "fatal_error": fatal_error,
            "protected_hashes_before": protected_before,
            "protected_hashes_after": protected_after,
            "protected_unchanged": protected_before == protected_after,
        }
    )
    write_json(output_path, payload)
    print(f"P0_7F0_2_WORKER_OUTPUT={output_path}", flush=True)
    print(f"P0_7F0_2_WORKER_STATUS={'FAILED_TECHNICAL' if fatal_error else 'COLLECTED'}", flush=True)
    return 1 if fatal_error else 0


def launch_worker(
    script_path: Path, output_path: Path, worker_mode: str, worker_index: int, device: str | None
) -> dict[str, Any]:  # 父进程顺序启动且等待唯一worker。
    command = [
        sys.executable,
        str(script_path),
        "--worker_mode",
        worker_mode,
        "--worker_output",
        str(output_path),
        "--worker_index",
        str(worker_index),
        "--viz",
        "none",
    ]
    if device is not None:
        command.extend(("--device", device))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path = output_path.with_suffix(".log")
    log_path.write_text(
        f"COMMAND={json.dumps(command)}\nRETURN_CODE={completed.returncode}\n--- STDOUT ---\n{completed.stdout}\n--- STDERR ---\n{completed.stderr}",
        encoding="utf-8",
    )
    if not output_path.is_file():
        return {
            "fatal_error": {
                "type": "WORKER_OUTPUT_MISSING",
                "message": f"worker return_code={completed.returncode}; see {log_path}",
            },
            "worker_log": str(log_path),
        }
    result = read_json(output_path)
    result["worker_return_code"] = completed.returncode
    result["worker_log"] = str(log_path)
    return result


def extract_exact_results(batch: dict[str, Any], processes: list[dict[str, Any]]) -> list[dict[str, Any]]:  # 统一10次结果。
    results = list(batch.get("results", []))
    results.extend(item["result"] for item in processes if "result" in item)
    return results


def compare_call_counts(failing: dict[str, Any] | None, passing: dict[str, Any] | None) -> dict[str, Any]:  # 输出最小计数差异。
    if failing is None or passing is None:
        return {"available": False, "reason": "current run lacks both a failing and passing exact path"}
    keys = ("_write_object_material", "_read_materials", "_read_back_factor_values", "_assert_requested_matches_actual", "reset")
    return {
        "available": True,
        "failing_identity": {
            "process_id": failing.get("process_id"),
            "lifecycle_id": failing.get("lifecycle_id"),
            "instrumentation_mode": failing.get("instrumentation_mode"),
        },
        "passing_identity": {
            "process_id": passing.get("process_id"),
            "lifecycle_id": passing.get("lifecycle_id"),
            "instrumentation_mode": passing.get("instrumentation_mode"),
        },
        "counts": {
            key: {
                "old_failing": failing["call_counts"].get(key),
                "new_passing": passing["call_counts"].get(key),
                "difference": passing["call_counts"].get(key) - failing["call_counts"].get(key),
            }
            for key in keys
        },
        "failing_setter_calls": failing.get("setter_calls", []),
        "passing_setter_calls": passing.get("setter_calls", []),
    }


def orchestrate() -> int:  # 父进程聚合5 lifecycle、5 process和条件差分。
    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_output = PRE_ARGS.output_dir if PRE_ARGS.output_dir.is_absolute() else repo_root / PRE_ARGS.output_dir
    output_dir = (base_output / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    workers_dir = output_dir / "workers"
    workers_dir.mkdir()
    protected_before = protected_hashes(repo_root)
    fatal_error = None
    batch: dict[str, Any] = {}
    process_workers: list[dict[str, Any]] = []
    differential: dict[str, Any] | None = None
    try:
        batch_path = workers_dir / "same_process_lifecycle_batch.json"
        batch = launch_worker(script_path, batch_path, "lifecycle_batch", 0, PRE_ARGS.device)
        if batch.get("fatal_error"):
            raise RuntimeError(f"same-process worker failed: {batch['fatal_error']}")
        for index in range(FRESH_PROCESS_RUNS):
            process_path = workers_dir / f"fresh_process_{index}.json"
            worker = launch_worker(script_path, process_path, "process_exact", index, PRE_ARGS.device)
            process_workers.append(worker)
            if worker.get("fatal_error"):
                raise RuntimeError(f"fresh-process worker {index} failed: {worker['fatal_error']}")
        exact_results = extract_exact_results(batch, process_workers)
        reproduced = [result for result in exact_results if result["outcome"] == "OLD_ANOMALY_REPRODUCED"]
        if reproduced:
            differential_path = workers_dir / "differential.json"
            differential = launch_worker(script_path, differential_path, "differential", 0, PRE_ARGS.device)
            if differential.get("fatal_error"):
                raise RuntimeError(f"differential worker failed: {differential['fatal_error']}")
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}

    exact_results = extract_exact_results(batch, process_workers)
    reproduced = [result for result in exact_results if result.get("outcome") == "OLD_ANOMALY_REPRODUCED"]
    non_reproduced = [result for result in exact_results if result.get("outcome") == "OLD_ANOMALY_NOT_REPRODUCED"]
    instrumentation = differential.get("instrumentation_effect", "none") if differential else "none"
    sequence_results = differential.get("sequence_bisect", []) if differential else []
    minimal_sequence = next((result["sequence"] for result in sequence_results if result.get("status") == "FAIL"), None)

    if fatal_error:
        root_cause = "ROOT_CAUSE_UNRESOLVED"
        root_reason = "technical failure prevented the frozen replay matrix from completing"
    elif len(exact_results) == SAME_PROCESS_LIFECYCLES + FRESH_PROCESS_RUNS and not reproduced:
        root_cause = "HISTORICAL_ANOMALY_NOT_REPRODUCED"
        root_reason = "all five same-process lifecycles and five fresh simulator processes reset to 0.45"
    elif differential and differential.get("stop_reason") == "DIAGNOSTIC_INSTRUMENTATION_ARTIFACT":
        root_cause = "DIAGNOSTIC_INSTRUMENTATION_ARTIFACT"
        root_reason = f"no-monkeypatch passed while instrumentation effect was {instrumentation}"
    elif minimal_sequence is not None:
        root_cause = "SEQUENCE_DEPENDENT_MATERIAL_STATE_DEFECT"
        root_reason = f"first anomalous reset in the frozen bisect order was {minimal_sequence}"
    elif reproduced and non_reproduced:
        root_cause = "INTERMITTENT_PHYSX_MATERIAL_BEHAVIOR"
        root_reason = f"exact replay reproduced {len(reproduced)}/{len(exact_results)} times without a deterministic bisect trigger"
    else:
        root_cause = "ROOT_CAUSE_UNRESOLVED"
        root_reason = "completed evidence does not satisfy another frozen classification rule"

    parity_results = list(differential.get("instrumentation_parity", {}).values()) if differential else []
    failing = next(iter(reproduced), None)
    passing = next((result for result in parity_results if result.get("outcome") == "OLD_ANOMALY_NOT_REPRODUCED"), None)
    if passing is None:
        passing = next(iter(non_reproduced), None)
    call_comparison = compare_call_counts(failing, passing)
    protected_after = protected_hashes(repo_root)
    summary = {
        "evidence_label": EVIDENCE_LABEL,
        "qualification_eligible": False,
        "run_id": run_id,
        "task_id": TASK_ID,
        "factor_codes": list(FACTOR_CODES),
        "target_env_id": TARGET_ENV_ID,
        "direct_prefix": list(DIRECT_PREFIX),
        "same_process_lifecycle_results": batch.get("results", []),
        "fresh_simulator_process_results": [item.get("result") for item in process_workers],
        "exact_replay_total": len(exact_results),
        "exact_replay_reproduced": len(reproduced),
        "old_anomaly_reproduced": bool(reproduced),
        "differential_executed": differential is not None,
        "instrumentation_effect": instrumentation,
        "instrumentation_parity": differential.get("instrumentation_parity") if differential else None,
        "sequence_bisect": sequence_results,
        "minimal_sequence": minimal_sequence,
        "call_count_comparison": call_comparison,
        "root_cause_classification": root_cause,
        "root_cause_reason": root_reason,
        "core_modification_recommendation": "NO CORE CHANGE YET",
        "overall_status": "P0-7F0.2 COMPLETE" if fatal_error is None else "P0-7F0.2 FAILED_TECHNICAL",
        "fatal_error": fatal_error,
        "git_branch": git_text(repo_root, "branch", "--show-current"),
        "git_commit": git_text(repo_root, "rev-parse", "HEAD"),
        "script_sha256": sha256_file(script_path),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_unchanged": protected_before == protected_after,
        "worker_files": [str(path.relative_to(output_dir)) for path in sorted(workers_dir.glob("*.json"))],
    }
    summary_path = output_dir / "p0_friction_material_sequence_bisect_summary.json"
    write_json(summary_path, summary)
    checksum_path = output_dir / "checksums.sha256"
    with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
        for path in [summary_path, *sorted(workers_dir.glob("*.json")), *sorted(workers_dir.glob("*.log"))]:
            stream.write(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n")
    print(f"P0_7F0_2_OUTPUT_DIR={output_dir}", flush=True)
    print(f"P0_7F0_2_REPRODUCED={len(reproduced)}/{len(exact_results)}", flush=True)
    print(f"P0_7F0_2_ROOT_CAUSE={root_cause}", flush=True)
    print(f"P0_7F0_2_STATUS={summary['overall_status']}", flush=True)
    return 1 if fatal_error else 0


if __name__ == "__main__":
    if PRE_ARGS.worker_mode == "none":
        raise SystemExit(orchestrate())
    worker_exit = run_worker()
    assert SIMULATION_APP is not None
    SIMULATION_APP.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(worker_exit)
