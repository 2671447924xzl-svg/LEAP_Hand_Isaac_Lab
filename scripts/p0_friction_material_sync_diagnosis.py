"""P0-7F0.1 material setter visibility-boundary diagnosis; never qualification evidence."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Reorient-Cube-Leap-CRI"  # 复用冻结的CRI环境定义。
FACTOR_CODES = ("000", "100", "010", "001")  # 显式逐环境assignment。
TARGET_ENV_ID = 2  # 仅诊断friction factor环境。
REQUESTED_FRICTION = 0.45  # 冻结的low friction起始值。
HISTORY_FRICTION = 0.35  # 只用于复现已知历史状态，不构成candidate scan。
BASE_SEED = 42  # 所有reset使用同一固定种子。
ATOL = 1.0e-6  # 与CRI core material断言一致。
EVIDENCE_LABEL = "P0_7F0_1_DIAGNOSIS_ONLY_NOT_FOR_QUALIFICATION"  # 禁止转用为qualification。
VISIBILITY_CASES = ("immediate", "app_update", "physics_step", "env_step")  # 冻结可见性边界。
ROOT_CAUSE_CLASSES = (
    "PHYSX_SETTER_REQUIRES_SYNC_BOUNDARY",
    "RESET_EVENT_MATERIAL_ORDERING",
    "RESET_STATE_MATERIAL_CACHE_STALE",
    "NO_SYNC_EFFECT_FOUND",
    "ROOT_CAUSE_UNRESOLVED",
)  # 只允许科研审核批准的分类。

PROTECTED_PATHS = (
    "scripts/p0_physics_validation.py",
    "scripts/p0_timing_audit.py",
    "scripts/p0_friction_coupling_diagnosis.py",
    "scripts/p0_friction_material_diagnosis.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/factor_conditions.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/leap_hand_cri_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/reorientation_cri_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml",
    "logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth",
)  # 运行前后必须保持字节不变。


def build_parser() -> argparse.ArgumentParser:  # 仅暴露输出和Isaac启动参数。
    parser = argparse.ArgumentParser(description="Diagnose P0-7F0.1 material visibility boundaries.")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("logs/p0_friction_material_sync_diagnosis")
    )
    parser.add_argument("--static_only", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS = build_parser().parse_args()  # Kit启动前完成协议校验所需解析。


def validate_constants() -> None:  # 启动仿真前拒绝诊断规格漂移。
    assert FACTOR_CODES == ("000", "100", "010", "001")
    assert TARGET_ENV_ID == 2
    assert REQUESTED_FRICTION == 0.45
    assert HISTORY_FRICTION == 0.35
    assert VISIBILITY_CASES == ("immediate", "app_update", "physics_step", "env_step")
    assert set(ROOT_CAUSE_CLASSES) == {
        "PHYSX_SETTER_REQUIRES_SYNC_BOUNDARY",
        "RESET_EVENT_MATERIAL_ORDERING",
        "RESET_STATE_MATERIAL_CACHE_STALE",
        "NO_SYNC_EFFECT_FOUND",
        "ROOT_CAUSE_UNRESOLVED",
    }


validate_constants()
if ARGS.static_only:
    print("P0_7F0_1_STATIC_CONTRACT_PASS")
    raise SystemExit(0)


APP_LAUNCHER = AppLauncher(ARGS)  # 只为本次独占诊断启动Kit。
SIMULATION_APP = APP_LAUNCHER.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import LEAP_Isaaclab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def sha256_file(path: Path) -> str:  # 流式计算保护文件哈希。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes(repo_root: Path) -> dict[str, str | None]:  # 不存在项显式记为null。
    return {path: sha256_file(repo_root / path) if (repo_root / path).is_file() else None for path in PROTECTED_PATHS}


def git_text(repo_root: Path, *args: str) -> str:  # 只读记录运行代码身份。
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *args), capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else f"UNAVAILABLE:{completed.stderr.strip()}"


def tensor_payload(value: torch.Tensor) -> dict[str, Any]:  # 保存完整shape、dtype、device和值。
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "values": json_safe(value.detach().cpu().tolist()),
    }


def json_safe(value: Any) -> Any:  # 严格转换诊断对象，不允许NaN进入JSON。
    if isinstance(value, torch.Tensor):
        return tensor_payload(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def caller_name() -> str:  # 记录当前被包装函数的直接调用者。
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
        return caller.f_code.co_name if caller is not None else "UNAVAILABLE"
    finally:
        del frame


def close_enough(actual: float, expected: float) -> bool:  # 使用冻结绝对容差判断可见性。
    return abs(actual - expected) <= ATOL


class TraceRecorder:  # 只在当前实例包装调用，不修改CRI源码或控制语义。
    def __init__(self, core: Any, lifecycle_id: str):  # 保存独立环境生命周期和原始bound method。
        self.core = core
        self.lifecycle_id = lifecycle_id
        self.phase = "UNSET"
        self.reset_context: str | None = None
        self.events: list[dict[str, Any]] = []
        self.original_reset_idx = core._reset_idx
        self.original_apply = core._apply_factor_conditions
        self.original_read = core._read_materials
        self.original_write = core._write_object_material
        self.original_read_back = core._read_back_factor_values
        self.original_assert = core._assert_requested_matches_actual

    def clock(self) -> dict[str, Any]:  # 同时记录物理后端与环境计数器。
        return {
            "physics_step_count": int(self.core.sim.get_physics_step_count()),
            "simulation_time": float(self.core.sim.physics_manager.get_simulation_time()),
            "environment_sim_step_counter": int(self.core._sim_step_counter),
            "common_step_counter": int(self.core.common_step_counter),
        }

    def append(self, function: str, kind: str, payload: dict[str, Any]) -> None:  # 每条事件带阶段和时钟。
        event = {
            "event_index": len(self.events),
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_label": EVIDENCE_LABEL,
            "qualification_eligible": False,
            "lifecycle_id": self.lifecycle_id,
            "phase": self.phase,
            "reset_context": self.reset_context,
            "caller": payload.pop("caller", caller_name()),
            "function": function,
            "kind": kind,
            **self.clock(),
            **json_safe(payload),
        }
        self.events.append(event)
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)

    def install(self) -> None:  # 包装reset、setter、getter、buffer refresh与assert边界。
        recorder = self

        def traced_reset_idx(this: Any, env_ids: Any) -> Any:  # 记录完整reset入口、出口和异常。
            direct_caller = caller_name()
            recorder.append("ReorientationCRIEnv._reset_idx", "enter", {"caller": direct_caller, "env_ids": env_ids})
            try:
                result = recorder.original_reset_idx(env_ids)
            except Exception as exc:
                recorder.append(
                    "ReorientationCRIEnv._reset_idx",
                    "error",
                    {"caller": direct_caller, "env_ids": env_ids, "error_type": type(exc).__name__, "error": str(exc)},
                )
                raise
            recorder.append("ReorientationCRIEnv._reset_idx", "exit", {"caller": direct_caller, "env_ids": env_ids})
            return result

        def traced_apply(this: Any, env_ids: Any) -> Any:  # 标记factor setter所在reset阶段。
            direct_caller = caller_name()
            recorder.append("ReorientationCRIEnv._apply_factor_conditions", "enter", {"caller": direct_caller, "env_ids": env_ids})
            try:
                result = recorder.original_apply(env_ids)
            except Exception as exc:
                recorder.append(
                    "ReorientationCRIEnv._apply_factor_conditions",
                    "error",
                    {"caller": direct_caller, "env_ids": env_ids, "error_type": type(exc).__name__, "error": str(exc)},
                )
                raise
            recorder.append("ReorientationCRIEnv._apply_factor_conditions", "exit", {"caller": direct_caller, "env_ids": env_ids})
            return result

        def traced_read(this: Any, asset: Any, asset_name: str) -> torch.Tensor:  # 记录getter返回的完整材料张量。
            direct_caller = caller_name()
            result = recorder.original_read(asset, asset_name)
            recorder.append(
                "ReorientationCRIEnv._read_materials",
                "getter",
                {"caller": direct_caller, "asset_name": asset_name, "material_tensor": result},
            )
            return result

        def traced_write(
            this: Any, env_ids: torch.Tensor, static_friction: torch.Tensor, dynamic_friction: torch.Tensor
        ) -> None:  # 比较setter前与立即getter，不插入同步操作。
            direct_caller = caller_name()
            before = recorder.original_read(this.object, "object")
            recorder.original_write(env_ids, static_friction, dynamic_friction)
            after = recorder.original_read(this.object, "object")
            recorder.append(
                "ReorientationCRIEnv._write_object_material",
                "setter",
                {
                    "caller": direct_caller,
                    "env_ids": env_ids,
                    "requested_static_friction": static_friction,
                    "requested_dynamic_friction": dynamic_friction,
                    "material_tensor_before": before,
                    "material_tensor_after_setter": after,
                    "target_getter_before": float(before[TARGET_ENV_ID, 0, 0]),
                    "target_getter_after": float(after[TARGET_ENV_ID, 0, 0]),
                },
            )

        def traced_read_back(this: Any, env_ids: torch.Tensor) -> None:  # 区分raw getter和科研actual buffer。
            direct_caller = caller_name()
            before = this.actual_object_static_friction.clone()
            recorder.original_read_back(env_ids)
            recorder.append(
                "ReorientationCRIEnv._read_back_factor_values",
                "buffer_refresh",
                {
                    "caller": direct_caller,
                    "env_ids": env_ids,
                    "actual_static_buffer_before": before,
                    "actual_static_buffer_after": this.actual_object_static_friction,
                },
            )

        def traced_assert(this: Any, env_ids: torch.Tensor) -> None:  # 保存assert两侧值并原样传播异常。
            direct_caller = caller_name()
            payload = {
                "caller": direct_caller,
                "env_ids": env_ids,
                "requested_static_selected": this.requested_object_static_friction[env_ids],
                "actual_static_selected": this.actual_object_static_friction[env_ids],
            }
            try:
                recorder.original_assert(env_ids)
            except Exception as exc:
                recorder.append(
                    "ReorientationCRIEnv._assert_requested_matches_actual",
                    "assertion",
                    {**payload, "pass": False, "error_type": type(exc).__name__, "error": str(exc)},
                )
                raise
            recorder.append(
                "ReorientationCRIEnv._assert_requested_matches_actual",
                "assertion",
                {**payload, "pass": True, "error_type": None, "error": None},
            )

        self.core._reset_idx = types.MethodType(traced_reset_idx, self.core)
        self.core._apply_factor_conditions = types.MethodType(traced_apply, self.core)
        self.core._read_materials = types.MethodType(traced_read, self.core)
        self.core._write_object_material = types.MethodType(traced_write, self.core)
        self.core._read_back_factor_values = types.MethodType(traced_read_back, self.core)
        self.core._assert_requested_matches_actual = types.MethodType(traced_assert, self.core)


def create_env() -> gym.Env:  # 保持原dt、decimation、material API和reset流程。
    cfg = parse_env_cfg(TASK_ID, device=ARGS.device, num_envs=len(FACTOR_CODES))
    cfg.factor_codes = FACTOR_CODES
    cfg.events = None
    cfg.enable_adr = False
    env = gym.make(TASK_ID, cfg=cfg)
    core = env.unwrapped
    if tuple(core.factor_code_labels) != FACTOR_CODES:
        env.close()
        raise RuntimeError("Explicit factor assignment mismatch")
    if core.cfg.events is not None:
        env.close()
        raise RuntimeError("P0-7F0.1 requires events=None")
    return env


def material_state(core: Any, recorder: TraceRecorder, refresh_buffer: bool) -> dict[str, Any]:  # 捕获raw getter与buffer。
    materials = recorder.original_read(core.object, "object")
    target_ids = torch.tensor((TARGET_ENV_ID,), dtype=torch.long, device=core.device)
    if refresh_buffer:
        core._read_back_factor_values(target_ids)
    return {
        "env_ids": tensor_payload(target_ids),
        "material_tensor": tensor_payload(materials),
        "getter_static_all_envs": materials[:, 0, 0].detach().cpu().tolist(),
        "getter_dynamic_all_envs": materials[:, 0, 1].detach().cpu().tolist(),
        "getter_static_target": float(materials[TARGET_ENV_ID, 0, 0]),
        "getter_dynamic_target": float(materials[TARGET_ENV_ID, 0, 1]),
        "requested_static_all_envs": core.requested_object_static_friction.detach().cpu().tolist(),
        "requested_dynamic_all_envs": core.requested_object_dynamic_friction.detach().cpu().tolist(),
        "actual_static_buffer_all_envs": core.actual_object_static_friction.detach().cpu().tolist(),
        "actual_dynamic_buffer_all_envs": core.actual_object_dynamic_friction.detach().cpu().tolist(),
        "actual_static_buffer_target": float(core.actual_object_static_friction[TARGET_ENV_ID]),
        "actual_dynamic_buffer_target": float(core.actual_object_dynamic_friction[TARGET_ENV_ID]),
        **recorder.clock(),
    }


def attempt_reset(env: gym.Env, recorder: TraceRecorder, phase: str) -> dict[str, Any]:  # 捕获assert失败而不修改reset。
    recorder.phase = phase
    recorder.reset_context = phase
    before = material_state(env.unwrapped, recorder, refresh_buffer=False)
    torch.manual_seed(BASE_SEED)
    error: dict[str, Any] | None = None
    try:
        env.reset(seed=BASE_SEED)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    after = material_state(env.unwrapped, recorder, refresh_buffer=False)
    recorder.append(
        "gym.Env.reset",
        "reset_result",
        {"caller": "attempt_reset", "before": before, "after": after, "pass": error is None, "error": error},
    )
    recorder.reset_context = None
    return {"pass": error is None, "error": error, "before": before, "after": after}


def write_target(core: Any, recorder: TraceRecorder, friction: float, phase: str) -> dict[str, Any]:  # 使用未修改的CRI setter。
    recorder.phase = phase
    env_ids = torch.tensor((TARGET_ENV_ID,), dtype=torch.long, device=core.device)
    value = torch.tensor((friction,), dtype=torch.float32, device=core.device)
    before = material_state(core, recorder, refresh_buffer=False)
    core._write_object_material(env_ids, value, value)
    after_setter = material_state(core, recorder, refresh_buffer=False)
    return {"requested": friction, "env_ids": tensor_payload(env_ids), "before": before, "after_setter": after_setter}


def observe_target(core: Any, recorder: TraceRecorder, phase: str) -> dict[str, Any]:  # getter后刷新科研buffer并判断一致性。
    recorder.phase = phase
    state = material_state(core, recorder, refresh_buffer=True)
    getter_match = close_enough(state["getter_static_target"], REQUESTED_FRICTION)
    buffer_match = close_enough(state["actual_static_buffer_target"], REQUESTED_FRICTION)
    return {
        "requested": REQUESTED_FRICTION,
        "getter": state["getter_static_target"],
        "buffer": state["actual_static_buffer_target"],
        "getter_match": getter_match,
        "buffer_match": buffer_match,
        "match": getter_match and buffer_match,
        "state": state,
    }


def prepare_known_history(env: gym.Env, recorder: TraceRecorder, phase: str) -> dict[str, Any]:  # 固定0.35历史后触发原reset。
    core = env.unwrapped
    prior = write_target(core, recorder, HISTORY_FRICTION, f"{phase}_WRITE_HISTORY_0.35")
    prior_read = material_state(core, recorder, refresh_buffer=True)
    reset = attempt_reset(env, recorder, f"{phase}_RESET")
    return {"history_write": prior, "history_readback": prior_read, "reset": reset}


def run_visibility_case(
    env: gym.Env, recorder: TraceRecorder, case_name: str, boundary: Callable[[], Any] | None
) -> dict[str, Any]:  # 每个case复现同一0.35历史后测试一个边界。
    core = env.unwrapped
    preparation = prepare_known_history(env, recorder, f"CASE_{case_name.upper()}")
    write = write_target(core, recorder, REQUESTED_FRICTION, f"CASE_{case_name.upper()}_WRITE_0.45")
    before_boundary = recorder.clock()
    boundary_result: dict[str, Any] = {"called": boundary is not None, "error": None}
    if boundary is not None:
        recorder.phase = f"CASE_{case_name.upper()}_BOUNDARY"
        try:
            result = boundary()
            boundary_result["return_type"] = type(result).__name__
        except Exception as exc:
            boundary_result["error"] = {
                "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()
            }
    after_boundary = recorder.clock()
    observation = observe_target(core, recorder, f"CASE_{case_name.upper()}_READBACK")
    return {
        "case": case_name,
        "requested": REQUESTED_FRICTION,
        "preparation": preparation,
        "write": write,
        "boundary": boundary_result,
        "clock_before_boundary": before_boundary,
        "clock_after_boundary": after_boundary,
        "readback": observation["getter"],
        "buffer": observation["buffer"],
        "match": observation["match"] and boundary_result["error"] is None,
        "observation": observation,
    }


def run_isolated_history_scenario(label: str, with_history: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # 每个Run使用新环境。
    env = create_env()
    core = env.unwrapped
    recorder = TraceRecorder(core, lifecycle_id=f"history_{label}")
    recorder.install()
    try:
        initial_reset = attempt_reset(env, recorder, f"HISTORY_{label.upper()}_INITIAL_RESET")
        if not initial_reset["pass"]:
            return {"initial_reset": initial_reset, "pass": False, "error": "initial clean reset failed"}, recorder.events
        if with_history:
            history_write = write_target(core, recorder, HISTORY_FRICTION, f"HISTORY_{label.upper()}_WRITE_0.35")
            history_readback = material_state(core, recorder, refresh_buffer=True)
            tested_reset = attempt_reset(env, recorder, f"HISTORY_{label.upper()}_RESET_AFTER_0.35")
        else:
            history_write = None
            history_readback = None
            tested_reset = initial_reset
        write = write_target(core, recorder, REQUESTED_FRICTION, f"HISTORY_{label.upper()}_WRITE_0.45")
        observation = observe_target(core, recorder, f"HISTORY_{label.upper()}_READBACK")
        result = {
            "lifecycle_id": recorder.lifecycle_id,
            "process_id": os.getpid(),
            "with_history": with_history,
            "initial_reset": initial_reset,
            "history_write": history_write,
            "history_readback": history_readback,
            "tested_reset": tested_reset,
            "write": write,
            "observation": observation,
            "pass": tested_reset["pass"] and observation["match"],
            "post_reset_write_match": observation["match"],
        }
        return result, recorder.events
    finally:
        env.close()


def run_history_tests() -> tuple[dict[str, Any], list[dict[str, Any]]]:  # A/B/C分别使用独立环境生命周期。
    results: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for label, with_history in (("fresh_a", False), ("fresh_b", False), ("history_c", True)):
        result, scenario_events = run_isolated_history_scenario(label, with_history)
        results[label] = result
        events.extend(scenario_events)
    results["fresh_run_pass_history_run_fail"] = bool(
        results["fresh_a"]["pass"] and results["fresh_b"]["pass"] and not results["history_c"]["pass"]
    )
    results["independent_environment_lifecycles"] = True
    return results, events


def detect_reset_overwrite(events: list[dict[str, Any]]) -> dict[str, Any]:  # 只判定可见的setter调用顺序，不猜测后端写入。
    reset_groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        context = event.get("reset_context")
        if context is not None:
            reset_groups.setdefault(context, []).append(event)
    evidence: list[dict[str, Any]] = []
    overwrite_found = False
    for context, group in reset_groups.items():
        setters = [event for event in group if event["function"] == "ReorientationCRIEnv._write_object_material"]
        target_values: list[float | None] = []
        for event in setters:
            ids = event.get("env_ids", {}).get("values", [])
            values = event.get("requested_static_friction", {}).get("values", [])
            requested = None
            if TARGET_ENV_ID in ids:
                requested = float(values[ids.index(TARGET_ENV_ID)])
            target_values.append(requested)
        has_045 = any(value is not None and close_enough(value, REQUESTED_FRICTION) for value in target_values)
        later_035 = False
        if has_045:
            seen_045 = False
            for value in target_values:
                if value is not None and close_enough(value, REQUESTED_FRICTION):
                    seen_045 = True
                elif seen_045 and value is not None and close_enough(value, HISTORY_FRICTION):
                    later_035 = True
                    break
        overwrite_found = overwrite_found or later_035
        evidence.append({"reset_context": context, "target_setter_values": target_values, "write_0.45_then_0.35": later_035})
    return {"detected": overwrite_found, "reset_groups": evidence}


def classify_root_cause(
    visibility: list[dict[str, Any]], reset_overwrite: dict[str, Any], history: dict[str, Any]
) -> tuple[str, str]:  # 依据冻结决策树选择唯一分类。
    matches = {case["case"]: bool(case["match"]) for case in visibility}
    if not matches["immediate"]:
        for boundary in ("app_update", "physics_step", "env_step"):
            if matches[boundary]:
                return "PHYSX_SETTER_REQUIRES_SYNC_BOUNDARY", f"immediate failed; first matching boundary={boundary}"
    if reset_overwrite["detected"]:
        return "RESET_EVENT_MATERIAL_ORDERING", "trace contains an explicit target write 0.45 followed by 0.35 in one reset"
    contaminated_reset = history["history_c"]["tested_reset"]
    reset_getter = contaminated_reset["after"]["getter_static_target"]
    if (
        matches["immediate"]
        and not contaminated_reset["pass"]
        and close_enough(reset_getter, HISTORY_FRICTION)
        and not reset_overwrite["detected"]
    ):
        return (
            "RESET_STATE_MATERIAL_CACHE_STALE",
            "single-env setter is immediately visible, while reset's 0.45 setter returns historical 0.35 without a later 0.35 setter call",
        )
    if not any(matches.values()):
        return "NO_SYNC_EFFECT_FOUND", "no tested boundary exposed the requested 0.45 value"
    return "ROOT_CAUSE_UNRESOLVED", "observations do not satisfy another frozen classification rule"


def write_json(path: Path, payload: dict[str, Any]) -> None:  # create-only写入，避免覆盖既有证据。
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:  # 顺序执行边界、reset调用链和历史污染诊断。
    repo_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_output = ARGS.output_dir if ARGS.output_dir.is_absolute() else repo_root / ARGS.output_dir
    output_dir = (base_output / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = output_dir / "material_sync_trace.jsonl"
    summary_path = output_dir / "p0_friction_material_sync_summary.json"
    checksum_path = output_dir / "checksums.sha256"
    protected_before = protected_hashes(repo_root)
    env: gym.Env | None = None
    recorder: TraceRecorder | None = None
    all_events: list[dict[str, Any]] = []
    visibility: list[dict[str, Any]] = []
    history: dict[str, Any] = {}
    runtime_metadata: dict[str, Any] = {}
    fatal_error: dict[str, Any] | None = None
    root_cause = "ROOT_CAUSE_UNRESOLVED"
    root_cause_reason = "diagnosis did not reach classification"
    try:
        env = create_env()
        core = env.unwrapped
        runtime_metadata = {
            "physics_dt": float(core.cfg.sim.dt),
            "decimation": int(core.cfg.decimation),
            "events_cfg_is_none": bool(core.cfg.events is None),
        }
        recorder = TraceRecorder(core, lifecycle_id="visibility_boundary")
        recorder.install()
        initial_reset = attempt_reset(env, recorder, "INITIAL_RESET")
        if not initial_reset["pass"]:
            raise RuntimeError(f"Initial clean reset failed: {initial_reset['error']}")

        zero_action = torch.zeros((core.num_envs, core.cfg.action_space), dtype=torch.float32, device=core.device)
        boundaries: dict[str, Callable[[], Any] | None] = {
            "immediate": None,
            "app_update": SIMULATION_APP.update,
            "physics_step": lambda: core.sim.step(render=False),
            "env_step": lambda: env.step(zero_action),
        }
        for case_name in VISIBILITY_CASES:
            visibility.append(run_visibility_case(env, recorder, case_name, boundaries[case_name]))
        all_events.extend(recorder.events)
        env.close()
        env = None

        history, history_events = run_history_tests()
        all_events.extend(history_events)
        reset_overwrite = detect_reset_overwrite(all_events)
        root_cause, root_cause_reason = classify_root_cause(visibility, reset_overwrite, history)
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if recorder is not None and not all_events:
            all_events.extend(recorder.events)
        reset_overwrite = detect_reset_overwrite(all_events)
    finally:
        if env is not None:
            env.close()
        events = all_events if all_events else (recorder.events if recorder is not None else [])
        protected_after = protected_hashes(repo_root)
        with trace_path.open("x", encoding="utf-8", newline="\n") as stream:
            for global_index, event in enumerate(events):
                record = {"global_event_index": global_index, **event}
                stream.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
        case_matches = {case["case"]: bool(case["match"]) for case in visibility}
        synchronization_effect = "none"
        if case_matches.get("immediate") is False:
            for boundary_name in ("app_update", "physics_step", "env_step"):
                if case_matches.get(boundary_name):
                    synchronization_effect = boundary_name
                    break
        summary = {
            "evidence_label": EVIDENCE_LABEL,
            "qualification_eligible": False,
            "run_id": run_id,
            "process_id": os.getpid(),
            "task_id": TASK_ID,
            "factor_codes": list(FACTOR_CODES),
            "target_env_id": TARGET_ENV_ID,
            "requested_friction": REQUESTED_FRICTION,
            "history_reproducer_friction": HISTORY_FRICTION,
            "physics_dt": runtime_metadata.get("physics_dt"),
            "decimation": runtime_metadata.get("decimation"),
            "events_cfg_is_none": runtime_metadata.get("events_cfg_is_none"),
            "material_api": "RigidObject.root_view.set_material_properties/get_material_properties",
            "reset_call_chain_contract": [
                "gym.Env.reset",
                "DirectRLEnv.reset",
                "ReorientationCRIEnv._reset_idx",
                "ReorientationEnv._reset_idx",
                "DirectRLEnv._reset_idx",
                "scene.reset",
                "event manager skipped because events=None",
                "ReorientationCRIEnv._apply_factor_conditions",
                "ReorientationCRIEnv._write_object_material",
                "PhysX set_material_properties",
                "ReorientationCRIEnv._read_back_factor_values",
                "PhysX get_material_properties",
                "ReorientationCRIEnv._assert_requested_matches_actual",
                "scene.write_data_to_sim + sim.forward only if reset returns",
            ],
            "visibility_cases": visibility,
            "immediate_setter_visible": case_matches.get("immediate"),
            "synchronization_effect": synchronization_effect,
            "reset_overwrite": reset_overwrite,
            "history_tests": history,
            "root_cause_classification": root_cause,
            "root_cause_reason": root_cause_reason,
            "fatal_error": fatal_error,
            "git_branch": git_text(repo_root, "branch", "--show-current"),
            "git_commit": git_text(repo_root, "rev-parse", "HEAD"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "protected_hashes_before": protected_before,
            "protected_hashes_after": protected_after,
            "protected_unchanged": protected_before == protected_after,
        }
        write_json(summary_path, summary)
        with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
            for path in (trace_path, summary_path):
                stream.write(f"{sha256_file(path)}  {path.name}\n")
        print(f"P0_7F0_1_OUTPUT_DIR={output_dir}", flush=True)
        print(f"P0_7F0_1_ROOT_CAUSE={root_cause}", flush=True)
        print(f"P0_7F0_1_STATUS={'FAILED_TECHNICAL' if fatal_error else 'DIAGNOSIS_COLLECTED'}", flush=True)
    return 1 if fatal_error else 0


if __name__ == "__main__":
    exit_code = main()
    SIMULATION_APP.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
