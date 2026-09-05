"""P0-7F0 material write/readback diagnosis; no qualification or physics scan."""

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

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Reorient-Cube-Leap-CRI"  # 复用冻结CRI任务。
FACTOR_CODES = ("000", "100", "010", "001")  # 显式4-env assignment。
TARGET_ENV_ID = 2  # 仅诊断friction factor环境。
DIRECT_CASES = (0.45, 0.40, 0.35)  # 最小即时写入/回读复现。
CANDIDATE_SEQUENCE = (0.45, 0.40, 0.35, 0.30)  # 仅检查顺序污染，不做qualification。
BASE_SEED = 42
ATOL = 1.0e-6
EVIDENCE_LABEL = "P0_7F0_DIAGNOSIS_ONLY_NOT_FOR_QUALIFICATION"

PROTECTED_PATHS = (
    "scripts/p0_physics_validation.py",
    "scripts/p0_timing_audit.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/factor_conditions.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/leap_hand_cri_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/reorientation_cri_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml",
    "logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth",
)


def build_parser() -> argparse.ArgumentParser:  # 本脚本只提供单一诊断路径。
    parser = argparse.ArgumentParser(description="Diagnose P0-7F0 material write/readback semantics.")
    parser.add_argument("--output_dir", type=Path, default=Path("logs/p0_friction_material_diagnosis"))
    parser.add_argument("--static_only", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS = build_parser().parse_args()


def validate_constants() -> None:  # Kit启动前拒绝诊断规格漂移。
    assert FACTOR_CODES == ("000", "100", "010", "001")
    assert TARGET_ENV_ID == 2
    assert DIRECT_CASES == (0.45, 0.40, 0.35)
    assert CANDIDATE_SEQUENCE == (0.45, 0.40, 0.35, 0.30)


validate_constants()
if ARGS.static_only:
    print("P0_7F0_STATIC_CONTRACT_PASS")
    raise SystemExit(0)


APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import LEAP_Isaaclab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def sha256_file(path: Path) -> str:  # 流式计算证据哈希。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(repo_root: Path, *args: str) -> str:  # 只读Git身份。
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *args), capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else f"UNAVAILABLE:{completed.stderr.strip()}"


def protected_hashes(repo_root: Path) -> dict[str, str | None]:  # 逐项验证受保护文件未变化。
    return {path: sha256_file(repo_root / path) if (repo_root / path).is_file() else None for path in PROTECTED_PATHS}


def tensor_payload(value: torch.Tensor) -> dict[str, Any]:  # 保留形状、dtype、device与完整值。
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "values": value.detach().cpu().tolist(),
    }


def json_safe(value: Any) -> Any:  # 将诊断对象转换为严格JSON。
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


class TraceRecorder:  # 在实例层包装原函数，不修改CRI core源码。
    def __init__(self, core: Any):  # 保存原始bound methods供透明转发。
        self.core = core
        self.phase = "UNSET"
        self.events: list[dict[str, Any]] = []
        self.original_read = core._read_materials
        self.original_write = core._write_object_material
        self.original_read_back = core._read_back_factor_values
        self.original_assert = core._assert_requested_matches_actual

    def append(self, kind: str, payload: dict[str, Any]) -> None:  # 每条记录均声明非qualification用途。
        event = {
            "event_index": len(self.events),
            "evidence_label": EVIDENCE_LABEL,
            "phase": self.phase,
            "kind": kind,
            **json_safe(payload),
        }
        self.events.append(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)

    def install(self) -> None:  # 仅替换当前env实例的方法入口。
        recorder = self

        def traced_read(this: Any, asset: Any, asset_name: str) -> torch.Tensor:  # 记录getter完整张量。
            result = recorder.original_read(asset, asset_name)
            recorder.append("read_materials", {"asset_name": asset_name, "material_tensor": result})
            return result

        def traced_write(
            this: Any, env_ids: torch.Tensor, static_friction: torch.Tensor, dynamic_friction: torch.Tensor
        ) -> None:  # 记录setter前后及索引变换。
            before = recorder.original_read(this.object, "object")
            cpu_ids = env_ids.to(device="cpu", dtype=torch.int64)
            setter_indices = cpu_ids.to(dtype=torch.int32)
            selected_before = before.cpu()[cpu_ids]
            recorder.original_write(env_ids, static_friction, dynamic_friction)
            after = recorder.original_read(this.object, "object")
            recorder.append(
                "write_object_material",
                {
                    "env_ids": env_ids,
                    "cpu_index_ids": cpu_ids,
                    "physx_setter_indices": setter_indices,
                    "requested_static_friction": static_friction,
                    "requested_dynamic_friction": dynamic_friction,
                    "material_tensor_before": before,
                    "selected_material_before": selected_before,
                    "selected_shape_after_indexing": list(selected_before.shape),
                    "static_rhs_shape": list(static_friction.cpu()[:, None].shape),
                    "dynamic_rhs_shape": list(dynamic_friction.cpu()[:, None].shape),
                    "material_tensor_after_immediate_get": after,
                },
            )

        def traced_read_back(this: Any, env_ids: torch.Tensor) -> None:  # 记录科研actual buffer刷新。
            before_static = this.actual_object_static_friction.clone()
            before_dynamic = this.actual_object_dynamic_friction.clone()
            recorder.original_read_back(env_ids)
            recorder.append(
                "read_back_factor_values",
                {
                    "env_ids": env_ids,
                    "actual_static_before": before_static,
                    "actual_static_after": this.actual_object_static_friction,
                    "actual_dynamic_before": before_dynamic,
                    "actual_dynamic_after": this.actual_object_dynamic_friction,
                },
            )

        def traced_assert(this: Any, env_ids: torch.Tensor) -> None:  # 保留原异常并补充actual/expected。
            payload = {
                "env_ids": env_ids,
                "actual_static_selected": this.actual_object_static_friction[env_ids],
                "requested_static_selected": this.requested_object_static_friction[env_ids],
                "actual_dynamic_selected": this.actual_object_dynamic_friction[env_ids],
                "requested_dynamic_selected": this.requested_object_dynamic_friction[env_ids],
            }
            try:
                recorder.original_assert(env_ids)
            except Exception as exc:
                recorder.append("assert_requested_matches_actual", {**payload, "pass": False, "error": str(exc)})
                raise
            recorder.append("assert_requested_matches_actual", {**payload, "pass": True, "error": None})

        self.core._read_materials = types.MethodType(traced_read, self.core)
        self.core._write_object_material = types.MethodType(traced_write, self.core)
        self.core._read_back_factor_values = types.MethodType(traced_read_back, self.core)
        self.core._assert_requested_matches_actual = types.MethodType(traced_assert, self.core)


def create_env() -> gym.Env:  # 保持冻结任务、4-env assignment与事件设置。
    cfg = parse_env_cfg(TASK_ID, device=ARGS.device, num_envs=len(FACTOR_CODES))
    cfg.factor_codes = FACTOR_CODES
    cfg.events = None
    cfg.enable_adr = False
    env = gym.make(TASK_ID, cfg=cfg)
    if tuple(env.unwrapped.factor_code_labels) != FACTOR_CODES:
        env.close()
        raise RuntimeError("Explicit factor assignment mismatch")
    return env


def full_material_state(core: Any, recorder: TraceRecorder) -> dict[str, Any]:  # 同时保存getter与科研buffer状态。
    materials = recorder.original_read(core.object, "object")
    return {
        "material_tensor": tensor_payload(materials),
        "static_all_envs": materials[:, 0, 0].detach().cpu().tolist(),
        "dynamic_all_envs": materials[:, 0, 1].detach().cpu().tolist(),
        "requested_static_all_envs": core.requested_object_static_friction.detach().cpu().tolist(),
        "requested_dynamic_all_envs": core.requested_object_dynamic_friction.detach().cpu().tolist(),
        "actual_static_buffer_all_envs": core.actual_object_static_friction.detach().cpu().tolist(),
        "actual_dynamic_buffer_all_envs": core.actual_object_dynamic_friction.detach().cpu().tolist(),
    }


def attempt_reset(env: gym.Env, recorder: TraceRecorder, phase: str) -> dict[str, Any]:  # 保留原assert并捕获失败证据。
    recorder.phase = phase
    torch.manual_seed(BASE_SEED)
    try:
        env.reset(seed=BASE_SEED)
    except Exception as exc:
        return {"pass": False, "error_type": type(exc).__name__, "error": str(exc)}
    return {"pass": True, "error_type": None, "error": None}


def write_then_read(core: Any, recorder: TraceRecorder, friction: float, phase: str) -> dict[str, Any]:  # 单env即时写入/回读。
    recorder.phase = phase
    env_ids = torch.tensor((TARGET_ENV_ID,), dtype=torch.long, device=core.device)
    values = torch.tensor((friction,), dtype=torch.float32, device=core.device)
    before = full_material_state(core, recorder)
    core._write_object_material(env_ids, values, values)
    after_write = full_material_state(core, recorder)
    core._read_back_factor_values(env_ids)
    after_readback = full_material_state(core, recorder)
    actual_static = float(after_readback["static_all_envs"][TARGET_ENV_ID])
    actual_dynamic = float(after_readback["dynamic_all_envs"][TARGET_ENV_ID])
    return {
        "requested": friction,
        "actual_static": actual_static,
        "actual_dynamic": actual_dynamic,
        "pass": abs(actual_static - friction) <= ATOL and abs(actual_dynamic - friction) <= ATOL,
        "env_ids": tensor_payload(env_ids),
        "before": before,
        "after_write": after_write,
        "after_readback": after_readback,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:  # create-only证据文件。
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:  # 只运行材料写入/回读诊断，不执行controlled-slip。
    repo_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (repo_root / ARGS.output_dir / run_id).resolve() if not ARGS.output_dir.is_absolute() else (ARGS.output_dir / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = output_dir / "material_trace.jsonl"
    summary_path = output_dir / "p0_7f0_summary.json"
    checksum_path = output_dir / "checksums.sha256"
    protected_before = protected_hashes(repo_root)
    script_path = Path(__file__).resolve()
    env: gym.Env | None = None
    recorder: TraceRecorder | None = None
    direct_cases: list[dict[str, Any]] = []
    sequence: list[dict[str, Any]] = []
    fatal_error: dict[str, Any] | None = None
    try:
        env = create_env()
        core = env.unwrapped
        recorder = TraceRecorder(core)
        recorder.install()

        initial_reset = attempt_reset(env, recorder, "INITIAL_PARENT_RESET")
        recorder.append("initial_reset_result", {"result": initial_reset, "state": full_material_state(core, recorder)})

        for friction in DIRECT_CASES:
            result = write_then_read(core, recorder, friction, f"DIRECT_CASE_{friction:.2f}")
            direct_cases.append(result)

        for friction in CANDIDATE_SEQUENCE:
            recorder.phase = f"SEQUENCE_{friction:.2f}_BEFORE_RESET"
            before_reset = full_material_state(core, recorder)
            reset_result = attempt_reset(env, recorder, f"SEQUENCE_{friction:.2f}_RESET")
            after_reset = full_material_state(core, recorder)
            write_result = write_then_read(core, recorder, friction, f"SEQUENCE_{friction:.2f}_WRITE")
            after_readback = full_material_state(core, recorder)
            sequence.append(
                {
                    "candidate": friction,
                    "before_reset": before_reset,
                    "reset_result": reset_result,
                    "after_reset": after_reset,
                    "after_write": write_result["after_write"],
                    "after_readback": after_readback,
                    "write_result": {
                        "requested": write_result["requested"],
                        "actual_static": write_result["actual_static"],
                        "actual_dynamic": write_result["actual_dynamic"],
                        "pass": write_result["pass"],
                    },
                }
            )
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if env is not None:
            env.close()
        protected_after = protected_hashes(repo_root)
        events = recorder.events if recorder is not None else []
        with trace_path.open("x", encoding="utf-8", newline="\n") as stream:
            for event in events:
                stream.write(json.dumps(json_safe(event), ensure_ascii=False, allow_nan=False) + "\n")
        summary = {
            "evidence_label": EVIDENCE_LABEL,
            "qualification_eligible": False,
            "run_id": run_id,
            "process_id": os.getpid(),
            "task_id": TASK_ID,
            "factor_codes": list(FACTOR_CODES),
            "target_env_id": TARGET_ENV_ID,
            "direct_cases": direct_cases,
            "candidate_sequence": sequence,
            "fatal_error": fatal_error,
            "git_branch": git_text(repo_root, "branch", "--show-current"),
            "git_commit": git_text(repo_root, "rev-parse", "HEAD"),
            "script_sha256": sha256_file(script_path),
            "protected_hashes_before": protected_before,
            "protected_hashes_after": protected_after,
            "protected_unchanged": protected_before == protected_after,
        }
        write_json(summary_path, summary)
        with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
            for path in (trace_path, summary_path):
                stream.write(f"{sha256_file(path)}  {path.name}\n")
        print(f"P0_7F0_OUTPUT_DIR={output_dir}", flush=True)
        print(f"P0_7F0_STATUS={'FAILED_TECHNICAL' if fatal_error else 'DIAGNOSIS_COLLECTED'}", flush=True)
    return 1 if fatal_error else 0


if __name__ == "__main__":
    exit_code = main()
    SIMULATION_APP.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
