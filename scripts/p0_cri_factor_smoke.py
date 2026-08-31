"""运行四环境 CRI factor direct-read-back smoke test。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Reorient-Cube-Leap-CRI"  # 冻结的独立科研任务。
SMOKE_CODES = ("000", "100", "010", "001")  # 显式逐环境 assignment。
PROTECTED_PATHS = (
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/__init__.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/assets/leap.py",
    "logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth",
)  # smoke 前后必须保持字节不变的 baseline 资产。


def build_parser() -> argparse.ArgumentParser:
    """定义独立 smoke 参数，不引入训练器参数。"""

    parser = argparse.ArgumentParser(description="Run the LEAP Hand four-environment CRI read-back smoke test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("logs/p0_cri_factor_smoke"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS = build_parser().parse_args()  # 在启动 Kit 前解析参数。
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import gymnasium as gym  # noqa: E402  # Kit 启动后再导入仿真依赖。
import torch  # noqa: E402

import LEAP_Isaaclab.tasks  # noqa: E402,F401  # 触发 task 自动注册。
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes(repo_root: Path) -> dict[str, str | None]:
    """记录 baseline 文件和 checkpoint 当前字节状态。"""

    return {
        relative: sha256_file(repo_root / relative) if (repo_root / relative).is_file() else None
        for relative in PROTECTED_PATHS
    }


def tensor_to_json(value: Any) -> Any:
    """将 smoke 数据递归转换为 JSON 可序列化对象。"""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [tensor_to_json(item) for item in value]
    if isinstance(value, dict):
        return {key: tensor_to_json(item) for key, item in value.items()}
    return value


def add_check(checks: dict[str, dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    """记录一个具名验收项。"""

    checks[name] = {"status": "PASS" if passed else "FAIL", "detail": detail}


def tensors_close(left: torch.Tensor, right: torch.Tensor, atol: float = 1.0e-6, rtol: float = 1.0e-5) -> bool:
    """返回有限容差内的张量比较结果。"""

    return bool(torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False))


def compare_factor_snapshots(
    before: dict[str, Any], after: dict[str, Any], env_ids: list[int]
) -> tuple[bool, str]:
    """比较指定环境的全部 PhysX 实际因子字段。"""

    tensor_fields = (
        "actual_object_mass",
        "actual_object_inertia",
        "actual_object_static_friction",
        "actual_object_dynamic_friction",
        "actual_hand_static_friction",
        "actual_hand_dynamic_friction",
        "actual_effort_limit",
    )  # 只比较 direct-read-back 字段。
    failed = []
    for field in tensor_fields:
        if not tensors_close(before[field][env_ids], after[field][env_ids]):
            failed.append(field)
    for field in ("actual_hand_combine_mode", "actual_object_combine_mode"):
        if tuple(before[field][index] for index in env_ids) != tuple(after[field][index] for index in env_ids):
            failed.append(field)
    return not failed, "unchanged" if not failed else f"changed fields: {failed}"


def write_csv(path: Path, snapshot: dict[str, Any]) -> None:
    """写出每环境一行的 requested/actual 证据。"""

    fields = (
        "env_id",
        "factor_code",
        "factor_bits",
        "friction_condition",
        "mass_scale",
        "requested_mass",
        "actual_mass",
        "requested_inertia",
        "actual_inertia",
        "requested_object_static_friction",
        "actual_object_static_friction",
        "requested_object_dynamic_friction",
        "actual_object_dynamic_friction",
        "actual_hand_static_friction",
        "actual_hand_dynamic_friction",
        "hand_friction_combine_mode",
        "object_friction_combine_mode",
        "motor_capacity_scale",
        "requested_effort_limits",
        "actual_effort_limits",
        "episode_id",
        "reset_count",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for env_id, code in enumerate(snapshot["factor_codes"]):
            writer.writerow(
                {
                    "env_id": env_id,
                    "factor_code": code,
                    "factor_bits": json.dumps(tensor_to_json(snapshot["factor_code_bits"][env_id])),
                    "friction_condition": snapshot["friction_conditions"][env_id],
                    "mass_scale": float(snapshot["mass_scale"][env_id]),
                    "requested_mass": json.dumps(tensor_to_json(snapshot["requested_object_mass"][env_id])),
                    "actual_mass": json.dumps(tensor_to_json(snapshot["actual_object_mass"][env_id])),
                    "requested_inertia": json.dumps(tensor_to_json(snapshot["requested_object_inertia"][env_id])),
                    "actual_inertia": json.dumps(tensor_to_json(snapshot["actual_object_inertia"][env_id])),
                    "requested_object_static_friction": float(
                        snapshot["requested_object_static_friction"][env_id]
                    ),
                    "actual_object_static_friction": float(snapshot["actual_object_static_friction"][env_id]),
                    "requested_object_dynamic_friction": float(
                        snapshot["requested_object_dynamic_friction"][env_id]
                    ),
                    "actual_object_dynamic_friction": float(snapshot["actual_object_dynamic_friction"][env_id]),
                    "actual_hand_static_friction": float(snapshot["actual_hand_static_friction"][env_id]),
                    "actual_hand_dynamic_friction": float(snapshot["actual_hand_dynamic_friction"][env_id]),
                    "hand_friction_combine_mode": snapshot["actual_hand_combine_mode"][env_id],
                    "object_friction_combine_mode": snapshot["actual_object_combine_mode"][env_id],
                    "motor_capacity_scale": float(snapshot["motor_capacity_scale"][env_id]),
                    "requested_effort_limits": json.dumps(tensor_to_json(snapshot["requested_effort_limit"][env_id])),
                    "actual_effort_limits": json.dumps(tensor_to_json(snapshot["actual_effort_limit"][env_id])),
                    "episode_id": int(snapshot["episode_id"][env_id]),
                    "reset_count": int(snapshot["reset_count"][env_id]),
                }
            )


def run_smoke(repo_root: Path, checks: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行唯一一次四环境零动作 smoke。"""

    baseline_spec = gym.spec("Isaac-Reorient-Cube-Leap")
    cri_spec = gym.spec(TASK_ID)
    registration_ok = baseline_spec is not None and cri_spec is not None
    add_check(checks, "task_registration_integrity", registration_ok, "baseline and CRI task IDs resolve")

    env_cfg = parse_env_cfg(TASK_ID, device=ARGS.device, num_envs=4)
    env_cfg.seed = ARGS.seed
    env_cfg.factor_codes = SMOKE_CODES  # smoke 必须显式指定四个 factor code。
    add_check(
        checks,
        "explicit_factor_assignment",
        tuple(env_cfg.factor_codes) == SMOKE_CODES,
        f"configured codes={tuple(env_cfg.factor_codes)}",
    )
    add_check(
        checks,
        "events_and_adr_disabled",
        env_cfg.events is None and env_cfg.enable_adr is False,
        f"events={env_cfg.events}, enable_adr={env_cfg.enable_adr}",
    )

    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        core = env.unwrapped
        torch.manual_seed(ARGS.seed)
        observation, reset_extras = env.reset(seed=ARGS.seed)
        policy_obs = observation["policy"]
        expected_obs_shape = (4, env_cfg.observation_space)
        add_check(
            checks,
            "reset_observation_integrity",
            tuple(policy_obs.shape) == expected_obs_shape and bool(torch.isfinite(policy_obs).all()),
            f"shape={tuple(policy_obs.shape)}, finite={bool(torch.isfinite(policy_obs).all())}",
        )
        add_check(
            checks,
            "reset_episode_bookkeeping",
            bool(torch.equal(core.episode_length_buf, torch.zeros_like(core.episode_length_buf))),
            f"episode_length_buf={core.episode_length_buf.tolist()}",
        )

        hand_q = core.hand.data.joint_pos.torch
        hand_qdot = core.hand.data.joint_vel.torch
        expected_object_pos = core.object.data.default_root_pose.torch[:, 0:3] + core.scene.env_origins
        object_pos = core.object.data.root_pos_w.torch
        object_quat = core.object.data.root_quat_w.torch
        object_vel = core.object.data.root_vel_w.torch
        state_finite = all(
            bool(torch.isfinite(value).all())
            for value in (hand_q, hand_qdot, object_pos, object_quat, object_vel)
        )
        initial_state_ok = (
            state_finite
            and tensors_close(hand_q, core.override_default_joint_pos, atol=1.0e-5, rtol=0.0)
            and tensors_close(hand_qdot, torch.zeros_like(hand_qdot), atol=1.0e-5, rtol=0.0)
            and tensors_close(object_pos, expected_object_pos, atol=1.0e-5, rtol=0.0)
            and tensors_close(object_vel, torch.zeros_like(object_vel), atol=1.0e-5, rtol=0.0)
        )
        add_check(
            checks,
            "reset_hand_object_state_integrity",
            initial_state_ok,
            f"finite={state_finite}, q_match={tensors_close(hand_q, core.override_default_joint_pos)}",
        )

        all_env_ids = torch.arange(4, dtype=torch.long, device=core.device)
        core._read_back_factor_values(all_env_ids)  # 强制刷新 direct-read-back 证据。
        reset_snapshot = core.get_factor_snapshot()
        availability_ok = all(bool(value.all()) for value in reset_snapshot["actual_availability"].values())
        add_check(checks, "direct_readback_available", availability_ok, "all actual availability flags are true")

        requested_actual_ok = all(
            (
                tensors_close(reset_snapshot["requested_object_mass"], reset_snapshot["actual_object_mass"]),
                tensors_close(
                    reset_snapshot["requested_object_inertia"], reset_snapshot["actual_object_inertia"], atol=1.0e-7
                ),
                tensors_close(
                    reset_snapshot["requested_object_static_friction"],
                    reset_snapshot["actual_object_static_friction"],
                ),
                tensors_close(
                    reset_snapshot["requested_object_dynamic_friction"],
                    reset_snapshot["actual_object_dynamic_friction"],
                ),
                tensors_close(reset_snapshot["requested_effort_limit"], reset_snapshot["actual_effort_limit"]),
            )
        )
        add_check(checks, "requested_matches_physx_actual", requested_actual_ok, "mass/inertia/material/effort match")

        mass = reset_snapshot["actual_object_mass"]
        inertia = reset_snapshot["actual_object_inertia"]
        friction = reset_snapshot["actual_object_static_friction"]
        effort = reset_snapshot["actual_effort_limit"]
        one_factor_ok = (
            tensors_close(mass[1], mass[0] * 1.35)
            and tensors_close(inertia[1], inertia[0] * 1.35)
            and tensors_close(friction[1], friction[0])
            and tensors_close(effort[1], effort[0])
            and tensors_close(mass[2], mass[0])
            and tensors_close(inertia[2], inertia[0])
            and abs(float(friction[2]) - 0.45) <= 1.0e-6
            and tensors_close(effort[2], effort[0])
            and tensors_close(mass[3], mass[0])
            and tensors_close(inertia[3], inertia[0])
            and tensors_close(friction[3], friction[0])
            and bool(torch.allclose(effort[3], torch.full_like(effort[3], 0.40), atol=1.0e-6, rtol=0.0))
        )
        add_check(checks, "one_factor_only_conditions", one_factor_ok, "100/010/001 change only approved factors")

        combine_ok = all(mode == "min" for mode in reset_snapshot["actual_hand_combine_mode"]) and all(
            mode == "min" for mode in reset_snapshot["actual_object_combine_mode"]
        )
        add_check(checks, "friction_combine_mode_readback", combine_ok, "hand/object USD modes are min")

        common_step_before = int(core.common_step_counter)
        zero_actions = torch.zeros((4, env_cfg.action_space), dtype=torch.float32, device=core.device)
        next_observation, reward, terminated, truncated, step_extras = env.step(zero_actions)
        step_outputs_ok = (
            tuple(terminated.shape) == (4,)
            and tuple(truncated.shape) == (4,)
            and terminated.dtype == torch.bool
            and truncated.dtype == torch.bool
            and bool(torch.isfinite(reward).all())
            and bool(torch.isfinite(next_observation["policy"]).all())
            and not bool(terminated.any())
            and not bool(truncated.any())
        )
        bookkeeping_ok = bool(torch.equal(core.episode_length_buf, torch.ones_like(core.episode_length_buf))) and int(
            core.common_step_counter
        ) == common_step_before + 1
        add_check(
            checks,
            "termination_and_step_outputs",
            step_outputs_ok,
            f"terminated={terminated.tolist()}, truncated={truncated.tolist()}",
        )
        add_check(
            checks,
            "step_episode_bookkeeping",
            bookkeeping_ok,
            f"episode_length_buf={core.episode_length_buf.tolist()}, common_step={int(core.common_step_counter)}",
        )

        core._read_back_factor_values(all_env_ids)
        after_step_snapshot = core.get_factor_snapshot()
        static_ok, static_detail = compare_factor_snapshots(reset_snapshot, after_step_snapshot, [0, 1, 2, 3])
        add_check(checks, "episode_static_factor_readback", static_ok, static_detail)

        before_indexed_reapply = core.get_factor_snapshot()
        core._apply_factor_conditions(torch.tensor([1], dtype=torch.long, device=core.device))
        core._read_back_factor_values(all_env_ids)
        after_indexed_reapply = core.get_factor_snapshot()
        isolation_ok, isolation_detail = compare_factor_snapshots(
            before_indexed_reapply, after_indexed_reapply, [0, 2, 3]
        )
        add_check(checks, "indexed_write_cross_env_isolation", isolation_ok, isolation_detail)

        factor_buffer_ok = tuple(core.factor_code_labels) == SMOKE_CODES and bool(
            torch.equal(core.factor_code_bits, core._frozen_factor_code_bits)
        )
        add_check(checks, "factor_buffer_immutable_during_reset_and_step", factor_buffer_ok, "assignment unchanged")

        ppo_loaded = any(name == "rl_games" or name.startswith("rl_games.") for name in sys.modules)
        add_check(checks, "no_ppo_or_checkpoint_loaded", not ppo_loaded, f"rl_games_loaded={ppo_loaded}")

        lifecycle = {
            "reset_extras_keys": sorted(reset_extras),
            "step_extras_keys": sorted(step_extras),
            "observation_shape": list(policy_obs.shape),
            "observation_dtype": str(policy_obs.dtype),
            "terminated": tensor_to_json(terminated),
            "truncated": tensor_to_json(truncated),
            "episode_length_after_step": tensor_to_json(core.episode_length_buf),
            "common_step_counter_after_step": int(core.common_step_counter),
            "history_shape": list(core.obs_hist_buf.shape),
            "history_finite": bool(torch.isfinite(core.obs_hist_buf).all()),
        }  # 证明 events=None 未关闭正常 lifecycle。
        return after_indexed_reapply, lifecycle
    finally:
        env.close()


def main() -> int:
    """生成 smoke 证据并以非零状态暴露失败。"""

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = ARGS.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "p0_cri_factor_readback.csv"
    json_path = output_dir / "p0_cri_factor_smoke.json"
    checksum_path = output_dir / "p0_cri_factor_smoke.sha256"
    before_hashes = protected_hashes(repo_root)  # 运行前固定 baseline 字节状态。
    checks: dict[str, dict[str, Any]] = {}
    snapshot: dict[str, Any] | None = None
    lifecycle: dict[str, Any] | None = None
    execution_error: str | None = None

    try:
        snapshot, lifecycle = run_smoke(repo_root, checks)
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    after_hashes = protected_hashes(repo_root)
    protected_ok = before_hashes == after_hashes
    add_check(checks, "protected_baseline_bytes_unchanged", protected_ok, "pre/post SHA-256 maps match")
    if snapshot is not None:
        write_csv(csv_path, snapshot)

    all_checks_pass = bool(checks) and all(item["status"] == "PASS" for item in checks.values())
    report = {
        "overall_status": "PASS" if execution_error is None and all_checks_pass else "FAIL",
        "execution_error": execution_error,
        "task_id": TASK_ID,
        "protocol_id": "CRI_RESEARCH_TASK_CORE_IMPLEMENTATION",
        "factor_schema_version": "CRI_FACTOR_SCHEMA_V1",
        "seed": ARGS.seed,
        "num_envs": 4,
        "factor_codes": list(SMOKE_CODES),
        "events": None,
        "adr_enabled": False,
        "ppo_used": False,
        "checkpoint_used": False,
        "checks": checks,
        "lifecycle": lifecycle,
        "factor_snapshot": None if snapshot is None else tensor_to_json(snapshot),
        "protected_hashes_before": before_hashes,
        "protected_hashes_after": after_hashes,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=repo_root)
        .decode()
        .splitlines(),
    }
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    artifact_hashes = {json_path.name: sha256_file(json_path)}
    if csv_path.exists():
        artifact_hashes[csv_path.name] = sha256_file(csv_path)
    with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
        for name, digest in artifact_hashes.items():
            stream.write(f"{digest}  {name}\n")

    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "output_dir": str(output_dir),
                "json": str(json_path),
                "csv": str(csv_path) if csv_path.exists() else None,
                "checksums": str(checksum_path),
                "failed_checks": [name for name, item in checks.items() if item["status"] != "PASS"],
                "execution_error": execution_error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        SIMULATION_APP.close()
