"""Unified P0 physics validation for the frozen LEAP Hand CRI factor schema.

This is an audit-only executable.  It does not import a trainer, load a checkpoint,
or alter the formal CRI/baseline task implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Reorient-Cube-Leap-CRI"
SCHEMA_ID = "CRI_FACTOR_SCHEMA_V1"
FACTOR_CODES = ("000", "100", "010", "001")
NUM_ENVS = 4
REPEATS = 3
BASE_SEED = 42

MASS_ENVS = (0, 1)
FRICTION_ENVS = (0, 2)
MOTOR_ENVS = (0, 3)

MASS_FORCE_X_N = 1.0
MASS_TORQUE_Z_NM = 0.001
MASS_STEPS = 24
ISOLATION_HEIGHT_M = 5.0

FRICTION_SETTLE_STEPS = 120
FRICTION_FORCE_MIN_N = 0.0
FRICTION_FORCE_MAX_N = 2.20
FRICTION_FORCE_STEP_N = 0.01
FRICTION_HOLD_STEPS = 8
FRICTION_VELOCITY_THRESHOLD_MPS = 0.01
FRICTION_DISPLACEMENT_THRESHOLD_M = 0.0005
FRICTION_ONSET_CONSECUTIVE_STEPS = 5
FRICTION_CLEARANCE_M = 0.001
PROBE_ROOT_PATH = "/World/ground"
PROBE_TEMP_MATERIAL_PATH = "/World/P0_5_audit_ground_material"
PROBE_STATIC_FRICTION = 0.80
PROBE_DYNAMIC_FRICTION = 0.80
PROBE_COMBINE_MODE = "min"

MOTOR_JOINT_NAME = "a_0"
MOTOR_NC_ACTION = 0.05
MOTOR_NC_CONTROL_STEPS = 12
MOTOR_STRESS_ACTION = 1.0
MOTOR_STRESS_DRIVE_CONTROL_STEPS = 4
MOTOR_STRESS_HOLD_CONTROL_STEPS = 16
MOTOR_JOINT_LIMIT_TOL_RAD = 1.0e-6
MOTOR_MIN_RELATIVE_AUC_GAP = 0.05
MOTOR_NC_ABSOLUTE_BOUND = 0.02
MOTOR_NC_STRESS_FRACTION = 0.5
MOTOR_SUPPORT_NUMERICAL_TOL = 1.0e-6

MASS_RATIO_RANGE = (0.68, 0.82)
FRICTION_MAX_ONSET_RATIO = 0.75

PROTECTED_PATHS = (
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/__init__.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/assets/leap.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/factor_conditions.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/leap_hand_cri_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/reorientation_cri_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml",
    "scripts/p0_timing_audit.py",
    "logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Unified P0 Physics Validation protocols.")
    parser.add_argument("--protocol", choices=("mass", "friction", "motor", "all"), default="all")
    parser.add_argument("--output_dir", type=Path, default=Path("logs/p0_physics_validation"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS = build_parser().parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

import LEAP_Isaaclab.tasks  # noqa: E402,F401
import isaaclab  # noqa: E402
from isaaclab.sim.utils import get_current_stage, get_first_matching_child_prim  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg  # noqa: E402
from isaacsim.core.version import get_version as get_isaac_sim_version  # noqa: E402


MASS_FIELDS = (
    "run_id", "repeat", "phase", "factor_code", "env_id", "physics_step", "sim_time_s",
    "requested_mass_kg", "actual_mass_kg", "requested_inertia_xx", "requested_inertia_yy",
    "requested_inertia_zz", "actual_inertia_xx", "actual_inertia_yy", "actual_inertia_zz",
    "force_x_n", "force_y_n", "force_z_n", "torque_x_nm", "torque_y_nm", "torque_z_nm",
    "vx_pre_mps", "vx_post_mps", "omega_z_pre_radps", "omega_z_post_radps",
    "delta_vx_mps", "delta_omega_z_radps", "estimated_ax_mps2", "estimated_alpha_z_radps2",
)

FRICTION_FIELDS = (
    "run_id", "repeat", "phase", "factor_code", "env_id", "level_index", "plateau_substep",
    "physics_step", "sim_time_s", "force_x_n", "actual_mass_kg", "normal_load_estimate_n",
    "requested_object_static_friction", "requested_object_dynamic_friction",
    "object_static_friction", "object_dynamic_friction", "object_combine_mode", "material_readback_scope",
    "probe_static_friction", "probe_dynamic_friction", "probe_combine_mode",
    "derived_pair_static_friction", "derived_pair_dynamic_friction", "derivation",
    "x_m", "vx_mps", "displacement_m", "velocity_condition", "displacement_condition",
    "within_plateau_consecutive", "onset_flag", "onset_force_n", "slip_onset_physics_step",
)

MOTOR_FIELDS = (
    "run_id", "repeat", "phase", "protocol_phase", "factor_code", "env_id", "joint_name",
    "joint_index", "action_index", "control_step", "physics_substep", "physics_step", "sim_time_s",
    "raw_commanded_action", "processed_action", "target_before", "commanded_target",
    "joint_pos_target_buffer", "physx_position_target", "requested_effort_limit_n", "actual_effort_limit_n",
    "applied_effort", "q_pre_rad", "q_post_rad", "qdot_pre_radps", "qdot_post_radps",
    "tracking_residual_rad", "reset_flag", "event_sequence",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_bytes(repo_root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True).stdout


def full_dirty_state(repo_root: Path) -> dict[str, Any]:
    """Hash tracked diff plus framed paths/content of all untracked files."""

    digest = hashlib.sha256()
    diff = git_bytes(repo_root, "diff", "--binary", "HEAD")
    digest.update(struct.pack(">Q", len(diff)))
    digest.update(diff)
    raw_paths = git_bytes(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    paths = sorted(path for path in raw_paths.split(b"\0") if path)
    manifest: dict[str, str] = {}
    for raw_path in paths:
        path_text = os.fsdecode(raw_path)
        content = (repo_root / path_text).read_bytes()
        digest.update(struct.pack(">Q", len(raw_path)))
        digest.update(raw_path)
        digest.update(struct.pack(">Q", len(content)))
        digest.update(content)
        manifest[path_text.replace("\\", "/")] = hashlib.sha256(content).hexdigest()
    return {"sha256": digest.hexdigest(), "untracked_manifest": manifest}


def protected_hashes(repo_root: Path) -> dict[str, str | None]:
    return {
        relative: sha256_file(repo_root / relative) if (repo_root / relative).is_file() else None
        for relative in PROTECTED_PATHS
    }


def tensor_to_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "torch"):
        return value.torch.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): tensor_to_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_to_json(item) for item in value]
    return value


def as_torch(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    return wp.to_torch(value)


def scalar(value: Any, *indices: int) -> float:
    tensor = as_torch(value)
    return float(tensor[indices].detach().cpu().item())


def ftext(value: float) -> str:
    return "NaN" if math.isnan(value) else f"{value:.12e}"


def close(left: Any, right: Any, atol: float = 1.0e-6, rtol: float = 1.0e-5) -> bool:
    return bool(torch.allclose(as_torch(left), as_torch(right), atol=atol, rtol=rtol, equal_nan=False))


def add_check(checks: dict[str, dict[str, Any]], name: str, passed: bool, evidence: str) -> None:
    checks[name] = {"status": "PASS" if passed else "FAIL", "evidence": evidence}


class CsvSink:
    def __init__(self, path: Path, fields: Iterable[str]):
        self.path = path
        self.stream = path.open("x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=tuple(fields))
        self.writer.writeheader()
        self.stream.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.stream.flush()

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def reset_environment(env: gym.Env, seed: int) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    return env.reset(seed=seed)


def direct_factor_snapshot(core: Any) -> dict[str, Any]:
    all_ids = torch.arange(NUM_ENVS, dtype=torch.long, device=core.device)
    core._read_back_factor_values(all_ids)
    return core.get_factor_snapshot()


def verify_pair_isolation(snapshot: dict[str, Any], pair: tuple[int, int], allowed: set[str]) -> tuple[bool, str]:
    tensor_fields = (
        "actual_object_mass", "actual_object_inertia", "actual_object_static_friction",
        "actual_object_dynamic_friction", "actual_hand_static_friction", "actual_hand_dynamic_friction",
        "actual_effort_limit",
    )
    changed = []
    for field in tensor_fields:
        if not close(snapshot[field][pair[0]], snapshot[field][pair[1]]):
            changed.append(field)
    for field in ("actual_hand_combine_mode", "actual_object_combine_mode"):
        if snapshot[field][pair[0]] != snapshot[field][pair[1]]:
            changed.append(field)
    unexpected = sorted(set(changed) - allowed)
    missing = sorted(allowed - set(changed))
    return not unexpected and not missing, f"changed={changed}; unexpected={unexpected}; missing={missing}"


def lifecycle_check(env: gym.Env, core: Any, checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    observation, reset_extras = reset_environment(env, BASE_SEED)
    policy = observation["policy"]
    hand_q = core.hand.data.joint_pos.torch
    hand_qdot = core.hand.data.joint_vel.torch
    expected_object = core.object.data.default_root_pose.torch[:, :3] + core.scene.env_origins
    state_ok = (
        tuple(policy.shape) == (NUM_ENVS, int(core.cfg.observation_space))
        and bool(torch.isfinite(policy).all())
        and close(hand_q, core.override_default_joint_pos, atol=1.0e-5, rtol=0.0)
        and close(hand_qdot, torch.zeros_like(hand_qdot), atol=1.0e-5, rtol=0.0)
        and close(core.object.data.root_pos_w.torch, expected_object, atol=1.0e-5, rtol=0.0)
        and close(core.object.data.root_vel_w.torch, torch.zeros_like(core.object.data.root_vel_w.torch), atol=1.0e-5, rtol=0.0)
    )
    add_check(checks, "baseline_lifecycle_reset_state", state_ok, f"observation_shape={tuple(policy.shape)}")
    book_reset = bool(torch.equal(core.episode_length_buf, torch.zeros_like(core.episode_length_buf)))
    add_check(checks, "baseline_lifecycle_reset_bookkeeping", book_reset, str(core.episode_length_buf.tolist()))
    before_counter = int(core.common_step_counter)
    zero = torch.zeros((NUM_ENVS, int(core.cfg.action_space)), device=core.device)
    next_obs, reward, terminated, truncated, step_extras = env.step(zero)
    outputs_ok = (
        bool(torch.isfinite(next_obs["policy"]).all()) and bool(torch.isfinite(reward).all())
        and not bool(terminated.any()) and not bool(truncated.any())
    )
    book_step = (
        bool(torch.equal(core.episode_length_buf, torch.ones_like(core.episode_length_buf)))
        and int(core.common_step_counter) == before_counter + 1
    )
    add_check(checks, "baseline_lifecycle_step_outputs", outputs_ok, f"terminated={terminated.tolist()}, truncated={truncated.tolist()}")
    add_check(checks, "baseline_lifecycle_step_bookkeeping", book_step, str(core.episode_length_buf.tolist()))
    return {
        "reset_extras_keys": sorted(reset_extras), "step_extras_keys": sorted(step_extras),
        "observation_shape": list(policy.shape), "terminated": tensor_to_json(terminated),
        "truncated": tensor_to_json(truncated), "episode_length_after_step": tensor_to_json(core.episode_length_buf),
    }


def set_object_state(core: Any, env_ids: tuple[int, ...], positions: torch.Tensor, identity: bool = True) -> None:
    ids = torch.tensor(env_ids, dtype=torch.long, device=core.device)
    pose = core.object.data.default_root_pose.torch[ids].clone()
    pose[:, :3] = positions
    if identity:
        # This project/Isaac stack stores quaternions as (x, y, z, w).
        pose[:, 3:7] = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=pose.dtype, device=core.device)
    velocity = torch.zeros((len(env_ids), 6), dtype=pose.dtype, device=core.device)
    core.object.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    core.object.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
    if identity:
        expected_quat = pose[:, 3:7]
        actual_quat = core.object.data.root_quat_w.torch[ids]
        if not torch.allclose(actual_quat, expected_quat, atol=1.0e-6, rtol=0.0):
            raise RuntimeError(f"Identity quaternion write/read mismatch: expected={expected_quat}, actual={actual_quat}")


def isolation_positions(core: Any, env_ids: tuple[int, ...], height: float) -> torch.Tensor:
    ids = torch.tensor(env_ids, dtype=torch.long, device=core.device)
    positions = core.scene.env_origins[ids].clone()
    positions[:, 2] += height
    return positions


def set_world_wrench(core: Any, env_ids: tuple[int, ...], force: tuple[float, float, float], torque: tuple[float, float, float]) -> None:
    ids = torch.tensor(env_ids, dtype=torch.long, device=core.device)
    forces = torch.tensor(force, dtype=torch.float32, device=core.device).reshape(1, 1, 3).repeat(len(env_ids), 1, 1)
    torques = torch.tensor(torque, dtype=torch.float32, device=core.device).reshape(1, 1, 3).repeat(len(env_ids), 1, 1)
    core.object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces, torques=torques, env_ids=ids, is_global=True
    )


def direct_physics_step(core: Any, dt: float) -> None:
    core.scene.write_data_to_sim()
    core.sim.step(render=False)
    core.scene.update(dt)


def run_mass_protocol(env: gym.Env, core: Any, sink: CsvSink, snapshot: dict[str, Any], dt: float) -> None:
    masses = snapshot["actual_object_mass"]
    inertias = snapshot["actual_object_inertia"]
    requested_masses = snapshot["requested_object_mass"]
    requested_inertias = snapshot["requested_object_inertia"]
    for repeat in range(REPEATS):
        for phase, force, torque in (
            ("linear", (MASS_FORCE_X_N, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ("angular", (0.0, 0.0, 0.0), (0.0, 0.0, MASS_TORQUE_Z_NM)),
        ):
            reset_environment(env, BASE_SEED)
            core.object.permanent_wrench_composer.reset()
            set_object_state(core, MASS_ENVS, isolation_positions(core, MASS_ENVS, ISOLATION_HEIGHT_M))
            set_world_wrench(core, MASS_ENVS, force, torque)
            try:
                start_step = int(core.sim.get_physics_step_count())
                for step in range(MASS_STEPS):
                    pre_vel = core.object.data.root_vel_w.torch.clone()
                    direct_physics_step(core, dt)
                    post_vel = core.object.data.root_vel_w.torch.clone()
                    for env_id in MASS_ENVS:
                        inertia = inertias[env_id, 0].reshape(-1)
                        requested_inertia = requested_inertias[env_id, 0].reshape(-1)
                        delta_vx = float(post_vel[env_id, 0] - pre_vel[env_id, 0])
                        delta_omega_z = float(post_vel[env_id, 5] - pre_vel[env_id, 5])
                        sink.write({
                            "run_id": f"mass_{phase}_r{repeat:02d}", "repeat": repeat, "phase": phase,
                            "factor_code": FACTOR_CODES[env_id], "env_id": env_id,
                            "physics_step": step, "sim_time_s": ftext((int(core.sim.get_physics_step_count()) - start_step) * dt),
                            "requested_mass_kg": ftext(float(requested_masses[env_id, 0])),
                            "actual_mass_kg": ftext(float(masses[env_id, 0])),
                            "requested_inertia_xx": ftext(float(requested_inertia[0])),
                            "requested_inertia_yy": ftext(float(requested_inertia[4])),
                            "requested_inertia_zz": ftext(float(requested_inertia[8])),
                            "actual_inertia_xx": ftext(float(inertia[0])), "actual_inertia_yy": ftext(float(inertia[4])),
                            "actual_inertia_zz": ftext(float(inertia[8])), "force_x_n": ftext(force[0]),
                            "force_y_n": ftext(force[1]), "force_z_n": ftext(force[2]),
                            "torque_x_nm": ftext(torque[0]), "torque_y_nm": ftext(torque[1]),
                            "torque_z_nm": ftext(torque[2]), "vx_pre_mps": ftext(float(pre_vel[env_id, 0])),
                            "vx_post_mps": ftext(float(post_vel[env_id, 0])),
                            "omega_z_pre_radps": ftext(float(pre_vel[env_id, 5])),
                            "omega_z_post_radps": ftext(float(post_vel[env_id, 5])),
                            "delta_vx_mps": ftext(delta_vx), "delta_omega_z_radps": ftext(delta_omega_z),
                            "estimated_ax_mps2": ftext(delta_vx / dt),
                            "estimated_alpha_z_radps2": ftext(delta_omega_z / dt),
                        })
            finally:
                core.object.permanent_wrench_composer.reset()


class GroundProbeBinding:
    def __init__(self):
        self.stage = get_current_stage()
        self.collider = None
        self.binding_api = None
        self.prior_path = None
        self.prior_strength = None
        self.metadata: dict[str, Any] = {"probe_path": PROBE_ROOT_PATH}

    def __enter__(self) -> "GroundProbeBinding":
        collider = get_first_matching_child_prim(
            PROBE_ROOT_PATH, predicate=lambda prim: prim.GetTypeName() == "Plane", stage=self.stage
        )
        if collider is None or not collider.IsValid():
            raise RuntimeError("P0-5 ground Plane collider missing")
        self.collider = collider
        self.binding_api = UsdShade.MaterialBindingAPI(collider)
        prior = self.binding_api.GetDirectBinding("physics")
        self.prior_path = prior.GetMaterialPath()
        prior_rel = prior.GetBindingRel()
        if not self.prior_path:
            raise RuntimeError("P0-5 ground Plane has no restorable direct physics material binding")
        self.prior_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(prior_rel)
        if self.stage.GetPrimAtPath(PROBE_TEMP_MATERIAL_PATH).IsValid():
            raise RuntimeError(f"Temporary probe material already exists: {PROBE_TEMP_MATERIAL_PATH}")
        self.metadata.update({
            "collider_path": str(collider.GetPath()), "temporary_material_path": PROBE_TEMP_MATERIAL_PATH,
            "prior_material_path": str(self.prior_path), "prior_binding_strength": str(self.prior_strength),
            "restored": False, "restored_strength": False, "temporary_material_removed": False,
        })
        try:
            cfg = PhysxRigidBodyMaterialCfg(
                static_friction=PROBE_STATIC_FRICTION, dynamic_friction=PROBE_DYNAMIC_FRICTION,
                restitution=0.0, friction_combine_mode=PROBE_COMBINE_MODE,
            )
            temp_prim = cfg.func(PROBE_TEMP_MATERIAL_PATH, cfg)
            self.binding_api.Bind(
                UsdShade.Material(temp_prim), bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            static = float(UsdPhysics.MaterialAPI(temp_prim).GetStaticFrictionAttr().Get())
            dynamic = float(UsdPhysics.MaterialAPI(temp_prim).GetDynamicFrictionAttr().Get())
            mode = str(PhysxSchema.PhysxMaterialAPI(temp_prim).GetFrictionCombineModeAttr().Get())
            bound = self.binding_api.GetDirectBinding("physics").GetMaterialPath()
            if bound != temp_prim.GetPath():
                raise RuntimeError(f"Temporary probe binding failed: bound={bound}, expected={temp_prim.GetPath()}")
            if not (
                math.isclose(static, 0.80, abs_tol=1.0e-6)
                and math.isclose(dynamic, 0.80, abs_tol=1.0e-6)
                and mode == "min"
            ):
                raise RuntimeError(f"Temporary probe direct read-back mismatch: {static}/{dynamic}/{mode}")
            self.metadata.update({
                "actual_static_friction": static, "actual_dynamic_friction": dynamic,
                "actual_combine_mode": mode,
            })
            return self
        except Exception:
            self._restore()
            raise

    def _restore(self) -> None:
        if self.binding_api is None or self.prior_path is None or self.prior_strength is None:
            return
        self.binding_api.UnbindDirectBinding("physics")
        prior_prim = self.stage.GetPrimAtPath(self.prior_path)
        if not prior_prim.IsValid():
            raise RuntimeError(f"Prior P0-5 material disappeared before restoration: {self.prior_path}")
        self.binding_api.Bind(
            UsdShade.Material(prior_prim), bindingStrength=self.prior_strength, materialPurpose="physics"
        )
        restored_binding = self.binding_api.GetDirectBinding("physics")
        restored_path = restored_binding.GetMaterialPath()
        restored_rel = restored_binding.GetBindingRel()
        restored_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(restored_rel)
        if self.stage.GetPrimAtPath(PROBE_TEMP_MATERIAL_PATH).IsValid():
            self.stage.RemovePrim(PROBE_TEMP_MATERIAL_PATH)
        temp_removed = not self.stage.GetPrimAtPath(PROBE_TEMP_MATERIAL_PATH).IsValid()
        self.metadata.update({
            "restored_material_path": str(restored_path),
            "restored_binding_strength": str(restored_strength),
            "restored": restored_path == self.prior_path,
            "restored_strength": restored_strength == self.prior_strength,
            "temporary_material_removed": temp_removed,
        })
        if not (self.metadata["restored"] and self.metadata["restored_strength"] and temp_removed):
            raise RuntimeError(
                "P0-5 exact restoration failed: "
                f"path={restored_path}, strength={restored_strength}, temp_removed={temp_removed}"
            )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._restore()


def cube_local_bounds(core: Any) -> dict[str, Any]:
    prim_path = f"{core.scene.env_prim_paths[0]}/object"
    prim = get_current_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Cube prim missing for local bounds: {prim_path}")
    local_range = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]
    ).ComputeLocalBound(prim).ComputeAlignedRange()
    minimum = tuple(float(v) for v in local_range.GetMin())
    maximum = tuple(float(v) for v in local_range.GetMax())
    size = tuple(float(v) for v in local_range.GetSize())
    if not all(math.isfinite(v) for v in minimum + maximum + size) or min(size) <= 0.0:
        raise RuntimeError(f"Invalid cube local bounds: min={minimum}, max={maximum}, size={size}")
    return {"prim_path": prim_path, "minimum": minimum, "maximum": maximum, "size": size, "half_height_m": 0.5 * size[2]}


def run_friction_protocol(
    env: gym.Env, core: Any, sink: CsvSink, snapshot: dict[str, Any], dt: float, probe_metadata: dict[str, Any]
) -> dict[str, Any]:
    bounds = cube_local_bounds(core)
    probe_metadata["cube_local_bounds"] = bounds
    object_static = snapshot["actual_object_static_friction"]
    object_dynamic = snapshot["actual_object_dynamic_friction"]
    requested_object_static = snapshot["requested_object_static_friction"]
    requested_object_dynamic = snapshot["requested_object_dynamic_friction"]
    object_modes = snapshot["actual_object_combine_mode"]
    masses = snapshot["actual_object_mass"]
    probe_static = float(probe_metadata["actual_static_friction"])
    probe_dynamic = float(probe_metadata["actual_dynamic_friction"])
    probe_mode = str(probe_metadata["actual_combine_mode"])
    for env_id in FRICTION_ENVS:
        if object_modes[env_id] != "min" or probe_mode != "min":
            raise RuntimeError("derived_pair_friction requires direct read-back combine=min on both materials")

    force_levels = np.round(
        np.arange(FRICTION_FORCE_MIN_N, FRICTION_FORCE_MAX_N + 0.5 * FRICTION_FORCE_STEP_N, FRICTION_FORCE_STEP_N), 10
    ).tolist()
    for repeat in range(REPEATS):
        reset_environment(env, BASE_SEED)
        core.object.permanent_wrench_composer.reset()
        ids = torch.tensor(FRICTION_ENVS, dtype=torch.long, device=core.device)
        positions = core.scene.env_origins[ids].clone()
        positions[:, 2] = bounds["half_height_m"] + FRICTION_CLEARANCE_M
        set_object_state(core, FRICTION_ENVS, positions)
        start_step = int(core.sim.get_physics_step_count())
        zero = (0.0, 0.0, 0.0)
        set_world_wrench(core, FRICTION_ENVS, zero, zero)
        for settle_step in range(FRICTION_SETTLE_STEPS):
            direct_physics_step(core, dt)
            pos = core.object.data.root_pos_w.torch
            vel = core.object.data.root_lin_vel_w.torch
            for env_id in FRICTION_ENVS:
                derived_s = min(float(object_static[env_id]), probe_static)
                derived_d = min(float(object_dynamic[env_id]), probe_dynamic)
                sink.write({
                    "run_id": f"friction_r{repeat:02d}", "repeat": repeat, "phase": "settle",
                    "factor_code": FACTOR_CODES[env_id], "env_id": env_id, "level_index": -1,
                    "plateau_substep": settle_step, "physics_step": int(core.sim.get_physics_step_count()) - start_step,
                    "sim_time_s": ftext((int(core.sim.get_physics_step_count()) - start_step) * dt), "force_x_n": ftext(0.0),
                    "actual_mass_kg": ftext(float(masses[env_id, 0])),
                    "normal_load_estimate_n": ftext(float(masses[env_id, 0]) * 9.81),
                    "requested_object_static_friction": ftext(float(requested_object_static[env_id])),
                    "requested_object_dynamic_friction": ftext(float(requested_object_dynamic[env_id])),
                    "object_static_friction": ftext(float(object_static[env_id])),
                    "object_dynamic_friction": ftext(float(object_dynamic[env_id])),
                    "object_combine_mode": object_modes[env_id],
                    "material_readback_scope": "protocol_start_direct_readback",
                    "probe_static_friction": ftext(probe_static),
                    "probe_dynamic_friction": ftext(probe_dynamic), "probe_combine_mode": probe_mode,
                    "derived_pair_static_friction": ftext(derived_s), "derived_pair_dynamic_friction": ftext(derived_d),
                    "derivation": "direct material read-back + combine=min", "x_m": ftext(float(pos[env_id, 0])),
                    "vx_mps": ftext(float(vel[env_id, 0])), "displacement_m": ftext(0.0),
                    "velocity_condition": 0, "displacement_condition": 0, "within_plateau_consecutive": 0,
                    "onset_flag": 0, "onset_force_n": "NaN", "slip_onset_physics_step": "NaN",
                })
        x0 = core.object.data.root_pos_w.torch[:, 0].clone()
        onset_found = {env_id: False for env_id in FRICTION_ENVS}
        onset_force = {env_id: math.nan for env_id in FRICTION_ENVS}
        try:
            for level_index, force_x in enumerate(force_levels):
                consecutive = {env_id: 0 for env_id in FRICTION_ENVS}
                set_world_wrench(core, FRICTION_ENVS, (float(force_x), 0.0, 0.0), zero)
                for plateau_substep in range(FRICTION_HOLD_STEPS):
                    direct_physics_step(core, dt)
                    pos = core.object.data.root_pos_w.torch
                    vel = core.object.data.root_lin_vel_w.torch
                    for env_id in FRICTION_ENVS:
                        displacement = abs(float(pos[env_id, 0] - x0[env_id]))
                        velocity_condition = abs(float(vel[env_id, 0])) >= FRICTION_VELOCITY_THRESHOLD_MPS
                        displacement_condition = displacement >= FRICTION_DISPLACEMENT_THRESHOLD_M
                        if not onset_found[env_id] and velocity_condition and displacement_condition:
                            consecutive[env_id] += 1
                        elif not onset_found[env_id]:
                            consecutive[env_id] = 0
                        onset_flag = False
                        if not onset_found[env_id] and consecutive[env_id] >= FRICTION_ONSET_CONSECUTIVE_STEPS:
                            onset_found[env_id] = True
                            onset_force[env_id] = float(force_x)
                            onset_flag = True
                        derived_s = min(float(object_static[env_id]), probe_static)
                        derived_d = min(float(object_dynamic[env_id]), probe_dynamic)
                        sink.write({
                            "run_id": f"friction_r{repeat:02d}", "repeat": repeat, "phase": "ramp",
                            "factor_code": FACTOR_CODES[env_id], "env_id": env_id, "level_index": level_index,
                            "plateau_substep": plateau_substep,
                            "physics_step": int(core.sim.get_physics_step_count()) - start_step,
                            "sim_time_s": ftext((int(core.sim.get_physics_step_count()) - start_step) * dt),
                            "force_x_n": ftext(float(force_x)), "actual_mass_kg": ftext(float(masses[env_id, 0])),
                            "normal_load_estimate_n": ftext(float(masses[env_id, 0]) * 9.81),
                            "requested_object_static_friction": ftext(float(requested_object_static[env_id])),
                            "requested_object_dynamic_friction": ftext(float(requested_object_dynamic[env_id])),
                            "object_static_friction": ftext(float(object_static[env_id])),
                            "object_dynamic_friction": ftext(float(object_dynamic[env_id])),
                            "object_combine_mode": object_modes[env_id],
                            "material_readback_scope": "protocol_start_direct_readback",
                            "probe_static_friction": ftext(probe_static),
                            "probe_dynamic_friction": ftext(probe_dynamic), "probe_combine_mode": probe_mode,
                            "derived_pair_static_friction": ftext(derived_s),
                            "derived_pair_dynamic_friction": ftext(derived_d),
                            "derivation": "direct material read-back + combine=min", "x_m": ftext(float(pos[env_id, 0])),
                            "vx_mps": ftext(float(vel[env_id, 0])), "displacement_m": ftext(displacement),
                            "velocity_condition": int(velocity_condition), "displacement_condition": int(displacement_condition),
                            "within_plateau_consecutive": consecutive[env_id], "onset_flag": int(onset_flag),
                            "onset_force_n": ftext(onset_force[env_id]),
                            "slip_onset_physics_step": (
                                int(core.sim.get_physics_step_count()) - start_step if onset_flag else "NaN"
                            ),
                        })
        finally:
            core.object.permanent_wrench_composer.reset()
    return bounds


def fixed_motor_reset(env: gym.Env, core: Any, seed: int) -> None:
    reset_environment(env, seed)
    core.object.permanent_wrench_composer.reset()
    set_object_state(core, tuple(range(NUM_ENVS)), isolation_positions(core, tuple(range(NUM_ENVS)), ISOLATION_HEIGHT_M))
    q = core.override_default_joint_pos.clone()
    qdot = torch.zeros_like(q)
    core.prev_targets[:] = q
    core.cur_targets[:] = q
    core.hand_dof_targets[:] = q
    core.hand.set_joint_position_target_index(target=q)
    core.hand.write_joint_position_to_sim_index(position=q)
    core.hand.write_joint_velocity_to_sim_index(velocity=qdot)
    core.scene.write_data_to_sim()


def motor_joint_limit_guard(core: Any, joint_index: int) -> dict[str, Any]:
    moving_average = float(core.cfg.act_moving_average)
    decimation = int(core.cfg.decimation)
    limits = core.hand.data.joint_pos_limits.torch
    initial = core.override_default_joint_pos[:, joint_index]
    recurrence_count = MOTOR_STRESS_DRIVE_CONTROL_STEPS * decimation
    result: dict[str, Any] = {
        "joint_name": MOTOR_JOINT_NAME, "joint_index": joint_index, "decimation": decimation,
        "act_moving_average": moving_average, "recurrence_count": recurrence_count,
        "nominal_excursion_rad": recurrence_count * moving_average * MOTOR_STRESS_ACTION,
        "tolerance_rad": MOTOR_JOINT_LIMIT_TOL_RAD, "per_env": {}, "status": "PASS",
    }
    for env_id in MOTOR_ENVS:
        target = float(initial[env_id])
        lower = float(limits[env_id, joint_index, 0])
        upper = float(limits[env_id, joint_index, 1])
        minimum_margin = math.inf
        targets = []
        for _ in range(recurrence_count):
            target += moving_average * MOTOR_STRESS_ACTION
            targets.append(target)
            minimum_margin = min(minimum_margin, target - lower, upper - target)
        confounded = minimum_margin <= MOTOR_JOINT_LIMIT_TOL_RAD
        result["per_env"][str(env_id)] = {
            "factor_code": FACTOR_CODES[env_id], "initial_q_rad": float(initial[env_id]),
            "lower_limit_rad": lower, "upper_limit_rad": upper, "planned_final_target_rad": target,
            "minimum_limit_margin_rad": minimum_margin, "confounded": confounded,
        }
        if confounded:
            result["status"] = "MOTOR_PROTOCOL_JOINT_LIMIT_CONFOUNDED"
    return result


class MotorTracer:
    def __init__(self, core: Any, sink: CsvSink, joint_index: int, action_index: int, dt: float):
        self.core = core
        self.sink = sink
        self.joint_index = joint_index
        self.action_index = action_index
        self.dt = dt
        self.decimation = int(core.cfg.decimation)
        self.originals: dict[str, Any] = {}
        self.enabled = False
        self.active = False
        self.pending: dict[str, Any] | None = None
        self.repeat = -1
        self.phase = ""
        self.protocol_phase = ""
        self.control_step = -1
        self.physics_substep = 0
        self.start_physics_step = 0
        self.raw_actions: dict[int, float] = {}
        self.processed_actions: dict[int, float] = {}
        self.reset_flag = False

    def install(self) -> None:
        self.originals = {
            "pre": self.core._pre_physics_step, "apply": self.core._apply_action,
            "write": self.core.scene.write_data_to_sim, "sim": self.core.sim.step,
            "update": self.core.scene.update,
        }

        def pre(actions: torch.Tensor) -> Any:
            if not self.enabled:
                return self.originals["pre"](actions)
            if self.active:
                raise RuntimeError("Previous motor control step is still active")
            self.active = True
            self.control_step += 1
            self.physics_substep = 0
            self.raw_actions = {env_id: scalar(actions, env_id, self.action_index) for env_id in MOTOR_ENVS}
            result = self.originals["pre"](actions)
            self.processed_actions = {env_id: scalar(self.core.actions, env_id, self.action_index) for env_id in MOTOR_ENVS}
            return result

        def apply() -> Any:
            if not self.enabled or not self.active:
                return self.originals["apply"]()
            if self.pending is not None:
                raise RuntimeError("Previous motor physics substep is incomplete")
            pending = {"events": ["apply_action"], "envs": {}}
            for env_id in MOTOR_ENVS:
                pending["envs"][env_id] = {
                    "target_before": scalar(self.core.cur_targets, env_id, self.joint_index),
                    "q_pre": scalar(self.core.hand.data.joint_pos, env_id, self.joint_index),
                    "qdot_pre": scalar(self.core.hand.data.joint_vel, env_id, self.joint_index),
                }
            result = self.originals["apply"]()
            for env_id in MOTOR_ENVS:
                pending["envs"][env_id].update({
                    "commanded_target": scalar(self.core.cur_targets, env_id, self.joint_index),
                    "target_buffer": scalar(self.core.hand.data.joint_pos_target, env_id, self.joint_index),
                })
            self.pending = pending
            return result

        def write() -> Any:
            result = self.originals["write"]()
            if self.enabled and self.active and self.pending is not None:
                physx_targets = wp.to_torch(self.core.hand.root_view.get_dof_position_targets())
                effort_limits = wp.to_torch(self.core.hand.root_view.get_dof_max_forces())
                for env_id in MOTOR_ENVS:
                    self.pending["envs"][env_id]["physx_target"] = scalar(physx_targets, env_id, self.joint_index)
                    self.pending["envs"][env_id]["effort_limit"] = scalar(effort_limits, env_id, self.joint_index)
                self.pending["events"].append("write_data_to_sim")
            return result

        def sim_step(*args: Any, **kwargs: Any) -> Any:
            if self.enabled and self.active and self.pending is not None:
                if any("physx_target" not in self.pending["envs"][env_id] for env_id in MOTOR_ENVS):
                    raise RuntimeError("Motor sim.step reached before PhysX target read-back")
                self.pending["events"].append("sim_step")
            return self.originals["sim"](*args, **kwargs)

        def update(*args: Any, **kwargs: Any) -> Any:
            result = self.originals["update"](*args, **kwargs)
            if self.enabled and self.active and self.pending is not None:
                self.pending["events"].append("scene_update")
                self.finalize_substep()
            return result

        self.core._pre_physics_step = pre
        self.core._apply_action = apply
        self.core.scene.write_data_to_sim = write
        self.core.sim.step = sim_step
        self.core.scene.update = update

    def restore(self) -> None:
        if not self.originals:
            return
        self.core._pre_physics_step = self.originals["pre"]
        self.core._apply_action = self.originals["apply"]
        self.core.scene.write_data_to_sim = self.originals["write"]
        self.core.sim.step = self.originals["sim"]
        self.core.scene.update = self.originals["update"]

    def start(self, repeat: int, phase: str) -> None:
        self.repeat = repeat
        self.phase = phase
        self.protocol_phase = phase
        self.control_step = -1
        self.physics_substep = 0
        self.start_physics_step = int(self.core.sim.get_physics_step_count())
        self.pending = None
        self.active = False
        self.reset_flag = True
        self.enabled = True

    def run_action(self, env: gym.Env, value: float, protocol_phase: str) -> None:
        self.protocol_phase = protocol_phase
        action = torch.zeros((NUM_ENVS, int(self.core.cfg.action_space)), device=self.core.device)
        action[:, self.action_index] = value
        env.step(action)
        if not self.active or self.physics_substep != self.decimation or self.pending is not None:
            raise RuntimeError(
                f"Incomplete motor control step: active={self.active}, substeps={self.physics_substep}, pending={self.pending is not None}"
            )
        self.active = False
        self.reset_flag = False

    def finalize_substep(self) -> None:
        assert self.pending is not None
        physics_step = int(self.core.sim.get_physics_step_count()) - self.start_physics_step
        sequence = ">".join(self.pending["events"])
        for env_id in MOTOR_ENVS:
            item = self.pending["envs"][env_id]
            q_post = scalar(self.core.hand.data.joint_pos, env_id, self.joint_index)
            qdot_post = scalar(self.core.hand.data.joint_vel, env_id, self.joint_index)
            self.sink.write({
                "run_id": f"motor_{self.phase}_r{self.repeat:02d}", "repeat": self.repeat,
                "phase": self.phase, "protocol_phase": self.protocol_phase, "factor_code": FACTOR_CODES[env_id],
                "env_id": env_id, "joint_name": MOTOR_JOINT_NAME, "joint_index": self.joint_index,
                "action_index": self.action_index, "control_step": self.control_step,
                "physics_substep": self.physics_substep, "physics_step": physics_step,
                "sim_time_s": ftext(physics_step * self.dt), "raw_commanded_action": ftext(self.raw_actions[env_id]),
                "processed_action": ftext(self.processed_actions[env_id]), "target_before": ftext(item["target_before"]),
                "commanded_target": ftext(item["commanded_target"]),
                "joint_pos_target_buffer": ftext(item["target_buffer"]),
                "physx_position_target": ftext(item["physx_target"]),
                "requested_effort_limit_n": ftext(
                    scalar(self.core.requested_effort_limit, env_id, self.joint_index)
                ),
                "actual_effort_limit_n": ftext(item["effort_limit"]), "applied_effort": "NaN",
                "q_pre_rad": ftext(item["q_pre"]), "q_post_rad": ftext(q_post),
                "qdot_pre_radps": ftext(item["qdot_pre"]), "qdot_post_radps": ftext(qdot_post),
                "tracking_residual_rad": ftext(item["physx_target"] - q_post),
                "reset_flag": int(self.reset_flag), "event_sequence": sequence,
            })
        self.physics_substep += 1
        self.pending = None


def run_motor_protocol(env: gym.Env, core: Any, sink: CsvSink, dt: float) -> dict[str, Any]:
    if bool(getattr(core, "_physics_handles_decimation", False)):
        raise RuntimeError("Motor physics substep tracing is unavailable when physics handles decimation")
    joint_index = core.hand.joint_names.index(MOTOR_JOINT_NAME)
    action_index = core.actuated_dof_indices.index(joint_index)
    fixed_motor_reset(env, core, BASE_SEED)
    guard = motor_joint_limit_guard(core, joint_index)
    guard["action_index"] = action_index
    stiffness = wp.to_torch(core.hand.root_view.get_dof_stiffnesses())
    damping = wp.to_torch(core.hand.root_view.get_dof_dampings())
    guard["control_invariants"] = {
        "action_type": str(core.cfg.action_type),
        "act_moving_average": float(core.cfg.act_moving_average),
        "physics_dt": dt,
        "decimation": int(core.cfg.decimation),
        "per_env": {
            str(env_id): {
                "factor_code": FACTOR_CODES[env_id],
                "actual_kp": scalar(stiffness, env_id, joint_index),
                "actual_kd": scalar(damping, env_id, joint_index),
            }
            for env_id in MOTOR_ENVS
        },
    }
    if guard["status"] != "PASS":
        return guard

    tracer = MotorTracer(core, sink, joint_index, action_index, dt)
    original_get_dones = core._get_dones

    def audit_no_dones() -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(NUM_ENVS, dtype=torch.bool, device=core.device)
        return zeros, zeros.clone()

    tracer.install()
    core._get_dones = audit_no_dones
    guard["audit_only_termination_suppression"] = True
    try:
        for repeat in range(REPEATS):
            tracer.enabled = False
            fixed_motor_reset(env, core, BASE_SEED)
            tracer.start(repeat, "nc")
            for _ in range(MOTOR_NC_CONTROL_STEPS):
                tracer.run_action(env, MOTOR_NC_ACTION, "nc_drive")

            tracer.enabled = False
            fixed_motor_reset(env, core, BASE_SEED)
            tracer.start(repeat, "stress")
            for _ in range(MOTOR_STRESS_DRIVE_CONTROL_STEPS):
                tracer.run_action(env, MOTOR_STRESS_ACTION, "stress_drive")
            for _ in range(MOTOR_STRESS_HOLD_CONTROL_STEPS):
                tracer.run_action(env, 0.0, "stress_hold")
    finally:
        tracer.enabled = False
        tracer.restore()
        core._get_dones = original_get_dones
    guard["termination_method_restored"] = core._get_dones == original_get_dones
    return guard


def evaluate_mass(rows: list[dict[str, str]], checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["repeat"]), row["phase"], int(row["env_id"]))].append(row)
    results = []
    for repeat in range(REPEATS):
        linear = {}
        angular = {}
        for env_id in MASS_ENVS:
            lrows = grouped[(repeat, "linear", env_id)]
            arows = grouped[(repeat, "angular", env_id)]
            if len(lrows) != MASS_STEPS or len(arows) != MASS_STEPS:
                raise RuntimeError(f"Mass raw row count mismatch repeat={repeat}, env={env_id}")
            linear[env_id] = float(lrows[-1]["vx_post_mps"]) - float(lrows[0]["vx_pre_mps"])
            angular[env_id] = float(arows[-1]["omega_z_post_radps"]) - float(arows[0]["omega_z_pre_radps"])
        results.append({
            "repeat": repeat, "delta_v": {str(k): v for k, v in linear.items()},
            "delta_omega": {str(k): v for k, v in angular.items()},
            "linear_response_ratio_100_over_000": abs(linear[1]) / max(abs(linear[0]), 1.0e-12),
            "angular_response_ratio_100_over_000": abs(angular[1]) / max(abs(angular[0]), 1.0e-12),
        })
    first = grouped[(0, "linear", 0)][0]
    high = grouped[(0, "linear", 1)][0]
    mass_ratio = float(high["actual_mass_kg"]) / float(first["actual_mass_kg"])
    inertia_ratios = [float(high[key]) / float(first[key]) for key in ("actual_inertia_xx", "actual_inertia_yy", "actual_inertia_zz")]
    mass_absolute_ok = (
        math.isclose(float(first["actual_mass_kg"]), 0.2160, rel_tol=0.0, abs_tol=1.0e-6)
        and math.isclose(float(high["actual_mass_kg"]), 0.2916, rel_tol=0.0, abs_tol=1.0e-6)
        and all(
            math.isclose(float(row["requested_mass_kg"]), float(row["actual_mass_kg"]), rel_tol=1.0e-5, abs_tol=1.0e-6)
            for row in rows
        )
    )
    inertia_requested_actual_ok = all(
        math.isclose(
            float(row[f"requested_inertia_{axis}"]), float(row[f"actual_inertia_{axis}"]),
            rel_tol=1.0e-5, abs_tol=1.0e-7,
        )
        for row in rows for axis in ("xx", "yy", "zz")
    )
    linear_ratios = [item["linear_response_ratio_100_over_000"] for item in results]
    angular_directions = [
        item["delta_omega"]["0"] > 0.0 and item["delta_omega"]["1"] > 0.0
        and item["angular_response_ratio_100_over_000"] < 1.0 for item in results
    ]
    add_check(checks, "MASS-P1-direct_mass_readback", mass_absolute_ok and math.isclose(mass_ratio, 1.35, rel_tol=1.0e-5, abs_tol=1.0e-6), f"actual=({first['actual_mass_kg']},{high['actual_mass_kg']}), ratio={mass_ratio}, requested_actual={mass_absolute_ok}")
    add_check(checks, "MASS-P2-inertia_scales_with_mass", inertia_requested_actual_ok and all(math.isclose(v, 1.35, rel_tol=1.0e-5, abs_tol=1.0e-6) for v in inertia_ratios), f"ratios={inertia_ratios}, requested_actual={inertia_requested_actual_ok}")
    add_check(checks, "MASS-P3-linear_response_ratio", all(MASS_RATIO_RANGE[0] <= v <= MASS_RATIO_RANGE[1] for v in linear_ratios), f"ratios={linear_ratios}")
    add_check(checks, "MASS-P4-angular_response_direction", all(angular_directions), f"results={results}")
    add_check(checks, "MASS-P5-repeat_direction_consistency", all(item["delta_v"]["0"] > item["delta_v"]["1"] > 0.0 for item in results), f"results={results}")
    return {"mass_ratio": mass_ratio, "inertia_ratios": inertia_ratios, "repeats": results}


def evaluate_friction(rows: list[dict[str, str]], checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    onset_rows = [row for row in rows if row["phase"] == "ramp" and int(row["onset_flag"]) == 1]
    by_key = {(int(row["repeat"]), int(row["env_id"])): float(row["onset_force_n"]) for row in onset_rows}
    complete = all((repeat, env_id) in by_key for repeat in range(REPEATS) for env_id in FRICTION_ENVS)
    results = []
    if complete:
        for repeat in range(REPEATS):
            nominal = by_key[(repeat, 0)]
            low = by_key[(repeat, 2)]
            mass = next(float(row["actual_mass_kg"]) for row in rows if int(row["repeat"]) == repeat and int(row["env_id"]) == 0)
            results.append({
                "repeat": repeat, "onset_000_n": nominal, "onset_010_n": low,
                "onset_ratio_010_over_000": low / max(nominal, 1.0e-12),
                "mu_hat_000": nominal / (mass * 9.81), "mu_hat_010": low / (mass * 9.81),
            })
    material_ok = bool(rows) and all(
        math.isclose(float(row["probe_static_friction"]), 0.80, abs_tol=1.0e-6)
        and math.isclose(float(row["probe_dynamic_friction"]), 0.80, abs_tol=1.0e-6)
        and row["probe_combine_mode"] == "min" and row["object_combine_mode"] == "min"
        and math.isclose(
            float(row["requested_object_static_friction"]), float(row["object_static_friction"]),
            abs_tol=1.0e-6,
        )
        and math.isclose(
            float(row["requested_object_dynamic_friction"]), float(row["object_dynamic_friction"]),
            abs_tol=1.0e-6,
        )
        and row["material_readback_scope"] == "protocol_start_direct_readback"
        and row["derivation"] == "direct material read-back + combine=min" for row in rows
    )
    object_absolute_ok = bool(rows) and all(
        math.isclose(
            float(row["object_static_friction"]), 0.80 if int(row["env_id"]) == 0 else 0.45,
            abs_tol=1.0e-6,
        )
        for row in rows
    )
    add_check(checks, "FRIC-P1-direct_material_and_combine_readback", material_ok and object_absolute_ok, f"requested/actual object values, probe=0.80, combine=min, scope=protocol_start; absolute_ok={object_absolute_ok}")
    add_check(checks, "FRIC-P2-low_onset_exists_and_is_lower", complete and all(item["onset_010_n"] < item["onset_000_n"] for item in results), f"results={results}")
    add_check(checks, "FRIC-P3-onset_ratio", complete and all(item["onset_ratio_010_over_000"] <= FRICTION_MAX_ONSET_RATIO for item in results), f"results={results}")
    add_check(checks, "FRIC-P4-repeat_direction_consistency", complete and len(results) == REPEATS and all(item["onset_010_n"] < item["onset_000_n"] for item in results), f"results={results}")
    add_check(checks, "FRIC-P5_thresholds_not_identical", complete and all(not math.isclose(item["onset_010_n"], item["onset_000_n"], abs_tol=1.0e-12) for item in results), f"results={results}")
    return {"onset_complete": complete, "repeats": results, "onset_count": len(onset_rows)}


def motor_phase_metrics(rows: list[dict[str, str]], phase: str) -> dict[int, dict[int, dict[str, float]]]:
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["phase"] == phase:
            grouped[(int(row["repeat"]), int(row["env_id"]))].append(row)
    metrics: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    for (repeat, env_id), group in grouped.items():
        group.sort(key=lambda row: (int(row["control_step"]), int(row["physics_substep"])))
        initial_q = float(group[0]["q_pre_rad"])
        metrics[repeat][env_id] = {
            "auc": 0.0,
            "auc_dt_units": sum(abs(float(row["tracking_residual_rad"])) for row in group),
            "target_integral_dt_units": sum(abs(float(row["physx_position_target"]) - initial_q) for row in group),
            "final_excursion": abs(float(group[-1]["q_post_rad"]) - initial_q),
            "max_abs_qdot": max(abs(float(row["qdot_post_radps"])) for row in group),
            "row_count": float(len(group)),
        }
    return metrics


def evaluate_motor(
    rows: list[dict[str, str]], checks: dict[str, dict[str, Any]], dt: float, guard: dict[str, Any]
) -> dict[str, Any]:
    nc = motor_phase_metrics(rows, "nc")
    stress = motor_phase_metrics(rows, "stress")
    results = []
    complete = all(repeat in nc and repeat in stress and all(env_id in nc[repeat] and env_id in stress[repeat] for env_id in MOTOR_ENVS) for repeat in range(REPEATS))
    if complete:
        for repeat in range(REPEATS):
            for phase_metrics in (nc[repeat], stress[repeat]):
                for item in phase_metrics.values():
                    item["auc"] = item["auc_dt_units"] * dt
                    item["target_integral"] = item["target_integral_dt_units"] * dt
            stress_nominal = stress[repeat][0]
            stress_low = stress[repeat][3]
            nc_denominator = max(nc[repeat][0]["target_integral"], 1.0e-6)
            stress_denominator = max(stress_nominal["target_integral"], 1.0e-6)
            d_nc = sum(
                abs(float(a["q_post_rad"]) - float(b["q_post_rad"])) * dt
                for a, b in zip(
                    [row for row in rows if row["phase"] == "nc" and int(row["repeat"]) == repeat and int(row["env_id"]) == 0],
                    [row for row in rows if row["phase"] == "nc" and int(row["repeat"]) == repeat and int(row["env_id"]) == 3],
                )
            ) / nc_denominator
            d_stress = sum(
                abs(float(a["q_post_rad"]) - float(b["q_post_rad"])) * dt
                for a, b in zip(
                    [row for row in rows if row["phase"] == "stress" and int(row["repeat"]) == repeat and int(row["env_id"]) == 0],
                    [row for row in rows if row["phase"] == "stress" and int(row["repeat"]) == repeat and int(row["env_id"]) == 3],
                )
            ) / stress_denominator
            results.append({
                "repeat": repeat, "auc_000": stress_nominal["auc"], "auc_001": stress_low["auc"],
                "auc_ratio_001_over_000": stress_low["auc"] / max(stress_nominal["auc"], 1.0e-12),
                "final_excursion_000": stress_nominal["final_excursion"],
                "final_excursion_001": stress_low["final_excursion"],
                "max_abs_qdot_000": stress_nominal["max_abs_qdot"],
                "max_abs_qdot_001": stress_low["max_abs_qdot"], "D_pair_NC": d_nc, "D_pair_stress": d_stress,
                "NC_bound": max(MOTOR_NC_ABSOLUTE_BOUND, MOTOR_NC_STRESS_FRACTION * d_stress),
            })
    effort_values = defaultdict(set)
    requested_effort_values = defaultdict(set)
    event_ok = True
    row_count_ok = complete
    target_chain_ok = True
    for row in rows:
        effort_values[int(row["env_id"])].add(round(float(row["actual_effort_limit_n"]), 6))
        requested_effort_values[int(row["env_id"])].add(round(float(row["requested_effort_limit_n"]), 6))
        event_ok = event_ok and row["event_sequence"] == "apply_action>write_data_to_sim>sim_step>scene_update"
        target_chain_ok = target_chain_ok and abs(float(row["commanded_target"]) - float(row["joint_pos_target_buffer"])) <= 1.0e-6 and abs(float(row["joint_pos_target_buffer"]) - float(row["physx_position_target"])) <= 1.0e-6
    expected_counts = {
        "nc": REPEATS * len(MOTOR_ENVS) * MOTOR_NC_CONTROL_STEPS * int(round(next(iter(nc[0].values()))["row_count"] / MOTOR_NC_CONTROL_STEPS)) if complete else -1,
        "stress": REPEATS * len(MOTOR_ENVS) * (MOTOR_STRESS_DRIVE_CONTROL_STEPS + MOTOR_STRESS_HOLD_CONTROL_STEPS) * int(round(next(iter(stress[0].values()))["row_count"] / (MOTOR_STRESS_DRIVE_CONTROL_STEPS + MOTOR_STRESS_HOLD_CONTROL_STEPS))) if complete else -1,
    }
    actual_counts = {phase: sum(1 for row in rows if row["phase"] == phase) for phase in ("nc", "stress")}
    row_count_ok = row_count_ok and actual_counts == expected_counts
    effort_ok = (
        effort_values[0] == {0.5} and effort_values[3] == {0.4}
        and requested_effort_values == effort_values
    )
    invariant_data = guard["control_invariants"]
    invariant_000 = invariant_data["per_env"]["0"]
    invariant_001 = invariant_data["per_env"]["3"]
    invariants_ok = (
        invariant_data["action_type"] == "relative"
        and math.isclose(invariant_000["actual_kp"], invariant_001["actual_kp"], abs_tol=1.0e-7)
        and math.isclose(invariant_000["actual_kd"], invariant_001["actual_kd"], abs_tol=1.0e-7)
        and math.isclose(invariant_data["physics_dt"], dt, abs_tol=1.0e-12)
        and invariant_data["decimation"] == int(guard["decimation"])
    )
    auc_ok = complete and all(item["auc_001"] > item["auc_000"] * (1.0 + MOTOR_MIN_RELATIVE_AUC_GAP) for item in results)
    support_ok = complete and all(
        item["final_excursion_000"] - item["final_excursion_001"] > MOTOR_SUPPORT_NUMERICAL_TOL
        or item["max_abs_qdot_000"] - item["max_abs_qdot_001"] > MOTOR_SUPPORT_NUMERICAL_TOL
        for item in results
    )
    nc_ok = complete and all(item["D_pair_NC"] <= item["NC_bound"] for item in results)
    direction_ok = complete and len(results) == REPEATS and all(item["auc_001"] > item["auc_000"] for item in results)
    trace_ok = row_count_ok and event_ok and target_chain_ok
    add_check(checks, "MOTOR-P1-direct_effort_limit_readback", effort_ok, f"requested={dict(requested_effort_values)}, actual={dict(effort_values)}")
    add_check(checks, "MOTOR-P2-control_invariants", invariants_ok and trace_ok, f"invariants={invariant_data}, counts={actual_counts}, expected={expected_counts}, order={event_ok}, target_chain={target_chain_ok}")
    add_check(checks, "motor_trace_integrity", trace_ok, f"counts={actual_counts}, expected={expected_counts}, order={event_ok}, target_chain={target_chain_ok}")
    add_check(checks, "MOTOR-P3-stress_residual_auc", auc_ok, f"results={results}")
    add_check(checks, "MOTOR-P4-weaker_supporting_response", support_ok, f"results={results}")
    add_check(checks, "MOTOR-P5-negative_control", nc_ok, f"results={results}")
    add_check(checks, "MOTOR-P6-repeat_direction_consistency", direction_ok, f"results={results}")
    protocol_validity_ok = all((effort_ok, invariants_ok, trace_ok, nc_ok, complete))
    status = "PASS" if all((protocol_validity_ok, auc_ok, support_ok, direction_ok)) else "FAIL"
    if protocol_validity_ok and not auc_ok:
        status = "TEST_NOT_EXCITING_CAPACITY"
    return {
        "status": status, "complete": complete, "protocol_validity_ok": protocol_validity_ok,
        "repeats": results, "actual_counts": actual_counts, "expected_counts": expected_counts,
    }


def plot_mass(rows: list[dict[str, str]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for phase, axis, key, ylabel in (
        ("linear", axes[0], "vx_post_mps", "v_x (m/s)"),
        ("angular", axes[1], "omega_z_post_radps", "omega_z (rad/s)"),
    ):
        for env_id, label in ((0, "000 nominal"), (1, "100 high mass")):
            group = [row for row in rows if int(row["repeat"]) == 0 and row["phase"] == phase and int(row["env_id"]) == env_id]
            axis.plot([float(row["sim_time_s"]) for row in group], [float(row[key]) for row in group], label=label)
        axis.set_xlabel("time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("P0-4 Mass/Inertia Same-Wrench Response")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_friction(rows: list[dict[str, str]], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for env_id, label in ((0, "000 nominal friction"), (2, "010 low friction")):
        group = [row for row in rows if int(row["repeat"]) == 0 and row["phase"] == "ramp" and int(row["env_id"]) == env_id]
        force = [float(row["force_x_n"]) for row in group]
        axes[0].plot(force, [float(row["vx_mps"]) for row in group], label=label)
        axes[1].plot(force, [float(row["displacement_m"]) for row in group], label=label)
        onset = next((float(row["onset_force_n"]) for row in group if int(row["onset_flag"]) == 1), None)
        if onset is not None:
            for axis in axes:
                axis.axvline(onset, linestyle="--", alpha=0.6)
    axes[0].axhline(FRICTION_VELOCITY_THRESHOLD_MPS, color="black", linestyle=":", label="velocity threshold")
    axes[1].axhline(FRICTION_DISPLACEMENT_THRESHOLD_M, color="black", linestyle=":", label="displacement threshold")
    axes[0].set_ylabel("v_x (m/s)")
    axes[1].set_ylabel("|x-x0| (m)")
    axes[1].set_xlabel("horizontal force plateau (N)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("P0-5 Controlled-Slip Response")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_motor(rows: list[dict[str, str]], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for env_id, label in ((0, "000 0.50 N"), (3, "001 0.40 N")):
        group = [row for row in rows if int(row["repeat"]) == 0 and row["phase"] == "stress" and int(row["env_id"]) == env_id]
        time = [float(row["sim_time_s"]) for row in group]
        axes[0].plot(time, [float(row["physx_position_target"]) for row in group], linestyle="--", label=f"target {label}")
        axes[0].plot(time, [float(row["q_post_rad"]) for row in group], label=f"q {label}")
        axes[1].plot(time, [abs(float(row["tracking_residual_rad"])) for row in group], label=label)
    axes[0].set_ylabel("joint angle (rad)")
    axes[1].set_ylabel("|target-q| (rad)")
    axes[1].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("P0-3 Same-Command Motor-Capacity Response")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def runtime_versions() -> dict[str, Any]:
    physx_extension_version = None
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        extension_id = manager.get_enabled_extension_id("omni.physx")
        physx_extension_version = manager.get_extension_dict(extension_id)["package"]["version"]
    except Exception:
        physx_extension_version = None
    lab_file = Path(isaaclab.__file__).resolve()
    release_file = next((parent / "VERSION" for parent in lab_file.parents if (parent / "VERSION").is_file()), None)
    return {
        "isaac_sim_version": get_isaac_sim_version()[0],
        "isaac_lab_package_version": getattr(isaaclab, "__version__", "unknown"),
        "isaac_lab_source_file": str(lab_file),
        "isaac_lab_release_version_file": release_file.read_text(encoding="utf-8").strip() if release_file else None,
        "physics_backend": "PhysX", "physx_extension_version": physx_extension_version,
        "physx_sdk_version": None,
    }


def write_checksums(paths: list[Path], checksum_path: Path) -> dict[str, str]:
    hashes = {path.name: sha256_file(path) for path in paths if path.is_file()}
    with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
        for name in sorted(hashes):
            stream.write(f"{hashes[name]}  {name}\n")
    return hashes


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = ARGS.output_dir.resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    mass_csv = output_dir / "p0_physics_mass_raw.csv"
    friction_csv = output_dir / "p0_physics_friction_raw.csv"
    motor_csv = output_dir / "p0_physics_motor_raw.csv"
    summary_path = output_dir / "p0_physics_summary.json"
    checksum_path = output_dir / "p0_physics_checksums.sha256"
    mass_plot = output_dir / "p0_physics_mass.png"
    friction_plot = output_dir / "p0_physics_friction.png"
    motor_plot = output_dir / "p0_physics_motor.png"

    before_hashes = protected_hashes(repo_root)
    dirty_before = full_dirty_state(repo_root)
    tracked_dirty_diff_hash = hashlib.sha256(git_bytes(repo_root, "diff", "--binary", "HEAD")).hexdigest()
    git_commit = git_bytes(repo_root, "rev-parse", "HEAD").decode().strip()
    git_branch = git_bytes(repo_root, "branch", "--show-current").decode().strip()
    git_status_before = git_bytes(repo_root, "status", "--porcelain=v1", "--untracked-files=all").decode().splitlines()
    script_hash = sha256_file(Path(__file__).resolve())

    checks: dict[str, dict[str, Any]] = {}
    protocol_results: dict[str, Any] = {}
    lifecycle = None
    snapshot = None
    probe_metadata: dict[str, Any] = {}
    motor_guard: dict[str, Any] | None = None
    execution_error = None
    error_status = None
    env = None
    sinks: dict[str, CsvSink] = {}
    requested_protocols = ("mass", "friction", "motor") if ARGS.protocol == "all" else (ARGS.protocol,)
    try:
        env_cfg = parse_env_cfg(TASK_ID, device=ARGS.device, num_envs=NUM_ENVS)
        env_cfg.seed = BASE_SEED
        env_cfg.factor_codes = FACTOR_CODES
        add_check(checks, "explicit_factor_assignment", tuple(env_cfg.factor_codes) == FACTOR_CODES, str(tuple(env_cfg.factor_codes)))
        add_check(checks, "events_and_adr_disabled", env_cfg.events is None and env_cfg.enable_adr is False, f"events={env_cfg.events}; adr={env_cfg.enable_adr}")
        env = gym.make(TASK_ID, cfg=env_cfg)
        core = env.unwrapped
        if NUM_ENVS != core.num_envs:
            raise RuntimeError(f"Expected {NUM_ENVS} envs, got {core.num_envs}")
        dt = float(core.sim.get_physics_dt())
        decimation = int(core.cfg.decimation)
        lifecycle = lifecycle_check(env, core, checks)
        reset_environment(env, BASE_SEED)
        snapshot = direct_factor_snapshot(core)
        availability = all(bool(value.all()) for value in snapshot["actual_availability"].values())
        add_check(checks, "direct_factor_readback_available", availability, "all actual availability buffers true")
        requested_actual_ok = all((
            close(snapshot["requested_object_mass"], snapshot["actual_object_mass"]),
            close(snapshot["requested_object_inertia"], snapshot["actual_object_inertia"], atol=1.0e-7),
            close(snapshot["requested_object_static_friction"], snapshot["actual_object_static_friction"]),
            close(snapshot["requested_object_dynamic_friction"], snapshot["actual_object_dynamic_friction"]),
            close(snapshot["requested_effort_limit"], snapshot["actual_effort_limit"]),
        ))
        add_check(checks, "requested_matches_direct_actual", requested_actual_ok, "mass/inertia/friction/effort requested tensors match independent getters")
        factor_buffer_ok = tuple(core.factor_code_labels) == FACTOR_CODES and bool(torch.equal(core.factor_code_bits, core._frozen_factor_code_bits))
        add_check(checks, "factor_assignment_buffer_frozen", factor_buffer_ok, str(tuple(core.factor_code_labels)))
        isolation_specs = {
            "mass": (MASS_ENVS, {"actual_object_mass", "actual_object_inertia"}),
            "friction": (FRICTION_ENVS, {"actual_object_static_friction", "actual_object_dynamic_friction"}),
            "motor": (MOTOR_ENVS, {"actual_effort_limit"}),
        }
        for protocol in requested_protocols:
            pair, allowed = isolation_specs[protocol]
            ok, detail = verify_pair_isolation(snapshot, pair, allowed)
            add_check(checks, f"{protocol}_single_factor_isolation", ok, detail)
        ppo_loaded = any(name == "rl_games" or name.startswith("rl_games.") for name in sys.modules)
        add_check(checks, "no_ppo_or_checkpoint_loaded", not ppo_loaded, f"rl_games_loaded={ppo_loaded}")

        if "mass" in requested_protocols:
            sinks["mass"] = CsvSink(mass_csv, MASS_FIELDS)
            run_mass_protocol(env, core, sinks["mass"], snapshot, dt)
            sinks["mass"].close()
            mass_rows = read_rows(mass_csv)
            protocol_results["mass"] = evaluate_mass(mass_rows, checks)
            plot_mass(mass_rows, mass_plot)

        if "friction" in requested_protocols:
            sinks["friction"] = CsvSink(friction_csv, FRICTION_FIELDS)
            with GroundProbeBinding() as probe:
                probe_metadata = probe.metadata
                bounds = run_friction_protocol(env, core, sinks["friction"], snapshot, dt, probe_metadata)
                probe_metadata["cube_local_bounds"] = bounds
            sinks["friction"].close()
            probe_restored = all(
                bool(probe_metadata.get(field))
                for field in ("restored", "restored_strength", "temporary_material_removed")
            )
            add_check(checks, "friction_probe_binding_restored", probe_restored, json.dumps(probe_metadata, ensure_ascii=False))
            friction_rows = read_rows(friction_csv)
            protocol_results["friction"] = evaluate_friction(friction_rows, checks)
            plot_friction(friction_rows, friction_plot)

        if "motor" in requested_protocols:
            sinks["motor"] = CsvSink(motor_csv, MOTOR_FIELDS)
            motor_guard = run_motor_protocol(env, core, sinks["motor"], dt)
            sinks["motor"].close()
            add_check(checks, "motor_joint_limit_guard", motor_guard["status"] == "PASS", json.dumps(motor_guard, ensure_ascii=False))
            if motor_guard["status"] != "PASS":
                error_status = "MOTOR_PROTOCOL_JOINT_LIMIT_CONFOUNDED"
            else:
                motor_rows = read_rows(motor_csv)
                protocol_results["motor"] = evaluate_motor(motor_rows, checks, dt, motor_guard)
                if protocol_results["motor"]["status"] == "TEST_NOT_EXCITING_CAPACITY":
                    error_status = "TEST_NOT_EXCITING_CAPACITY"
                plot_motor(motor_rows, motor_plot)
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        for sink in sinks.values():
            sink.close()
        if env is not None:
            env.close()

    after_hashes = protected_hashes(repo_root)
    add_check(checks, "protected_bytes_unchanged", before_hashes == after_hashes, "pre/post SHA-256 maps match")
    script_hash_after = sha256_file(Path(__file__).resolve())
    git_commit_after = git_bytes(repo_root, "rev-parse", "HEAD").decode().strip()
    add_check(
        checks, "authoritative_code_identity_unchanged_during_run",
        script_hash_after == script_hash and git_commit_after == git_commit,
        f"script_before={script_hash}; script_after={script_hash_after}; commit_before={git_commit}; commit_after={git_commit_after}",
    )
    dirty_after = full_dirty_state(repo_root)
    git_status_after = git_bytes(repo_root, "status", "--porcelain=v1", "--untracked-files=all").decode().splitlines()
    dt_metadata = float(dt) if "dt" in locals() else None
    decimation_metadata = int(decimation) if "decimation" in locals() else None
    all_checks_pass = bool(checks) and all(item["status"] == "PASS" for item in checks.values())
    failed_check_names = {name for name, item in checks.items() if item["status"] != "PASS"}
    motor_nonexcitation_failures = {"MOTOR-P3-stress_residual_auc", "MOTOR-P4-weaker_supporting_response"}
    if execution_error is not None:
        overall_status = "ERROR"
    elif error_status == "MOTOR_PROTOCOL_JOINT_LIMIT_CONFOUNDED":
        overall_status = error_status
    elif all_checks_pass:
        overall_status = "PASS"
    elif error_status == "TEST_NOT_EXCITING_CAPACITY" and failed_check_names <= motor_nonexcitation_failures:
        overall_status = error_status
    else:
        overall_status = "FAIL"

    report = {
        "overall_status": overall_status, "execution_error": execution_error, "run_id": run_id,
        "protocol_id": "UNIFIED_P0_PHYSICS_VALIDATION", "requested_protocol": ARGS.protocol,
        "task_id": TASK_ID, "factor_schema_version": SCHEMA_ID, "seed": BASE_SEED,
        "repeats": REPEATS, "num_envs": NUM_ENVS, "factor_codes": list(FACTOR_CODES),
        "events": None, "adr_enabled": False, "ppo_used": False, "checkpoint_used": False,
        "measurement_semantics": {
            "mass_inertia_material_effort_limits": "direct PhysX/USD read-back",
            "derived_pair_friction": "derived from two direct material read-backs with verified combine=min; not a directly measured contact-pair coefficient",
            "applied_effort": "unavailable from the implicit-drive API used by this task; raw CSV stores NaN and no commanded/computed torque substitute is used",
        },
        "physics_dt": dt_metadata, "decimation": decimation_metadata,
        "frozen_protocol": {
            "mass": {"force_x_n": MASS_FORCE_X_N, "torque_z_nm": MASS_TORQUE_Z_NM, "steps": MASS_STEPS, "isolation_height_m": ISOLATION_HEIGHT_M, "object_orientation_xyzw": [0.0, 0.0, 0.0, 1.0], "linear_ratio_range": MASS_RATIO_RANGE},
            "friction": {"settle_steps": FRICTION_SETTLE_STEPS, "force_range_n": [FRICTION_FORCE_MIN_N, FRICTION_FORCE_MAX_N], "force_step_n": FRICTION_FORCE_STEP_N, "hold_physics_steps": FRICTION_HOLD_STEPS, "complete_range_executed": True, "velocity_threshold_mps": FRICTION_VELOCITY_THRESHOLD_MPS, "displacement_threshold_m": FRICTION_DISPLACEMENT_THRESHOLD_M, "consecutive_steps_within_plateau": FRICTION_ONSET_CONSECUTIVE_STEPS, "onset_ratio_max": FRICTION_MAX_ONSET_RATIO, "material_readback_scope": "protocol_start_direct_readback"},
            "motor": {"joint_name": MOTOR_JOINT_NAME, "negative_control": {"action": MOTOR_NC_ACTION, "control_steps": MOTOR_NC_CONTROL_STEPS}, "stress": {"action": MOTOR_STRESS_ACTION, "drive_control_steps": MOTOR_STRESS_DRIVE_CONTROL_STEPS, "hold_zero_control_steps": MOTOR_STRESS_HOLD_CONTROL_STEPS}, "joint_limit_tolerance_rad": MOTOR_JOINT_LIMIT_TOL_RAD, "minimum_auc_gap": MOTOR_MIN_RELATIVE_AUC_GAP, "support_numerical_tolerance": MOTOR_SUPPORT_NUMERICAL_TOL},
        },
        "checks": checks, "protocol_results": protocol_results, "lifecycle": lifecycle,
        "factor_snapshot": None if snapshot is None else tensor_to_json(snapshot),
        "probe_metadata": probe_metadata, "motor_joint_limit_guard": motor_guard,
        "versions": runtime_versions(), "git": {
            "branch": git_branch, "commit": git_commit, "commit_after": git_commit_after,
            "status_before": git_status_before,
            "status_after": git_status_after, "dirty_state_before": dirty_before,
            "dirty_state_after": dirty_after, "validation_script_sha256": script_hash,
            "validation_script_sha256_after": script_hash_after,
            "dirty_diff_hash": tracked_dirty_diff_hash,
        },
        "protected_hashes_before": before_hashes, "protected_hashes_after": after_hashes,
        "artifacts": {
            "mass_csv": str(mass_csv) if mass_csv.is_file() else None,
            "friction_csv": str(friction_csv) if friction_csv.is_file() else None,
            "motor_csv": str(motor_csv) if motor_csv.is_file() else None,
            "mass_plot": str(mass_plot) if mass_plot.is_file() else None,
            "friction_plot": str(friction_plot) if friction_plot.is_file() else None,
            "motor_plot": str(motor_plot) if motor_plot.is_file() else None,
        },
    }
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    artifact_paths = [mass_csv, friction_csv, motor_csv, summary_path, mass_plot, friction_plot, motor_plot]
    artifact_hashes = write_checksums(artifact_paths, checksum_path)
    print(json.dumps({
        "overall_status": overall_status, "output_dir": str(output_dir),
        "summary": str(summary_path), "checksums": str(checksum_path),
        "artifact_hashes": artifact_hashes,
        "failed_checks": [name for name, item in checks.items() if item["status"] != "PASS"],
        "execution_error": execution_error,
    }, ensure_ascii=False, indent=2))
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        # On Windows, SimulationApp.close() terminates Kit with code zero before returning.
        # main() has already closed the environment and persisted/flushed every artifact.
        # Preserve the audit's nonzero scientific/error status at the OS boundary.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    SIMULATION_APP.close()
