"""Pre-B0 process-isolated dynamics sensitivity baseline for the frozen LEAP Hand CRI task.

This is a descriptive fixed-policy study.  It does not train, alter the task, or
perform H1/CRDA ambiguity search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "PRE_B0_DYNAMICS_SENSITIVITY_BASELINE_V1"
TASK_ID = "Isaac-Reorient-Cube-Leap-CRI"
SCHEMA_ID = "CRI_FACTOR_SCHEMA_V1"
NUM_ENVS = 1
PAIRED_SEEDS = tuple(range(4200, 4224))
HISTORY_WINDOW = 3
CONTACT_FORCE_THRESHOLD_N = 1.0e-6
NUMERICAL_SEPARATION_TOL = 1.0e-6

CONDITIONS: dict[str, dict[str, Any]] = {
    "D0": {"factor_code": "000", "mass_kg": 0.2160, "friction": 0.80},
    "D1": {"factor_code": "010", "mass_kg": 0.2160, "friction": 0.45},
    "D2": {"factor_code": "100", "mass_kg": 0.2916, "friction": 0.80},
}
PAIR_IDS = (("D0", "D1"), ("D0", "D2"), ("D1", "D2"))
ALLOWED_INTERPRETATIONS = (
    "DYNAMICS_RESPONSE_SEPARATED",
    "DYNAMICS_EFFECT_WEAK",
    "TECHNICAL_FAILURE",
)

REFERENCE_CHECKPOINT = Path("logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth")
REFERENCE_CHECKPOINT_SHA256 = "46675eecc4e51372ec2c336d9637ef588a61b80e24aeb796de263bd1e75ce364"
REFERENCE_AGENT_CFG = Path(
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml"
)
CORE_PROTECTED_PATHS = (
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/factor_conditions.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/leap_hand_cri_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/reorientation_cri_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml",
    str(REFERENCE_CHECKPOINT).replace("\\", "/"),
)


def build_pre_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--static_only", action="store_true")
    parser.add_argument("--technical_smoke", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("logs/b0_dynamics_sensitivity"))
    parser.add_argument("--worker_condition", choices=tuple(CONDITIONS), default=None)
    parser.add_argument("--worker_output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


PRE_ARGS, _ = build_pre_parser().parse_known_args()


def validate_constants() -> None:
    assert tuple(CONDITIONS) == ("D0", "D1", "D2")
    assert CONDITIONS["D0"] == {"factor_code": "000", "mass_kg": 0.216, "friction": 0.80}
    assert CONDITIONS["D1"] == {"factor_code": "010", "mass_kg": 0.216, "friction": 0.45}
    assert CONDITIONS["D2"] == {"factor_code": "100", "mass_kg": 0.2916, "friction": 0.80}
    assert len(PAIRED_SEEDS) == 24 and len(set(PAIRED_SEEDS)) == 24
    assert set(ALLOWED_INTERPRETATIONS) == {
        "DYNAMICS_RESPONSE_SEPARATED",
        "DYNAMICS_EFFECT_WEAK",
        "TECHNICAL_FAILURE",
    }


validate_constants()
if PRE_ARGS.static_only:
    print("PRE_B0_STATIC_CONTRACT_PASS")
    raise SystemExit(0)


APP_LAUNCHER = None
SIMULATION_APP = None
ARGS = PRE_ARGS
if PRE_ARGS.worker_condition is not None:
    from isaaclab.app import AppLauncher

    def build_worker_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Pre-B0 isolated condition worker")
        parser.add_argument("--technical_smoke", action="store_true")
        parser.add_argument("--output_dir", type=Path, default=PRE_ARGS.output_dir)
        parser.add_argument("--worker_condition", choices=tuple(CONDITIONS), required=True)
        parser.add_argument("--worker_output", type=Path, required=True)
        AppLauncher.add_app_launcher_args(parser)
        return parser

    ARGS = build_worker_parser().parse_args()
    APP_LAUNCHER = AppLauncher(ARGS)
    SIMULATION_APP = APP_LAUNCHER.app

    import gymnasium as gym
    import torch
    import warp as wp
    import yaml

    import LEAP_Isaaclab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_physx.physics import PhysxManager as SimulationManager


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_paths(repo_root: Path) -> tuple[str, ...]:
    p0_paths = tuple(
        path.relative_to(repo_root).as_posix() for path in sorted((repo_root / "scripts").glob("p0_*.py"))
    )
    return tuple(dict.fromkeys((*p0_paths, *CORE_PROTECTED_PATHS)))


def protected_hashes(repo_root: Path) -> dict[str, str | None]:
    return {
        relative: sha256_file(repo_root / relative) if (repo_root / relative).is_file() else None
        for relative in protected_paths(repo_root)
    }


def json_safe(value: Any) -> Any:
    if "torch" in globals() and isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_create(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def git_text(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(("git", "-C", str(repo_root), *args), capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else f"UNAVAILABLE:{completed.stderr.strip()}"


def tensor_numpy(tensor: Any, dtype: Any = np.float32) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(dtype, copy=True)


def quat_mul_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )


def target_axis_increment(previous: np.ndarray, current: np.ndarray) -> float:
    prev = previous / np.linalg.norm(previous)
    curr = current / np.linalg.norm(current)
    conjugate = np.asarray((-prev[0], -prev[1], -prev[2], prev[3]), dtype=np.float64)
    delta = quat_mul_xyzw(curr, conjugate)
    if delta[3] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[:3]))
    if vector_norm < 1.0e-12:
        return 0.0
    angle = 2.0 * math.atan2(vector_norm, max(float(delta[3]), 0.0))
    return float(delta[2] * angle / vector_norm)


def validate_j_rot_semantics() -> dict[str, Any]:
    def zquat(angle: float) -> np.ndarray:
        return np.asarray((0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)), dtype=np.float64)

    forward = target_axis_increment(zquat(0.0), zquat(0.2))
    reverse = target_axis_increment(zquat(0.2), zquat(0.1))
    tilt = target_axis_increment(
        zquat(0.0), np.asarray((math.sin(0.1), 0.0, 0.0, math.cos(0.1)), dtype=np.float64)
    )
    passed = (
        math.isclose(forward, 0.2, abs_tol=1.0e-9)
        and math.isclose(reverse, -0.1, abs_tol=1.0e-9)
        and abs(tilt) < 1.0e-9
    )
    return {"status": "PASS" if passed else "FAIL", "forward": forward, "reverse": reverse, "tilt": tilt}


class ReferenceController:
    """Strict inference-only loader for the frozen RL-Games policy."""

    def __init__(self, env: Any, repo_root: Path):
        self.raw_env = env
        self.repo_root = repo_root
        self.wrapper: Any = None
        self.agent: Any = None

    def strict_load(self) -> dict[str, Any]:
        checkpoint_path = self.repo_root / REFERENCE_CHECKPOINT
        actual_hash = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
        if actual_hash != REFERENCE_CHECKPOINT_SHA256:
            return {
                "status": "FAIL",
                "reason": "checkpoint_sha_mismatch",
                "expected_sha256": REFERENCE_CHECKPOINT_SHA256,
                "actual_sha256": actual_hash,
            }
        try:
            from rl_games.common import env_configurations, vecenv
            from rl_games.common.player import BasePlayer
            from rl_games.torch_runner import Runner
            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

            with (self.repo_root / REFERENCE_AGENT_CFG).open("r", encoding="utf-8") as stream:
                agent_cfg = yaml.safe_load(stream)
            rl_device = agent_cfg["params"]["config"]["device"]
            clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
            clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
            self.wrapper = RlGamesVecEnvWrapper(self.raw_env, rl_device, clip_obs, clip_actions)
            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
            )
            env_configurations.register(
                "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: self.wrapper}
            )
            agent_cfg["params"]["load_checkpoint"] = True
            agent_cfg["params"]["load_path"] = str(checkpoint_path)
            agent_cfg["params"]["config"]["num_actors"] = NUM_ENVS
            runner = Runner()
            runner.load(agent_cfg)
            agent: BasePlayer = runner.create_player()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            expected_state = checkpoint["model"]
            model_state = agent.model.state_dict()
            missing = sorted(set(model_state) - set(expected_state))
            unexpected = sorted(set(expected_state) - set(model_state))
            shape_mismatch = {
                key: [list(model_state[key].shape), list(expected_state[key].shape)]
                for key in set(model_state).intersection(expected_state)
                if tuple(model_state[key].shape) != tuple(expected_state[key].shape)
            }
            if missing or unexpected or shape_mismatch:
                raise RuntimeError(
                    f"strict state mismatch: missing={missing}, unexpected={unexpected}, shapes={shape_mismatch}"
                )
            agent.restore(str(checkpoint_path))
            agent.reset()
            self.agent = agent
            obs_shape = tuple(self.wrapper.observation_space.shape)
            action_shape = tuple(self.wrapper.action_space.shape)
            if obs_shape != (96,) or action_shape != (16,):
                raise RuntimeError(f"Controller contract mismatch: obs={obs_shape}, action={action_shape}")
            return {
                "status": "PASS",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": actual_hash,
                "observation_shape": list(obs_shape),
                "action_shape": list(action_shape),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "shape_mismatch": shape_mismatch,
                "training_performed": False,
            }
        except Exception as exc:
            return {
                "status": "FAIL",
                "reason": f"{type(exc).__name__}: {exc}",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": actual_hash,
                "traceback": traceback.format_exc(),
            }

    def reset(self, seed: int) -> tuple[Any, str, dict[str, Any], int]:
        if self.wrapper is None or self.agent is None:
            raise RuntimeError("Reference controller is not loaded")
        torch.manual_seed(seed)
        core = self.raw_env.unwrapped
        # The baseline reset does not clear its temporal observation buffer.  Clear
        # only this evaluation state before reset so every paired episode starts
        # from the same finite-history boundary without changing observation IO.
        core.obs_hist_buf.zero_()
        obs_dict, _ = self.raw_env.reset(seed=seed)
        processed = self.wrapper._process_obs(obs_dict)
        if not isinstance(processed, dict) or "obs" not in processed:
            raise RuntimeError("Official RL-Games observation mapping lacks obs key")
        obs = processed["obs"]
        q = tensor_numpy(core.hand.data.joint_pos.torch[0], np.float64)
        qdot = tensor_numpy(core.hand.data.joint_vel.torch[0], np.float64)
        pose = tensor_numpy(
            torch.cat((core.object.data.root_pos_w.torch[0], core.object.data.root_quat_w.torch[0])), np.float64
        )
        velocity = tensor_numpy(core.object.data.root_vel_w.torch[0], np.float64)
        initial_policy_obs = policy_vector(obs).astype(np.float64)
        physical_fingerprint = hashlib.sha256(
            q.tobytes() + qdot.tobytes() + pose.tobytes() + velocity.tobytes()
        ).hexdigest()
        policy_obs_fingerprint = hashlib.sha256(initial_policy_obs.tobytes()).hexdigest()
        fingerprint = hashlib.sha256(
            q.tobytes()
            + qdot.tobytes()
            + pose.tobytes()
            + velocity.tobytes()
            + initial_policy_obs.tobytes()
        ).hexdigest()
        planned_horizon_steps = int(core.randomized_episode_lengths[0].item())
        self.agent.reset()
        _ = self.agent.get_batch_size(obs, 1)
        if self.agent.is_rnn:
            self.agent.init_rnn()
        initial_state = {
            "q_rad": q,
            "qdot_radps": qdot,
            "object_pose_world_xyzw": pose,
            "object_velocity_world": velocity,
            "initial_policy_observation": initial_policy_obs,
            "physical_state_sha256": physical_fingerprint,
            "policy_observation_sha256": policy_obs_fingerprint,
            "combined_initial_state_sha256": fingerprint,
        }
        return obs, fingerprint, initial_state, planned_horizon_steps

    def action(self, obs: Any) -> Any:
        if self.agent is None:
            raise RuntimeError("Reference controller is not loaded")
        with torch.inference_mode():
            return self.agent.get_action(self.agent.obs_to_torch(obs), is_deterministic=True)


class ObjectHandContactReader:
    """Read actual PhysX pairwise object-to-hand contact forces without changing the task."""

    def __init__(self, core: Any):
        sim_view = SimulationManager.get_physics_sim_view()
        if sim_view is None:
            raise RuntimeError("PhysX simulation view is unavailable")
        object_paths = list(core.object.root_view.prim_paths)
        if len(object_paths) != 1:
            raise RuntimeError(f"Expected one object rigid body, got {object_paths}")
        hand_link_paths = list(core.hand.root_view.link_paths[0])
        if not hand_link_paths:
            raise RuntimeError("LEAP articulation has no rigid links")
        self.view = sim_view.create_rigid_contact_view(object_paths[0], filter_patterns=hand_link_paths)
        if self.view.sensor_count != 1 or self.view.filter_count < 1:
            raise RuntimeError(
                f"Invalid contact view: sensors={self.view.sensor_count}, filters={self.view.filter_count}"
            )
        self.dt = float(core.cfg.sim.dt)
        self.metadata = {
            "source": "PhysX RigidContactView.get_contact_force_matrix",
            "contact_report_api_enabled_on_object_by_evaluation_script": True,
            "dynamics_or_task_io_changed": False,
            "sensor_path": object_paths[0],
            "filter_paths": hand_link_paths,
            "sensor_count": int(self.view.sensor_count),
            "filter_count": int(self.view.filter_count),
            "force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
            "proxy_used": False,
        }

    def read(self) -> tuple[bool, float]:
        forces = self.view.get_contact_force_matrix(self.dt)
        if isinstance(forces, torch.Tensor):
            tensor = forces
        else:
            tensor = wp.to_torch(forces)
        tensor = tensor.reshape(int(self.view.sensor_count), int(self.view.filter_count), 3)
        norm = float(torch.linalg.vector_norm(tensor[0], dim=-1).sum().item())
        return norm > CONTACT_FORCE_THRESHOLD_N, norm


class StepCapture:
    """Capture post-step state before DirectRLEnv performs an automatic reset."""

    def __init__(self, core: Any, contact_reader: ObjectHandContactReader):
        self.core = core
        self.contact_reader = contact_reader
        self.original = core._get_dones
        self.latest: dict[str, Any] | None = None

    def install(self) -> None:
        def capture() -> tuple[Any, Any]:
            terminated, truncated = self.original()
            contact_state, contact_force_norm = self.contact_reader.read()
            self.latest = {
                "terminated": bool(terminated[0].item()),
                "truncated": bool(truncated[0].item()),
                "q": tensor_numpy(self.core.hand.data.joint_pos.torch[0]),
                "qdot": tensor_numpy(self.core.hand.data.joint_vel.torch[0]),
                "object_pose": tensor_numpy(
                    torch.cat(
                        (
                            self.core.object.data.root_pos_w.torch[0],
                            self.core.object.data.root_quat_w.torch[0],
                        )
                    )
                ),
                "object_velocity": tensor_numpy(self.core.object.data.root_vel_w.torch[0]),
                "contact_state": contact_state,
                "contact_force_norm_n": contact_force_norm,
            }
            return terminated, truncated

        self.core._get_dones = capture

    def restore(self) -> None:
        self.core._get_dones = self.original


def create_env(factor_code: str) -> Any:
    cfg = parse_env_cfg(TASK_ID, device=ARGS.device, num_envs=NUM_ENVS)
    cfg.factor_codes = (factor_code,)
    cfg.events = None
    cfg.enable_adr = False
    # Evaluation-only reporting instrumentation. This applies PhysxContactReportAPI
    # but does not alter collision, material, mass, timing, reward, observation, or action.
    cfg.object_cfg.spawn.activate_contact_sensors = True
    env = gym.make(TASK_ID, cfg=cfg)
    core = env.unwrapped
    if tuple(core.factor_code_labels) != (factor_code,):
        env.close()
        raise RuntimeError(f"Factor assignment mismatch: {tuple(core.factor_code_labels)}")
    if core.cfg.events is not None or core.cfg.enable_adr:
        env.close()
        raise RuntimeError("Pre-B0 requires events=None and ADR disabled")
    return env


def factor_readback(core: Any, condition_id: str) -> dict[str, Any]:
    snapshot = core.get_factor_snapshot()
    actual_mass = float(snapshot["actual_object_mass"][0, 0].item())
    actual_friction = float(snapshot["actual_object_static_friction"][0].item())
    expected = CONDITIONS[condition_id]
    mass_ok = math.isclose(actual_mass, float(expected["mass_kg"]), rel_tol=1.0e-5, abs_tol=1.0e-6)
    friction_ok = math.isclose(actual_friction, float(expected["friction"]), rel_tol=0.0, abs_tol=1.0e-6)
    status = "PASS" if mass_ok and friction_ok else "FAIL"
    return {
        "status": status,
        "condition": condition_id,
        "factor_code": snapshot["factor_codes"][0],
        "requested_mass_kg": float(snapshot["requested_object_mass"][0, 0].item()),
        "actual_mass_kg": actual_mass,
        "requested_static_friction": float(snapshot["requested_object_static_friction"][0].item()),
        "actual_static_friction": actual_friction,
        "actual_dynamic_friction": float(snapshot["actual_object_dynamic_friction"][0].item()),
        "hand_static_friction": float(snapshot["actual_hand_static_friction"][0].item()),
        "hand_dynamic_friction": float(snapshot["actual_hand_dynamic_friction"][0].item()),
        "object_combine_mode": snapshot["actual_object_combine_mode"][0],
        "hand_combine_mode": snapshot["actual_hand_combine_mode"][0],
        "actual_effort_limit": tensor_numpy(snapshot["actual_effort_limit"][0]),
    }


def policy_vector(obs: Any) -> np.ndarray:
    value = obs["obs"] if isinstance(obs, dict) else obs
    array = tensor_numpy(value[0] if value.ndim > 1 else value)
    if array.shape != (96,):
        raise RuntimeError(f"Policy observation shape changed: {array.shape}")
    return array


def allocate_trajectory(seed_count: int, max_steps: int) -> dict[str, np.ndarray]:
    return {
        "valid_mask": np.zeros((seed_count, max_steps), dtype=np.bool_),
        "policy_obs": np.full((seed_count, max_steps, 96), np.nan, dtype=np.float32),
        "raw_action": np.full((seed_count, max_steps, 16), np.nan, dtype=np.float32),
        "q": np.full((seed_count, max_steps, 16), np.nan, dtype=np.float32),
        "qdot": np.full((seed_count, max_steps, 16), np.nan, dtype=np.float32),
        "object_pose": np.full((seed_count, max_steps, 7), np.nan, dtype=np.float32),
        "object_velocity": np.full((seed_count, max_steps, 6), np.nan, dtype=np.float32),
        "cube_rotation_angle": np.full((seed_count, max_steps), np.nan, dtype=np.float32),
        "cube_angular_velocity": np.full((seed_count, max_steps, 3), np.nan, dtype=np.float32),
        "contact_state": np.zeros((seed_count, max_steps), dtype=np.bool_),
        "contact_force_norm_n": np.full((seed_count, max_steps), np.nan, dtype=np.float32),
        "drop_event": np.zeros((seed_count, max_steps), dtype=np.bool_),
        "survival": np.zeros((seed_count, max_steps), dtype=np.bool_),
        "terminated": np.zeros((seed_count, max_steps), dtype=np.bool_),
        "truncated": np.zeros((seed_count, max_steps), dtype=np.bool_),
        "episode_time_s": np.full((seed_count, max_steps), np.nan, dtype=np.float32),
    }


def run_condition_worker(repo_root: Path, condition_id: str, output_path: Path) -> dict[str, Any]:
    env: Any | None = None
    capture: StepCapture | None = None
    condition = CONDITIONS[condition_id]
    seeds = PAIRED_SEEDS[:2] if ARGS.technical_smoke else PAIRED_SEEDS
    env = create_env(str(condition["factor_code"]))
    core = env.unwrapped
    controller = ReferenceController(env, repo_root)
    policy_loading = controller.strict_load()
    if policy_loading["status"] != "PASS":
        env.close()
        raise RuntimeError(f"Reference checkpoint strict-load failed: {policy_loading.get('reason')}")
    contact_reader = ObjectHandContactReader(core)
    capture = StepCapture(core, contact_reader)
    capture.install()
    max_steps = 4 if ARGS.technical_smoke else int(core.max_episode_length) + 2
    arrays = allocate_trajectory(len(seeds), max_steps)
    planned_horizons = np.zeros(len(seeds), dtype=np.int32)
    completed_steps = np.zeros(len(seeds), dtype=np.int32)
    initial_hashes: list[str] = []
    initial_states: list[dict[str, Any]] = []
    material_checks: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    try:
        for episode_index, seed in enumerate(seeds):
            obs, fingerprint, initial_state, planned_horizon = controller.reset(seed)
            readback = factor_readback(core, condition_id)
            if readback["status"] != "PASS":
                raise RuntimeError(f"Condition readback mismatch: {readback}")
            initial_hashes.append(fingerprint)
            initial_states.append(initial_state)
            planned_horizons[episode_index] = planned_horizon
            material_checks.append({"seed": seed, **readback})
            previous_quat = np.asarray(initial_state["object_pose_world_xyzw"][3:7], dtype=np.float64)
            accumulated_rotation = 0.0
            dropped = False
            episode_complete = False
            step_limit = 4 if ARGS.technical_smoke else max_steps
            for step in range(step_limit):
                arrays["policy_obs"][episode_index, step] = policy_vector(obs)
                actions = controller.action(obs)
                arrays["raw_action"][episode_index, step] = tensor_numpy(actions[0])
                capture.latest = None
                obs, _, _, _ = controller.wrapper.step(actions)
                state = capture.latest
                if state is None:
                    raise RuntimeError("Post-step state capture did not execute")
                current_quat = np.asarray(state["object_pose"][3:7], dtype=np.float64)
                accumulated_rotation += target_axis_increment(previous_quat, current_quat)
                dropped_now = bool(state["terminated"])
                dropped = dropped or dropped_now
                arrays["valid_mask"][episode_index, step] = True
                arrays["q"][episode_index, step] = state["q"]
                arrays["qdot"][episode_index, step] = state["qdot"]
                arrays["object_pose"][episode_index, step] = state["object_pose"]
                arrays["object_velocity"][episode_index, step] = state["object_velocity"]
                arrays["cube_rotation_angle"][episode_index, step] = accumulated_rotation
                arrays["cube_angular_velocity"][episode_index, step] = state["object_velocity"][3:6]
                arrays["contact_state"][episode_index, step] = bool(state["contact_state"])
                arrays["contact_force_norm_n"][episode_index, step] = float(state["contact_force_norm_n"])
                arrays["drop_event"][episode_index, step] = dropped_now
                arrays["survival"][episode_index, step] = not dropped
                arrays["terminated"][episode_index, step] = bool(state["terminated"])
                arrays["truncated"][episode_index, step] = bool(state["truncated"])
                arrays["episode_time_s"][episode_index, step] = (step + 1) * float(core.step_dt)
                completed_steps[episode_index] = step + 1
                previous_quat = current_quat
                if state["terminated"] or state["truncated"]:
                    episode_complete = True
                    break
            if not ARGS.technical_smoke and not episode_complete:
                raise RuntimeError(
                    f"Seed {seed} did not terminate/truncate within {max_steps} steps; planned={planned_horizon}"
                )
            last = int(completed_steps[episode_index]) - 1
            duration = max(float(planned_horizon) * float(core.step_dt), 1.0e-12)
            episode_metrics.append(
                {
                    "seed": seed,
                    "planned_horizon_steps": planned_horizon,
                    "recorded_steps": int(completed_steps[episode_index]),
                    "initial_state_hash": fingerprint,
                    "j_rot_radps": float(arrays["cube_rotation_angle"][episode_index, last] / duration),
                    "survival": not dropped,
                    "drop": dropped,
                    "contact_fraction": float(
                        arrays["contact_state"][episode_index, : completed_steps[episode_index]].mean()
                    ),
                    "episode_complete": episode_complete,
                }
            )
    finally:
        if capture is not None:
            capture.restore()
        if env is not None:
            env.close()

    npz_path = output_path.parent / "condition_trajectory.npz"
    np.savez_compressed(
        npz_path,
        condition_id=np.asarray(condition_id),
        factor_code=np.asarray(condition["factor_code"]),
        seeds=np.asarray(seeds, dtype=np.int64),
        planned_horizon_steps=planned_horizons,
        completed_steps=completed_steps,
        initial_state_hashes=np.asarray(initial_hashes, dtype="U64"),
        **arrays,
    )
    dt = float(core.cfg.sim.dt)
    decimation = int(core.cfg.decimation)
    return {
        "protocol_id": PROTOCOL_ID,
        "condition": condition_id,
        "factor_code": condition["factor_code"],
        "pid": os.getpid(),
        "scientific_eligible": not ARGS.technical_smoke,
        "technical_smoke_only": bool(ARGS.technical_smoke),
        "policy_loading": policy_loading,
        "physics_dt": dt,
        "decimation": decimation,
        "control_dt": dt * decimation,
        "native_horizon_preserved": True,
        "native_horizon_policy": "seeded task randomized_episode_lengths; no buffer override",
        "policy_history_initialization": "obs_hist_buf zeroed immediately before task reset",
        "observation_space_or_definition_changed": False,
        "seeds": list(seeds),
        "planned_horizon_steps": planned_horizons,
        "completed_steps": completed_steps,
        "initial_state_hashes": initial_hashes,
        "initial_states": initial_states,
        "factor_readbacks": material_checks,
        "contact_readout": contact_reader.metadata,
        "episodes": episode_metrics,
        "trajectory_npz": str(npz_path),
        "ppo_reward_used": False,
        "training_performed": False,
        "domain_randomization": False,
        "H1_ambiguity_search_performed": False,
    }


def run_worker() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = ARGS.worker_output.resolve()
    before = protected_hashes(repo_root)
    result: dict[str, Any] | None = None
    fatal_error = None
    try:
        result = run_condition_worker(repo_root, str(ARGS.worker_condition), output_path)
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    after = protected_hashes(repo_root)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "condition": ARGS.worker_condition,
        "pid": os.getpid(),
        "technical_smoke_only": bool(ARGS.technical_smoke),
        "result": result,
        "fatal_error": fatal_error,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_unchanged": before == after,
        "simulation_app_close_contract": "ENTRYPOINT_FINALLY",
    }
    write_json_create(output_path, payload)
    print(f"PRE_B0_WORKER_OUTPUT={output_path}", flush=True)
    print(f"PRE_B0_WORKER_STATUS={'FAILED_TECHNICAL' if fatal_error else 'COLLECTED'}", flush=True)
    return 1 if fatal_error is not None or before != after else 0


def launch_worker(
    script_path: Path, condition_id: str, output_path: Path, device: str | None, technical_smoke: bool
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script_path),
        "--worker_condition",
        condition_id,
        "--worker_output",
        str(output_path),
        "--viz",
        "none",
    ]
    if technical_smoke:
        command.append("--technical_smoke")
    if device is not None:
        command.extend(("--device", device))
    started = datetime.now().isoformat(timespec="microseconds")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    finished = datetime.now().isoformat(timespec="microseconds")
    log_path = output_path.with_suffix(".log")
    log_path.write_text(
        f"COMMAND={json.dumps(command)}\nSTARTED={started}\nFINISHED={finished}\n"
        f"RETURN_CODE={completed.returncode}\n--- STDOUT ---\n{completed.stdout}\n--- STDERR ---\n{completed.stderr}",
        encoding="utf-8",
    )
    if not output_path.is_file():
        return {
            "condition": condition_id,
            "worker_return_code": completed.returncode,
            "worker_log": str(log_path),
            "fatal_error": {
                "type": "WORKER_OUTPUT_MISSING",
                "message": f"return_code={completed.returncode}; see {log_path}",
            },
        }
    payload = read_json(output_path)
    payload.update(
        {
            "worker_return_code": completed.returncode,
            "worker_log": str(log_path),
            "parent_started": started,
            "parent_finished": finished,
        }
    )
    return payload


def condition_metrics(arrays: dict[str, np.ndarray], condition_index: int, planned: np.ndarray) -> dict[str, Any]:
    valid = arrays["valid_mask"][condition_index]
    episodes = valid.shape[0]
    j_rot: list[float] = []
    survival: list[bool] = []
    contact_fraction: list[float] = []
    for episode in range(episodes):
        indices = np.flatnonzero(valid[episode])
        if indices.size == 0:
            raise RuntimeError(f"Condition index {condition_index}, episode {episode} has no valid data")
        last = int(indices[-1])
        duration = float(planned[condition_index, episode]) * float(arrays["control_dt"][condition_index])
        j_rot.append(float(arrays["cube_rotation_angle"][condition_index, episode, last] / duration))
        survival.append(not bool(arrays["drop_event"][condition_index, episode, indices].any()))
        contact_fraction.append(float(arrays["contact_state"][condition_index, episode, indices].mean()))
    return {
        "episodes": episodes,
        "median_j_rot_radps": float(np.median(j_rot)),
        "mean_j_rot_radps": float(np.mean(j_rot)),
        "survival_rate": float(np.mean(survival)),
        "drop_probability": float(1.0 - np.mean(survival)),
        "mean_contact_fraction": float(np.mean(contact_fraction)),
        "per_episode_j_rot_radps": j_rot,
        "per_episode_survival": survival,
        "per_episode_contact_fraction": contact_fraction,
    }


def pair_metrics(arrays: dict[str, np.ndarray], left_index: int, right_index: int) -> dict[str, Any]:
    joint_valid = arrays["valid_mask"][left_index] & arrays["valid_mask"][right_index]
    action_mae_episode: list[float] = []
    action_rms_episode: list[float] = []
    obs_l2_values: list[float] = []
    history_values: list[float] = []
    early_action: list[float] = []
    early_history: list[float] = []
    aligned_steps = 0
    early_steps = int(round(1.0 / float(arrays["control_dt"][left_index])))
    for episode in range(joint_valid.shape[0]):
        indices = np.flatnonzero(joint_valid[episode])
        if indices.size == 0:
            raise RuntimeError(f"Pair episode {episode} has no aligned valid steps")
        aligned_steps += int(indices.size)
        action_delta = (
            arrays["raw_action"][left_index, episode, indices]
            - arrays["raw_action"][right_index, episode, indices]
        )
        obs_delta = (
            arrays["policy_obs"][left_index, episode, indices]
            - arrays["policy_obs"][right_index, episode, indices]
        )
        action_mae_episode.append(float(np.mean(np.abs(action_delta))))
        action_rms_episode.append(float(np.sqrt(np.mean(np.square(action_delta)))))
        obs_l2 = np.linalg.norm(obs_delta, axis=-1)
        obs_l2_values.extend(obs_l2.tolist())
        for local_index in range(HISTORY_WINDOW - 1, len(indices)):
            window_indices = indices[local_index - HISTORY_WINDOW + 1 : local_index + 1]
            if not np.all(np.diff(window_indices) == 1):
                continue
            window_delta = (
                arrays["policy_obs"][left_index, episode, window_indices]
                - arrays["policy_obs"][right_index, episode, window_indices]
            )
            history_values.append(float(np.sqrt(np.mean(np.square(window_delta)))))
        early_count = min(len(indices), early_steps)
        early_action.append(float(np.mean(np.abs(action_delta[:early_count]))))
        early_obs = obs_delta[:early_count]
        early_history.append(float(np.sqrt(np.mean(np.square(early_obs)))))
    return {
        "aligned_steps": aligned_steps,
        "action_mean_absolute_difference": float(np.mean(action_mae_episode)),
        "trajectory_action_rms_divergence_median": float(np.median(action_rms_episode)),
        "trajectory_action_rms_divergence_per_episode": action_rms_episode,
        "observation_l2_distance_mean": float(np.mean(obs_l2_values)),
        "observation_l2_distance_median": float(np.median(obs_l2_values)),
        "history_window": HISTORY_WINDOW,
        "history_window_rms_distance_mean": float(np.mean(history_values)),
        "history_window_rms_distance_median": float(np.median(history_values)),
        "first_second_action_mae": float(np.mean(early_action)),
        "first_second_history_rms_distance": float(np.mean(early_history)),
        "analysis_scope": "descriptive paired sensitivity; no ambiguity/CRDA search",
    }


def combine_trajectories(workers: dict[str, dict[str, Any]], output_path: Path) -> dict[str, np.ndarray]:
    loaded: list[dict[str, np.ndarray]] = []
    for condition_id in CONDITIONS:
        result = workers[condition_id]["result"]
        with np.load(result["trajectory_npz"], allow_pickle=False) as data:
            loaded.append({key: data[key].copy() for key in data.files})
    array_keys = (
        "valid_mask",
        "policy_obs",
        "raw_action",
        "q",
        "qdot",
        "object_pose",
        "object_velocity",
        "cube_rotation_angle",
        "cube_angular_velocity",
        "contact_state",
        "contact_force_norm_n",
        "drop_event",
        "survival",
        "terminated",
        "truncated",
        "episode_time_s",
    )
    combined: dict[str, np.ndarray] = {key: np.stack([item[key] for item in loaded], axis=0) for key in array_keys}
    combined["condition_ids"] = np.asarray(tuple(CONDITIONS), dtype="U2")
    combined["factor_codes"] = np.asarray([CONDITIONS[key]["factor_code"] for key in CONDITIONS], dtype="U3")
    combined["seeds"] = loaded[0]["seeds"]
    combined["planned_horizon_steps"] = np.stack([item["planned_horizon_steps"] for item in loaded])
    combined["completed_steps"] = np.stack([item["completed_steps"] for item in loaded])
    combined["initial_state_hashes"] = np.stack([item["initial_state_hashes"] for item in loaded])
    combined["control_dt"] = np.asarray(
        [float(workers[key]["result"]["control_dt"]) for key in CONDITIONS], dtype=np.float64
    )
    np.savez_compressed(output_path, **combined)
    return combined


def orchestrate() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "technical_smoke_" if PRE_ARGS.technical_smoke else ""
    base = PRE_ARGS.output_dir if PRE_ARGS.output_dir.is_absolute() else repo_root / PRE_ARGS.output_dir
    output_dir = (base / f"{prefix}{run_id}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    before = protected_hashes(repo_root)
    checkpoint_path = repo_root / REFERENCE_CHECKPOINT
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    workers: dict[str, dict[str, Any]] = {}
    fatal_error = None
    interpretation = "TECHNICAL_FAILURE"
    performance: dict[str, Any] = {}
    pairwise: dict[str, Any] = {}
    pairing: dict[str, Any] = {}
    try:
        if checkpoint_sha != REFERENCE_CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Checkpoint SHA mismatch: actual={checkpoint_sha}, expected={REFERENCE_CHECKPOINT_SHA256}"
            )
        if validate_j_rot_semantics()["status"] != "PASS":
            raise RuntimeError("J_rot semantics validation failed")
        for condition_id in CONDITIONS:
            condition_dir = output_dir / condition_id
            condition_dir.mkdir()
            worker = launch_worker(
                script_path,
                condition_id,
                condition_dir / "worker_result.json",
                PRE_ARGS.device,
                PRE_ARGS.technical_smoke,
            )
            workers[condition_id] = worker
            if worker.get("worker_return_code") != 0 or worker.get("fatal_error"):
                raise RuntimeError(f"Worker {condition_id} failed: {worker.get('fatal_error')}")
            if not worker.get("protected_unchanged"):
                raise RuntimeError(f"Worker {condition_id} changed a protected file")
        if not PRE_ARGS.technical_smoke:
            trajectory_path = output_dir / "trajectory_data.npz"
            arrays = combine_trajectories(workers, trajectory_path)
            hashes = arrays["initial_state_hashes"]
            horizons = arrays["planned_horizon_steps"]
            pairing = {
                "paired_seed_count": len(PAIRED_SEEDS),
                "initial_state_hashes_match": bool(np.all(hashes == hashes[0:1])),
                "native_horizon_steps_match": bool(np.all(horizons == horizons[0:1])),
                "condition_processes_unique": len({workers[key]["pid"] for key in CONDITIONS}) == len(CONDITIONS),
            }
            if not all(pairing.values()):
                raise RuntimeError(f"Paired-design validation failed: {pairing}")
            for index, condition_id in enumerate(CONDITIONS):
                performance[condition_id] = condition_metrics(arrays, index, horizons)
            for left, right in PAIR_IDS:
                left_index = tuple(CONDITIONS).index(left)
                right_index = tuple(CONDITIONS).index(right)
                pairwise[f"{left}-{right}"] = pair_metrics(arrays, left_index, right_index)
            max_action = max(item["action_mean_absolute_difference"] for item in pairwise.values())
            max_history = max(item["history_window_rms_distance_mean"] for item in pairwise.values())
            endpoint_difference = max(
                max(
                    abs(performance[left]["median_j_rot_radps"] - performance[right]["median_j_rot_radps"]),
                    abs(performance[left]["survival_rate"] - performance[right]["survival_rate"]),
                    abs(performance[left]["mean_contact_fraction"] - performance[right]["mean_contact_fraction"]),
                )
                for left, right in PAIR_IDS
            )
            interpretation = (
                "DYNAMICS_RESPONSE_SEPARATED"
                if max_action > NUMERICAL_SEPARATION_TOL
                and max_history > NUMERICAL_SEPARATION_TOL
                and endpoint_difference > NUMERICAL_SEPARATION_TOL
                else "DYNAMICS_EFFECT_WEAK"
            )
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        interpretation = "TECHNICAL_FAILURE"

    after = protected_hashes(repo_root)
    if before != after and fatal_error is None:
        fatal_error = {"type": "PROTECTED_FILE_CHANGE", "message": "Protected hashes changed during Pre-B0"}
        interpretation = "TECHNICAL_FAILURE"
    metadata = {
        condition_id: {
            "condition": condition_id,
            **condition,
            "pid": workers.get(condition_id, {}).get("pid"),
            "factor_readback": (workers.get(condition_id, {}).get("result") or {}).get("factor_readbacks", [None])[0],
            "contact_readout": (workers.get(condition_id, {}).get("result") or {}).get("contact_readout"),
            "physics_dt": (workers.get(condition_id, {}).get("result") or {}).get("physics_dt"),
            "decimation": (workers.get(condition_id, {}).get("result") or {}).get("decimation"),
        }
        for condition_id, condition in CONDITIONS.items()
    }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "overall_status": "TECHNICAL_SMOKE_PASS" if PRE_ARGS.technical_smoke and fatal_error is None else "COMPLETE" if fatal_error is None else "FAILED_TECHNICAL",
        "scientific_eligible": not PRE_ARGS.technical_smoke and fatal_error is None,
        "technical_smoke_only": bool(PRE_ARGS.technical_smoke),
        "scientific_interpretation": interpretation,
        "interpretation_basis": {
            "type": "descriptive numerical separation",
            "numerical_tolerance": NUMERICAL_SEPARATION_TOL,
            "requires": "non-identical action, history trajectory, and at least one performance/contact endpoint",
            "effect_sizes_reported_without_H1_or_CRDA_claim": True,
        },
        "fatal_error": fatal_error,
        "policy_loading": {
            key: (workers.get(key, {}).get("result") or {}).get("policy_loading", {}).get("status", "FAIL")
            for key in CONDITIONS
        },
        "conditions": metadata,
        "pairing": pairing,
        "performance": performance,
        "pairwise_sensitivity": pairwise,
        "workers": workers,
        "reference_checkpoint": {
            "path": str(REFERENCE_CHECKPOINT),
            "expected_sha256": REFERENCE_CHECKPOINT_SHA256,
            "actual_sha256": checkpoint_sha,
        },
        "native_task_horizon_preserved": True,
        "ppo_reward_used": False,
        "training_performed": False,
        "domain_randomization": False,
        "H1_ambiguity_search_performed": False,
        "j_rot_semantics": validate_j_rot_semantics(),
        "git_commit": git_text(repo_root, "rev-parse", "HEAD"),
        "git_branch": git_text(repo_root, "branch", "--show-current"),
        "script_sha256": sha256_file(script_path),
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_unchanged": before == after,
    }
    write_json_create(output_dir / "condition_metadata.json", metadata)
    write_json_create(output_dir / "summary.json", summary)
    with (output_dir / "checkpoint_hash.txt").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{checkpoint_sha}  {REFERENCE_CHECKPOINT.as_posix()}\n")
    print(f"PRE_B0_OUTPUT_DIR={output_dir}", flush=True)
    print(f"PRE_B0_POLICY_LOADING={summary['policy_loading']}", flush=True)
    print(f"PRE_B0_INTERPRETATION={interpretation}", flush=True)
    print(f"PRE_B0_STATUS={summary['overall_status']}", flush=True)
    if fatal_error is not None:
        print(f"PRE_B0_FATAL={fatal_error['type']}: {fatal_error['message']}", flush=True)
    return 1 if fatal_error is not None else 0


if __name__ == "__main__":
    if PRE_ARGS.worker_condition is None:
        raise SystemExit(orchestrate())
    worker_exit = 1
    try:
        worker_exit = run_worker()
    finally:
        if SIMULATION_APP is not None:
            SIMULATION_APP.close()
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(worker_exit)
