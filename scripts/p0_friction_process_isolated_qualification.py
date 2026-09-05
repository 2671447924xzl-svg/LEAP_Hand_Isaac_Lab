"""P0-7F1 process-isolated friction qualification for the frozen LEAP Hand CRI task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


PROTOCOL_ID = "P0_7F1_PROCESS_ISOLATED_FRICTION_QUALIFICATION"  # 本轮唯一协议标识。
TASK_ID = "Isaac-Reorient-Cube-Leap-CRI"  # 复用冻结CRI任务。
SCHEMA_ID = "CRI_FACTOR_SCHEMA_V1"  # 只读记录，不修改schema。
FACTOR_CODES = ("000",)  # 单环境显式名义assignment，候选仅做一次直接材料写入。
NUM_ENVS = 1  # 每个物理条件独占一个单环境进程。
BASE_SEED = 42  # P0-5物理协议固定seed。
REPEATS = 3  # P0-5固定重复数。

FRICTION_NOMINAL = 0.80  # 唯一名义参照。
FRICTION_CANDIDATES = (0.45, 0.40, 0.35, 0.30)  # 禁止扩展或自动选择。
ALL_FRICTIONS = (FRICTION_NOMINAL, *FRICTION_CANDIDATES)  # 串行进程顺序。
MATERIAL_ATOL = 1.0e-6  # 与CRI材料断言一致。

FRICTION_SETTLE_STEPS = 120  # P0-5冻结稳定时长。
FRICTION_FORCE_MIN_N = 0.0  # P0-5冻结force ramp起点。
FRICTION_FORCE_MAX_N = 2.20  # P0-5冻结force ramp终点。
FRICTION_FORCE_STEP_N = 0.01  # P0-5冻结force increment。
FRICTION_HOLD_STEPS = 8  # 每个平台固定8个physics steps。
FRICTION_VELOCITY_THRESHOLD_MPS = 0.01  # slip onset速度阈值。
FRICTION_DISPLACEMENT_THRESHOLD_M = 0.0005  # slip onset位移阈值。
FRICTION_ONSET_CONSECUTIVE_STEPS = 5  # 必须在同一plateau连续满足。
FRICTION_MAX_ONSET_RATIO = 0.75  # 唯一physical qualification阈值。
FRICTION_CLEARANCE_M = 0.001  # cube置于probe上方的固定间隙。

PROBE_ROOT_PATH = "/World/ground"  # 仅P0-5 audit使用的固定probe。
PROBE_TEMP_MATERIAL_PATH = "/World/P0_7F1_audit_ground_material"  # worker内临时材料。
PROBE_STATIC_FRICTION = 0.80  # probe静摩擦。
PROBE_DYNAMIC_FRICTION = 0.80  # probe动摩擦。
PROBE_COMBINE_MODE = "min"  # 与CRI材料组合语义一致。

REFERENCE_CHECKPOINT = Path("logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth")  # 唯一控制器。
REFERENCE_CHECKPOINT_SHA256 = "46675eecc4e51372ec2c336d9637ef588a61b80e24aeb796de263bd1e75ce364"  # 冻结哈希。
REFERENCE_AGENT_CFG = Path(
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml"
)  # 只读加载原agent结构。
PAIRED_EPISODE_SEEDS = tuple(range(4200, 4224))  # 固定24个paired seeds。
MANIPULATION_HORIZON_S = 20.0  # 固定screen horizon。
EARLY_DROP_S = 1.0  # 冻结early-drop边界。

SCIENTIFIC_CLASSES = (
    "MATERIAL_INVALID",
    "FRICTION_EFFECT_TOO_WEAK",
    "FRICTION_CAUSES_GRASP_COLLAPSE",
    "QUALIFIED",
)  # candidate只允许四种结论。
FINAL_CONCLUSIONS = (
    "FRICTION_PRIMARY_FACTOR_QUALIFIED",
    "FRICTION_PRIMARY_FACTOR_NOT_QUALIFIED",
)  # 最终结论全集。

PROTECTED_PATHS = (
    "scripts/p0_physics_validation.py",
    "scripts/p0_timing_audit.py",
    "scripts/p0_friction_coupling_diagnosis.py",
    "scripts/p0_friction_material_diagnosis.py",
    "scripts/p0_friction_material_sync_diagnosis.py",
    "scripts/p0_friction_material_sequence_bisect.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/factor_conditions.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/leap_hand_cri_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/reorientation_cri_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml",
    str(REFERENCE_CHECKPOINT).replace("\\", "/"),
)  # 运行前后必须逐文件保持字节不变。

SLIP_FIELDS = (
    "protocol_id",
    "pid",
    "friction",
    "repeat",
    "phase",
    "level_index",
    "plateau_substep",
    "physics_step",
    "sim_time_s",
    "force_x_n",
    "actual_mass_kg",
    "object_static_friction",
    "object_dynamic_friction",
    "object_combine_mode",
    "probe_static_friction",
    "probe_dynamic_friction",
    "probe_combine_mode",
    "derived_pair_friction",
    "x_m",
    "vx_mps",
    "displacement_m",
    "velocity_condition",
    "displacement_condition",
    "within_plateau_consecutive",
    "onset_flag",
    "onset_force_n",
)  # 每个physics step的原始P0-5证据。

MANIPULATION_FIELDS = (
    "protocol_id",
    "pid",
    "friction",
    "seed",
    "control_step",
    "sim_time_s",
    "object_qx",
    "object_qy",
    "object_qz",
    "object_qw",
    "incremental_target_axis_rotation_rad",
    "accumulated_target_axis_rotation_rad",
    "yaw_unwrapped_rad",
    "dropped",
    "terminated",
    "truncated",
    "termination_reason",
    "episode_complete",
    "time_to_drop_s",
    "early_drop",
    "survival",
    "episode_j_rot_radps",
    "initial_state_sha256",
)  # reward-free manipulation trajectory。


def build_pre_parser() -> argparse.ArgumentParser:  # 父编排器不创建SimulationApp。
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--static_only", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("logs/p0_friction_process_isolated"))
    parser.add_argument("--worker_mode", choices=("none", "physical", "manipulation"), default="none")
    parser.add_argument("--worker_output", type=Path)
    parser.add_argument("--friction", type=float)
    parser.add_argument("--device", type=str, default=None)
    return parser


PRE_ARGS, _ = build_pre_parser().parse_known_args()  # 先隔离父进程和仿真worker参数。


def validate_constants() -> None:  # 启动Kit前拒绝任何协议漂移。
    assert FACTOR_CODES == ("000",) and NUM_ENVS == 1
    assert FRICTION_CANDIDATES == (0.45, 0.40, 0.35, 0.30)
    assert ALL_FRICTIONS == (0.80, 0.45, 0.40, 0.35, 0.30)
    assert REPEATS == 3 and BASE_SEED == 42
    assert (FRICTION_FORCE_MIN_N, FRICTION_FORCE_MAX_N, FRICTION_FORCE_STEP_N) == (0.0, 2.20, 0.01)
    assert FRICTION_HOLD_STEPS == 8 and FRICTION_ONSET_CONSECUTIVE_STEPS == 5
    assert FRICTION_VELOCITY_THRESHOLD_MPS == 0.01
    assert FRICTION_DISPLACEMENT_THRESHOLD_M == 0.0005
    assert FRICTION_MAX_ONSET_RATIO == 0.75
    assert len(PAIRED_EPISODE_SEEDS) == 24 and len(set(PAIRED_EPISODE_SEEDS)) == 24
    assert set(SCIENTIFIC_CLASSES) == {
        "MATERIAL_INVALID",
        "FRICTION_EFFECT_TOO_WEAK",
        "FRICTION_CAUSES_GRASP_COLLAPSE",
        "QUALIFIED",
    }
    assert len(FINAL_CONCLUSIONS) == 2


validate_constants()
if PRE_ARGS.static_only:
    print("P0_7F1_STATIC_CONTRACT_PASS")
    raise SystemExit(0)


APP_LAUNCHER = None  # 父进程保持无Kit状态。
SIMULATION_APP = None
ARGS = PRE_ARGS
if PRE_ARGS.worker_mode != "none":
    from isaaclab.app import AppLauncher

    def build_worker_parser() -> argparse.ArgumentParser:  # 每个worker独立解析Isaac参数。
        parser = argparse.ArgumentParser(description="P0-7F1 isolated Isaac Sim worker.")
        parser.add_argument("--static_only", action="store_true")
        parser.add_argument("--output_dir", type=Path, default=PRE_ARGS.output_dir)
        parser.add_argument("--worker_mode", choices=("physical", "manipulation"), required=True)
        parser.add_argument("--worker_output", type=Path, required=True)
        parser.add_argument("--friction", type=float, required=True)
        AppLauncher.add_app_launcher_args(parser)
        return parser

    ARGS = build_worker_parser().parse_args()
    APP_LAUNCHER = AppLauncher(ARGS)  # worker各自拥有唯一SimulationApp lifecycle。
    SIMULATION_APP = APP_LAUNCHER.app

    import gymnasium as gym
    import torch
    import warp as wp
    import yaml
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    import LEAP_Isaaclab.tasks  # noqa: F401
    from isaaclab.sim.utils import get_current_stage, get_first_matching_child_prim
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg


def sha256_file(path: Path) -> str:  # 流式计算checkpoint与artifact哈希。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes(repo_root: Path) -> dict[str, str | None]:  # 缺失保护项显式写为null。
    return {
        path: sha256_file(repo_root / path) if (repo_root / path).is_file() else None for path in PROTECTED_PATHS
    }


def git_text(repo_root: Path, *args: str) -> str:  # 只读记录Git代码身份。
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *args), capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else f"UNAVAILABLE:{completed.stderr.strip()}"


def json_safe(value: Any) -> Any:  # 禁止JSON写入NaN或设备Tensor。
    if "torch" in globals() and isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:  # create-only写入科研证据。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> dict[str, Any]:  # 父进程只读worker结果。
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def ftext(value: float) -> str:  # CSV中稳定表示有限值与NaN。
    return "NaN" if not math.isfinite(value) else f"{value:.12g}"


def friction_tag(value: float) -> str:  # 目录名固定为三位小数编码。
    return f"friction_{int(round(value * 100)):03d}"


class CsvSink:  # create-only CSV流式落盘器。
    def __init__(self, path: Path, fields: tuple[str, ...]):
        self.path = path
        self.stream = path.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)

    def close(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


def create_env() -> Any:  # 每个worker仅创建一个all-nominal CRI环境。
    cfg = parse_env_cfg(TASK_ID, device=ARGS.device, num_envs=NUM_ENVS)
    cfg.factor_codes = FACTOR_CODES
    cfg.events = None
    cfg.enable_adr = False
    env = gym.make(TASK_ID, cfg=cfg)
    core = env.unwrapped
    if tuple(core.factor_code_labels) != FACTOR_CODES:
        env.close()
        raise RuntimeError(f"Factor assignment mismatch: {tuple(core.factor_code_labels)}")
    if core.cfg.events is not None or core.cfg.enable_adr:
        env.close()
        raise RuntimeError("P0-7F1 requires events=None and ADR disabled")
    return env


def close_env(env: Any | None) -> None:  # 清除audit wrench并关闭当前环境。
    if env is None:
        return
    try:
        env.unwrapped.object.permanent_wrench_composer.reset()
        env.unwrapped.hand.permanent_wrench_composer.reset()
    except Exception:
        pass
    env.close()


def reset_env(env: Any, seed: int) -> tuple[Any, Any]:  # 保持固定seed reset边界。
    torch.manual_seed(seed)
    return env.reset(seed=seed)


def direct_physics_step(core: Any, dt: float) -> None:  # 固定write→step→update顺序。
    core.scene.write_data_to_sim()
    core.sim.step(render=False)
    core.scene.update(dt)


def write_condition_friction(core: Any, friction: float) -> dict[str, Any]:  # 单进程只写其唯一条件。
    ids = torch.tensor((0,), dtype=torch.long, device=core.device)
    values = torch.tensor((friction,), dtype=torch.float32, device=core.device)
    before = core._read_materials(core.object, "object")
    core._write_object_material(ids, values, values)
    actual = core._read_materials(core.object, "object")
    hand = core._read_materials(core.hand, "hand")
    object_mode = core._read_combine_mode(0, "object", "cri_object_physics_material")
    hand_mode = core._read_combine_mode(0, "Robot", "cri_hand_physics_material")
    static_ok = bool(torch.allclose(actual[0, :, 0], torch.full_like(actual[0, :, 0], friction), atol=MATERIAL_ATOL, rtol=0.0))
    dynamic_ok = bool(torch.allclose(actual[0, :, 1], torch.full_like(actual[0, :, 1], friction), atol=MATERIAL_ATOL, rtol=0.0))
    hand_ok = bool(torch.allclose(hand[0, :, :2], torch.full_like(hand[0, :, :2], 0.80), atol=MATERIAL_ATOL, rtol=0.0))
    passed = static_ok and dynamic_ok and hand_ok and object_mode == "min" and hand_mode == "min"
    mass = wp.to_torch(core.object.root_view.get_masses()).to(device=core.device)
    return {
        "status": "PASS" if passed else "INVALID",
        "requested_static": friction,
        "requested_dynamic": friction,
        "before_static": float(before[0, 0, 0]),
        "before_dynamic": float(before[0, 0, 1]),
        "actual_static": float(actual[0, 0, 0]),
        "actual_dynamic": float(actual[0, 0, 1]),
        "object_combine_mode": object_mode,
        "hand_static": float(hand[0, 0, 0]),
        "hand_dynamic": float(hand[0, 0, 1]),
        "hand_combine_mode": hand_mode,
        "mass_kg": float(mass[0, 0]),
        "physics_step_count": int(core.sim.get_physics_step_count()),
        "simulation_time_s": float(core.sim.physics_manager.get_simulation_time()),
    }


class GroundProbeBinding:  # 临时P0-5 ground material绑定并严格恢复。
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.stage = get_current_stage()
        self.binding_api: Any = None
        self.prior_path: Any = None
        self.prior_strength: Any = None
        self.metadata: dict[str, Any] = {"probe_path": root_path, "restored": False}

    def __enter__(self) -> dict[str, Any]:
        collider = get_first_matching_child_prim(
            self.root_path, predicate=lambda prim: prim.GetTypeName() == "Plane", stage=self.stage
        )
        if collider is None or not collider.IsValid():
            raise RuntimeError(f"No collision Plane under {self.root_path}")
        self.binding_api = UsdShade.MaterialBindingAPI(collider)
        prior = self.binding_api.GetDirectBinding("physics")
        self.prior_path = prior.GetMaterialPath()
        self.prior_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(prior.GetBindingRel())
        if not self.prior_path:
            raise RuntimeError("Ground probe lacks a restorable direct physics material")
        if self.stage.GetPrimAtPath(PROBE_TEMP_MATERIAL_PATH).IsValid():
            raise RuntimeError(f"Temporary material already exists: {PROBE_TEMP_MATERIAL_PATH}")
        self.metadata.update(
            {
                "collider_path": str(collider.GetPath()),
                "temporary_material_path": PROBE_TEMP_MATERIAL_PATH,
                "prior_material_path": str(self.prior_path),
                "prior_binding_strength": str(self.prior_strength),
                "requested_static_friction": PROBE_STATIC_FRICTION,
                "requested_dynamic_friction": PROBE_DYNAMIC_FRICTION,
                "requested_combine_mode": PROBE_COMBINE_MODE,
                "restored_strength": False,
                "temporary_material_removed": False,
            }
        )
        try:
            cfg = PhysxRigidBodyMaterialCfg(
                static_friction=PROBE_STATIC_FRICTION,
                dynamic_friction=PROBE_DYNAMIC_FRICTION,
                restitution=0.0,
                friction_combine_mode=PROBE_COMBINE_MODE,
            )
            temp_prim = cfg.func(PROBE_TEMP_MATERIAL_PATH, cfg)
            self.binding_api.Bind(
                UsdShade.Material(temp_prim),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            static = float(UsdPhysics.MaterialAPI(temp_prim).GetStaticFrictionAttr().Get())
            dynamic = float(UsdPhysics.MaterialAPI(temp_prim).GetDynamicFrictionAttr().Get())
            mode = str(PhysxSchema.PhysxMaterialAPI(temp_prim).GetFrictionCombineModeAttr().Get())
            bound = self.binding_api.GetDirectBinding("physics").GetMaterialPath()
            if bound != temp_prim.GetPath():
                raise RuntimeError(f"Temporary probe binding mismatch: {bound} != {temp_prim.GetPath()}")
            if not (
                math.isclose(static, PROBE_STATIC_FRICTION, abs_tol=MATERIAL_ATOL)
                and math.isclose(dynamic, PROBE_DYNAMIC_FRICTION, abs_tol=MATERIAL_ATOL)
                and mode == PROBE_COMBINE_MODE
            ):
                raise RuntimeError(f"Probe readback mismatch: static={static}, dynamic={dynamic}, mode={mode}")
            self.metadata.update(
                {"actual_static_friction": static, "actual_dynamic_friction": dynamic, "actual_combine_mode": mode}
            )
            return self.metadata
        except Exception:
            self._restore()
            raise

    def _restore(self) -> None:
        if self.binding_api is None or self.prior_path is None or self.prior_strength is None:
            return
        self.binding_api.UnbindDirectBinding("physics")
        prior_prim = self.stage.GetPrimAtPath(self.prior_path)
        if not prior_prim.IsValid():
            raise RuntimeError(f"Prior probe material disappeared: {self.prior_path}")
        self.binding_api.Bind(
            UsdShade.Material(prior_prim), bindingStrength=self.prior_strength, materialPurpose="physics"
        )
        restored = self.binding_api.GetDirectBinding("physics")
        restored_path = restored.GetMaterialPath()
        restored_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(restored.GetBindingRel())
        if self.stage.GetPrimAtPath(PROBE_TEMP_MATERIAL_PATH).IsValid():
            self.stage.RemovePrim(PROBE_TEMP_MATERIAL_PATH)
        temp_removed = not self.stage.GetPrimAtPath(PROBE_TEMP_MATERIAL_PATH).IsValid()
        self.metadata.update(
            {
                "restored_material_path": str(restored_path),
                "restored_binding_strength": str(restored_strength),
                "restored": restored_path == self.prior_path,
                "restored_strength": restored_strength == self.prior_strength,
                "temporary_material_removed": temp_removed,
            }
        )
        if not (self.metadata["restored"] and self.metadata["restored_strength"] and temp_removed):
            raise RuntimeError("P0-5 probe material restoration failed")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._restore()


def cube_half_height(core: Any) -> float:  # 从当前USD几何读取实际半高。
    prim_path = f"{core.scene.env_prim_paths[0]}/object"
    prim = get_current_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Cube prim missing: {prim_path}")
    local_range = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]
    ).ComputeLocalBound(prim).ComputeAlignedRange()
    size = tuple(float(value) for value in local_range.GetSize())
    if not all(math.isfinite(value) for value in size) or min(size) <= 0.0:
        raise RuntimeError(f"Invalid cube local size: {size}")
    return 0.5 * size[2]


def set_object_state_on_probe(core: Any, half_height: float) -> None:  # 固定pose且清零速度。
    ids = torch.tensor((0,), dtype=torch.long, device=core.device)
    pose = core.object.data.default_root_pose.torch[ids].clone()
    pose[:, :3] = core.scene.env_origins[ids]
    pose[:, 2] = half_height + FRICTION_CLEARANCE_M
    pose[:, 3:7] = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=pose.dtype, device=core.device)
    velocity = torch.zeros((1, 6), dtype=pose.dtype, device=core.device)
    core.object.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    core.object.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)


def set_object_wrench(core: Any, force_x: float) -> None:  # 对单一cube施加world-frame水平力。
    ids = torch.tensor((0,), dtype=torch.long, device=core.device)
    forces = torch.tensor((force_x, 0.0, 0.0), dtype=torch.float32, device=core.device).reshape(1, 1, 3)
    torques = torch.zeros_like(forces)
    core.object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces, torques=torques, env_ids=ids, is_global=True
    )


def run_one_slip_repeat(
    core: Any,
    sink: CsvSink,
    friction: float,
    repeat: int,
    dt: float,
    probe: dict[str, Any],
) -> dict[str, Any]:  # 完整执行一次冻结P0-5 ramp。
    half_height = cube_half_height(core)
    set_object_state_on_probe(core, half_height)
    core.object.permanent_wrench_composer.reset()
    set_object_wrench(core, 0.0)
    start_step = int(core.sim.get_physics_step_count())
    mass = wp.to_torch(core.object.root_view.get_masses()).to(device=core.device)
    actual_material = core._read_materials(core.object, "object")
    actual_mu_s = float(actual_material[0, 0, 0])
    actual_mu_d = float(actual_material[0, 0, 1])
    object_mode = core._read_combine_mode(0, "object", "cri_object_physics_material")
    probe_mu = float(probe["actual_static_friction"])
    for settle_step in range(FRICTION_SETTLE_STEPS):
        direct_physics_step(core, dt)
        pos = core.object.data.root_pos_w.torch
        vel = core.object.data.root_lin_vel_w.torch
        sink.write(
            {
                "protocol_id": PROTOCOL_ID,
                "pid": os.getpid(),
                "friction": ftext(friction),
                "repeat": repeat,
                "phase": "settle",
                "level_index": -1,
                "plateau_substep": settle_step,
                "physics_step": int(core.sim.get_physics_step_count()) - start_step,
                "sim_time_s": ftext((int(core.sim.get_physics_step_count()) - start_step) * dt),
                "force_x_n": ftext(0.0),
                "actual_mass_kg": ftext(float(mass[0, 0])),
                "object_static_friction": ftext(actual_mu_s),
                "object_dynamic_friction": ftext(actual_mu_d),
                "object_combine_mode": object_mode,
                "probe_static_friction": ftext(float(probe["actual_static_friction"])),
                "probe_dynamic_friction": ftext(float(probe["actual_dynamic_friction"])),
                "probe_combine_mode": probe["actual_combine_mode"],
                "derived_pair_friction": ftext(min(actual_mu_s, probe_mu)),
                "x_m": ftext(float(pos[0, 0])),
                "vx_mps": ftext(float(vel[0, 0])),
                "displacement_m": ftext(0.0),
                "velocity_condition": 0,
                "displacement_condition": 0,
                "within_plateau_consecutive": 0,
                "onset_flag": 0,
                "onset_force_n": "NaN",
            }
        )
    x0 = float(core.object.data.root_pos_w.torch[0, 0])
    onset_force = math.nan
    onset_found = False
    levels = np.round(
        np.arange(FRICTION_FORCE_MIN_N, FRICTION_FORCE_MAX_N + 0.5 * FRICTION_FORCE_STEP_N, FRICTION_FORCE_STEP_N),
        10,
    )
    try:
        for level_index, force_x in enumerate(levels):
            consecutive = 0
            set_object_wrench(core, float(force_x))
            for substep in range(FRICTION_HOLD_STEPS):
                direct_physics_step(core, dt)
                pos = core.object.data.root_pos_w.torch
                vel = core.object.data.root_lin_vel_w.torch
                displacement = abs(float(pos[0, 0]) - x0)
                velocity_ok = abs(float(vel[0, 0])) >= FRICTION_VELOCITY_THRESHOLD_MPS
                displacement_ok = displacement >= FRICTION_DISPLACEMENT_THRESHOLD_M
                if not onset_found and velocity_ok and displacement_ok:
                    consecutive += 1
                elif not onset_found:
                    consecutive = 0
                onset_flag = False
                if not onset_found and consecutive >= FRICTION_ONSET_CONSECUTIVE_STEPS:
                    onset_found = True
                    onset_force = float(force_x)
                    onset_flag = True
                sink.write(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "pid": os.getpid(),
                        "friction": ftext(friction),
                        "repeat": repeat,
                        "phase": "ramp",
                        "level_index": level_index,
                        "plateau_substep": substep,
                        "physics_step": int(core.sim.get_physics_step_count()) - start_step,
                        "sim_time_s": ftext((int(core.sim.get_physics_step_count()) - start_step) * dt),
                        "force_x_n": ftext(float(force_x)),
                        "actual_mass_kg": ftext(float(mass[0, 0])),
                        "object_static_friction": ftext(actual_mu_s),
                        "object_dynamic_friction": ftext(actual_mu_d),
                        "object_combine_mode": object_mode,
                        "probe_static_friction": ftext(float(probe["actual_static_friction"])),
                        "probe_dynamic_friction": ftext(float(probe["actual_dynamic_friction"])),
                        "probe_combine_mode": probe["actual_combine_mode"],
                        "derived_pair_friction": ftext(min(actual_mu_s, probe_mu)),
                        "x_m": ftext(float(pos[0, 0])),
                        "vx_mps": ftext(float(vel[0, 0])),
                        "displacement_m": ftext(displacement),
                        "velocity_condition": int(velocity_ok),
                        "displacement_condition": int(displacement_ok),
                        "within_plateau_consecutive": consecutive,
                        "onset_flag": int(onset_flag),
                        "onset_force_n": ftext(onset_force),
                    }
                )
    finally:
        core.object.permanent_wrench_composer.reset()
    return {
        "repeat": repeat,
        "onset_force_n": onset_force,
        "onset_found": onset_found,
        "start_physics_step": start_step,
        "end_physics_step": int(core.sim.get_physics_step_count()),
        "actual_static_friction": actual_mu_s,
        "actual_dynamic_friction": actual_mu_d,
    }


def quat_mul_xyzw(left: Any, right: Any) -> Any:  # 项目xyzw四元数乘法。
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )


def quat_conjugate_xyzw(quat: Any) -> Any:  # 单位四元数共轭。
    return np.array((-quat[0], -quat[1], -quat[2], quat[3]), dtype=np.float64)


def target_axis_increment(previous: Any, current: Any) -> float:  # quaternion增量投影到world z轴。
    prev = previous / np.linalg.norm(previous)
    curr = current / np.linalg.norm(current)
    delta = quat_mul_xyzw(curr, quat_conjugate_xyzw(prev))
    if delta[3] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[:3]))
    if vector_norm < 1.0e-12:
        return 0.0
    angle = 2.0 * math.atan2(vector_norm, max(float(delta[3]), 0.0))
    return float(delta[2] * angle / vector_norm)


def yaw_xyzw(quat: Any) -> float:  # 支持性unwrapped yaw，不替代J_rot主指标。
    x, y, z, w = quat / np.linalg.norm(quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def validate_j_rot_semantics() -> dict[str, Any]:  # 合成正反转检查J_rot符号语义。
    def zquat(angle: float) -> Any:
        return np.array((0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)), dtype=np.float64)

    forward = target_axis_increment(zquat(0.0), zquat(0.2))
    reverse = target_axis_increment(zquat(0.2), zquat(0.1))
    tilt = target_axis_increment(zquat(0.0), np.array((math.sin(0.1), 0.0, 0.0, math.cos(0.1))))
    passed = (
        math.isclose(forward, 0.2, abs_tol=1.0e-9)
        and math.isclose(reverse, -0.1, abs_tol=1.0e-9)
        and abs(tilt) < 1.0e-9
    )
    return {"status": "PASS" if passed else "FAIL", "forward": forward, "reverse": reverse, "tilt": tilt}


class DonesCapture:  # 在DirectRLEnv自动reset前保留真实terminal state。
    def __init__(self, core: Any):
        self.core = core
        self.original = core._get_dones
        self.quat: Any = None
        self.terminated: Any = None
        self.truncated: Any = None

    def install(self) -> None:
        def capture() -> tuple[Any, Any]:
            terminated, truncated = self.original()
            self.quat = self.core.object_rot.clone()
            self.terminated = terminated.clone()
            self.truncated = truncated.clone()
            return terminated, truncated

        self.core._get_dones = capture

    def restore(self) -> None:
        self.core._get_dones = self.original


class ReferenceController:  # 只加载冻结checkpoint，不训练或读取reward。
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
            }
        except Exception as exc:
            return {
                "status": "FAIL",
                "reason": f"{type(exc).__name__}: {exc}",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": actual_hash,
                "traceback": traceback.format_exc(),
            }

    def reset_with_seed(self, seed: int, horizon_steps: int) -> tuple[Any, str, dict[str, Any]]:
        if self.wrapper is None or self.agent is None:
            raise RuntimeError("Reference controller is not loaded")
        torch.manual_seed(seed)  # paired process使用完全相同的Torch reset seed。
        obs_dict, _ = self.raw_env.reset(seed=seed)
        core = self.raw_env.unwrapped
        core.randomized_episode_lengths[:] = horizon_steps + 1
        processed = self.wrapper._process_obs(obs_dict)
        if not isinstance(processed, dict) or "obs" not in processed:
            raise RuntimeError("Official RL-Games observation mapping lacks obs key")
        obs = processed["obs"]
        q = core.hand.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
        qdot = core.hand.data.joint_vel.torch[0].detach().cpu().numpy().astype(np.float64)
        cube_pose = torch.cat(
            (core.object.data.root_pos_w.torch[0], core.object.data.root_quat_w.torch[0])
        ).detach().cpu().numpy().astype(np.float64)
        cube_velocity = core.object.data.root_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
        fingerprint = hashlib.sha256(
            q.tobytes() + qdot.tobytes() + cube_pose.tobytes() + cube_velocity.tobytes()
        ).hexdigest()
        self.agent.reset()
        _ = self.agent.get_batch_size(obs, 1)
        if self.agent.is_rnn:
            self.agent.init_rnn()
        initial_state = {
            "q_rad": q.tolist(),
            "qdot_radps": qdot.tolist(),
            "cube_pose_xyzw": cube_pose.tolist(),
            "cube_velocity": cube_velocity.tolist(),
        }
        return obs, fingerprint, initial_state

    def action(self, obs: Any) -> Any:
        if self.agent is None:
            raise RuntimeError("Reference controller is not loaded")
        with torch.inference_mode():
            return self.agent.get_action(self.agent.obs_to_torch(obs), is_deterministic=True)


def run_physical_worker(repo_root: Path, friction: float, output_path: Path) -> dict[str, Any]:  # 单条件P0-5 worker。
    env: Any | None = None
    sink: CsvSink | None = None
    probe_binding: GroundProbeBinding | None = None
    result: dict[str, Any] = {}
    env = create_env()
    core = env.unwrapped
    dt = float(core.cfg.sim.dt)
    decimation = int(core.cfg.decimation)
    reset_env(env, BASE_SEED)
    initial_material = write_condition_friction(core, friction)
    checkpoint = ReferenceController(env, repo_root)
    controller_status = checkpoint.strict_load()
    if controller_status["status"] != "PASS":
        close_env(env)
        raise RuntimeError(f"Reference checkpoint strict-load failed: {controller_status.get('reason')}")
    csv_path = output_path.parent / "controlled_slip_raw.csv"
    sink = CsvSink(csv_path, SLIP_FIELDS)
    repeats: list[dict[str, Any]] = []
    material_checks = [initial_material]
    material_valid = initial_material["status"] == "PASS"
    try:
        if material_valid:
            probe_binding = GroundProbeBinding(PROBE_ROOT_PATH)
            with probe_binding as probe:
                for repeat in range(REPEATS):
                    if repeat > 0:
                        try:
                            reset_env(env, BASE_SEED)
                            repeat_material = write_condition_friction(core, friction)
                        except Exception as exc:
                            repeat_material = {
                                "status": "INVALID",
                                "requested_static": friction,
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                            }
                        material_checks.append(repeat_material)
                        if repeat_material["status"] != "PASS":
                            material_valid = False
                            break
                    repeats.append(run_one_slip_repeat(core, sink, friction, repeat, dt, probe))
    finally:
        if sink is not None:
            sink.close()
        probe_metadata = probe_binding.metadata if probe_binding is not None else None
        mass = initial_material.get("mass_kg")
        result = {
            "protocol_id": PROTOCOL_ID,
            "worker_mode": "physical",
            "pid": os.getpid(),
            "friction": friction,
            "material_readback": initial_material,
            "material_checks": material_checks,
            "material_valid": material_valid and len(material_checks) == REPEATS,
            "mass": mass,
            "checkpoint_sha256": controller_status.get("checkpoint_sha256"),
            "checkpoint_load": controller_status,
            "physics_dt": dt,
            "decimation": decimation,
            "seed": BASE_SEED,
            "factor_codes": list(FACTOR_CODES),
            "events": None,
            "adr_enabled": False,
            "controlled_slip_protocol": {
                "settle_steps": FRICTION_SETTLE_STEPS,
                "force_range_n": [FRICTION_FORCE_MIN_N, FRICTION_FORCE_MAX_N],
                "force_step_n": FRICTION_FORCE_STEP_N,
                "hold_physics_steps": FRICTION_HOLD_STEPS,
                "velocity_threshold_mps": FRICTION_VELOCITY_THRESHOLD_MPS,
                "displacement_threshold_m": FRICTION_DISPLACEMENT_THRESHOLD_M,
                "consecutive_steps_within_plateau": FRICTION_ONSET_CONSECUTIVE_STEPS,
                "probe": probe_metadata,
            },
            "slip_repeats": repeats,
            "slip_raw_csv": str(csv_path),
        }
        close_env(env)
    return result


def run_manipulation_worker(repo_root: Path, friction: float, output_path: Path) -> dict[str, Any]:  # 单条件24-seed worker。
    env: Any | None = None
    sink: CsvSink | None = None
    capture: DonesCapture | None = None
    env = create_env()
    core = env.unwrapped
    controller = ReferenceController(env, repo_root)
    controller_status = controller.strict_load()
    if controller_status["status"] != "PASS":
        close_env(env)
        raise RuntimeError(f"Reference checkpoint strict-load failed: {controller_status.get('reason')}")
    dt = float(core.cfg.sim.dt)
    decimation = int(core.cfg.decimation)
    horizon_steps = int(round(MANIPULATION_HORIZON_S / float(core.step_dt)))
    csv_path = output_path.parent / "manipulation_raw.csv"
    sink = CsvSink(csv_path, MANIPULATION_FIELDS)
    capture = DonesCapture(core)
    capture.install()
    episodes: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    material_checks: list[dict[str, Any]] = []
    material_valid = True
    try:
        for seed in PAIRED_EPISODE_SEEDS:
            try:
                obs, fingerprint, initial_state = controller.reset_with_seed(seed, horizon_steps)
                material = write_condition_friction(core, friction)
            except Exception as exc:
                material = {
                    "status": "INVALID",
                    "requested_static": friction,
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            material_checks.append({"seed": seed, **material})
            if material["status"] != "PASS":
                material_valid = False
                break
            fingerprints[str(seed)] = fingerprint
            previous = core.object.data.root_quat_w.torch[0].detach().cpu().numpy().astype(np.float64)
            previous_yaw = yaw_xyzw(previous)
            unwrapped_yaw = previous_yaw
            accumulated = 0.0
            dropped = False
            time_to_drop = MANIPULATION_HORIZON_S
            terminal_row: dict[str, Any] | None = None
            for step in range(horizon_steps):
                actions = controller.action(obs)
                obs, _, _, _ = controller.wrapper.step(actions)
                if capture.quat is None or capture.terminated is None or capture.truncated is None:
                    raise RuntimeError("Terminal-state capture did not execute")
                current = capture.quat[0].detach().cpu().numpy().astype(np.float64)
                increment = target_axis_increment(previous, current)
                accumulated += increment
                current_yaw = yaw_xyzw(current)
                yaw_delta = (current_yaw - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
                unwrapped_yaw += yaw_delta
                terminated = bool(capture.terminated[0].item())
                truncated = bool(capture.truncated[0].item())
                dropped_now = terminated
                if dropped_now:
                    dropped = True
                    time_to_drop = (step + 1) * float(core.step_dt)
                complete = dropped_now or step == horizon_steps - 1
                termination_reason = (
                    "task_terminated"
                    if terminated
                    else "task_truncated"
                    if truncated
                    else "fixed_horizon"
                    if step == horizon_steps - 1
                    else ""
                )
                row = {
                    "protocol_id": PROTOCOL_ID,
                    "pid": os.getpid(),
                    "friction": ftext(friction),
                    "seed": seed,
                    "control_step": step,
                    "sim_time_s": ftext((step + 1) * float(core.step_dt)),
                    "object_qx": ftext(float(current[0])),
                    "object_qy": ftext(float(current[1])),
                    "object_qz": ftext(float(current[2])),
                    "object_qw": ftext(float(current[3])),
                    "incremental_target_axis_rotation_rad": ftext(increment),
                    "accumulated_target_axis_rotation_rad": ftext(accumulated),
                    "yaw_unwrapped_rad": ftext(unwrapped_yaw),
                    "dropped": int(dropped_now),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "termination_reason": termination_reason,
                    "episode_complete": int(complete),
                    "time_to_drop_s": ftext(time_to_drop) if complete else "NaN",
                    "early_drop": int(dropped and time_to_drop < EARLY_DROP_S) if complete else "",
                    "survival": int(not dropped) if complete else "",
                    "episode_j_rot_radps": ftext(accumulated / MANIPULATION_HORIZON_S) if complete else "NaN",
                    "initial_state_sha256": fingerprint,
                }
                sink.write(row)
                if complete:
                    terminal_row = row
                previous = current
                previous_yaw = current_yaw
                if dropped_now:
                    break
            if terminal_row is None:
                raise RuntimeError(f"Episode {seed} lacks a terminal row")
            episodes.append(
                {
                    "seed": seed,
                    "initial_state_sha256": fingerprint,
                    "initial_state": initial_state,
                    "time_to_drop_s": float(terminal_row["time_to_drop_s"]),
                    "early_drop": bool(int(terminal_row["early_drop"])),
                    "survival": bool(int(terminal_row["survival"])),
                    "j_rot_radps": float(terminal_row["episode_j_rot_radps"]),
                    "termination_reason": terminal_row["termination_reason"],
                }
            )
    finally:
        if capture is not None:
            capture.restore()
        if sink is not None:
            sink.close()
        close_env(env)
    metrics = {
        "episodes": len(episodes),
        "median_time_to_drop_s": median(item["time_to_drop_s"] for item in episodes) if episodes else None,
        "early_drop_rate": sum(item["early_drop"] for item in episodes) / len(episodes) if episodes else None,
        "survival_rate": sum(item["survival"] for item in episodes) / len(episodes) if episodes else None,
        "median_j_rot_radps": median(item["j_rot_radps"] for item in episodes) if episodes else None,
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "worker_mode": "manipulation",
        "pid": os.getpid(),
        "friction": friction,
        "material_readback": material_checks[0] if material_checks else None,
        "material_checks": material_checks,
        "material_valid": material_valid and len(material_checks) == len(PAIRED_EPISODE_SEEDS),
        "mass": material_checks[0].get("mass_kg") if material_checks else None,
        "checkpoint_sha256": controller_status.get("checkpoint_sha256"),
        "checkpoint_load": controller_status,
        "physics_dt": dt,
        "decimation": decimation,
        "seed": list(PAIRED_EPISODE_SEEDS),
        "horizon_s": MANIPULATION_HORIZON_S,
        "horizon_steps": horizon_steps,
        "episodes": episodes,
        "fingerprints": fingerprints,
        "metrics": metrics,
        "manipulation_raw_csv": str(csv_path),
        "ppo_reward_used": False,
    }


def run_worker() -> int:  # worker写出单一condition artifact后由入口关闭SimulationApp。
    repo_root = Path(__file__).resolve().parents[1]
    output_path = ARGS.worker_output.resolve()
    friction = float(ARGS.friction)
    protected_before = protected_hashes(repo_root)
    fatal_error = None
    result: dict[str, Any] | None = None
    try:
        if not any(math.isclose(friction, item, abs_tol=1.0e-12) for item in ALL_FRICTIONS):
            raise ValueError(f"Unapproved friction condition: {friction}")
        if ARGS.worker_mode == "physical":
            result = run_physical_worker(repo_root, friction, output_path)
        elif ARGS.worker_mode == "manipulation":
            result = run_manipulation_worker(repo_root, friction, output_path)
        else:
            raise ValueError(f"Unknown worker mode: {ARGS.worker_mode}")
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    protected_after = protected_hashes(repo_root)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "qualification_eligible": True,
        "worker_mode": ARGS.worker_mode,
        "pid": os.getpid(),
        "friction": friction,
        "material_readback": result.get("material_readback") if result else None,
        "mass": result.get("mass") if result else None,
        "checkpoint_sha256": result.get("checkpoint_sha256") if result else None,
        "physics_dt": result.get("physics_dt") if result else None,
        "decimation": result.get("decimation") if result else None,
        "seed": result.get("seed") if result else None,
        "result": result,
        "fatal_error": fatal_error,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_unchanged": protected_before == protected_after,
        "simulation_app_close_contract": "ENTRYPOINT_FINALLY",
    }
    write_json(output_path, payload)
    print(f"P0_7F1_WORKER_OUTPUT={output_path}", flush=True)
    print(f"P0_7F1_WORKER_STATUS={'FAILED_TECHNICAL' if fatal_error else 'COLLECTED'}", flush=True)
    return 1 if fatal_error else 0


def launch_worker(
    script_path: Path,
    worker_mode: str,
    friction: float,
    output_path: Path,
    device: str | None,
) -> dict[str, Any]:  # subprocess.run保证前一个Kit完全退出后才启动下一个。
    command = [
        sys.executable,
        str(script_path),
        "--worker_mode",
        worker_mode,
        "--worker_output",
        str(output_path),
        "--friction",
        f"{friction:.2f}",
        "--viz",
        "none",
    ]
    if device is not None:
        command.extend(("--device", device))
    started = datetime.now().isoformat(timespec="microseconds")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    finished = datetime.now().isoformat(timespec="microseconds")
    log_path = output_path.with_suffix(".log")
    log_path.write_text(
        f"COMMAND={json.dumps(command)}\nSTARTED={started}\nFINISHED={finished}\nRETURN_CODE={completed.returncode}"
        f"\n--- STDOUT ---\n{completed.stdout}\n--- STDERR ---\n{completed.stderr}",
        encoding="utf-8",
    )
    if not output_path.is_file():
        return {
            "fatal_error": {
                "type": "WORKER_OUTPUT_MISSING",
                "message": f"return_code={completed.returncode}; see {log_path}",
            },
            "worker_return_code": completed.returncode,
            "worker_log": str(log_path),
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


def evaluate_physical(
    nominal: dict[str, Any], candidate: dict[str, Any], friction: float
) -> dict[str, Any]:  # 父进程按repeat离线计算candidate/nominal ratio。
    nominal_result = nominal.get("result") or {}
    candidate_result = candidate.get("result") or {}
    material_valid = bool(candidate_result.get("material_valid"))
    if not material_valid:
        return {
            "friction": friction,
            "material_status": "FAIL",
            "slip_status": "NOT_RUN",
            "classification": "MATERIAL_INVALID",
            "repeats": [],
        }
    nominal_repeats = nominal_result.get("slip_repeats", [])
    candidate_repeats = candidate_result.get("slip_repeats", [])
    if len(nominal_repeats) != REPEATS or len(candidate_repeats) != REPEATS:
        return {
            "friction": friction,
            "material_status": "PASS",
            "slip_status": "FAIL",
            "classification": "FRICTION_EFFECT_TOO_WEAK",
            "reason": "incomplete_onset_repeats",
            "repeats": [],
        }
    ratios: list[dict[str, Any]] = []
    for repeat in range(REPEATS):
        nominal_onset = nominal_repeats[repeat].get("onset_force_n")
        candidate_onset = candidate_repeats[repeat].get("onset_force_n")
        ratio = (
            float(candidate_onset) / float(nominal_onset)
            if nominal_onset is not None
            and candidate_onset is not None
            and float(nominal_onset) > 0.0
            else math.nan
        )
        passed = (
            math.isfinite(ratio)
            and float(candidate_onset) < float(nominal_onset)
            and ratio <= FRICTION_MAX_ONSET_RATIO
        )
        ratios.append(
            {
                "repeat": repeat,
                "nominal_onset_n": nominal_onset,
                "candidate_onset_n": candidate_onset,
                "ratio": ratio,
                "pass": passed,
            }
        )
    physical_pass = all(item["pass"] for item in ratios)
    return {
        "friction": friction,
        "material_status": "PASS",
        "slip_status": "PASS" if physical_pass else "FAIL",
        "classification": None if physical_pass else "FRICTION_EFFECT_TOO_WEAK",
        "repeats": ratios,
        "median_nominal_onset_n": median(float(item["nominal_onset_n"]) for item in ratios),
        "median_candidate_onset_n": median(float(item["candidate_onset_n"]) for item in ratios),
        "median_ratio": median(float(item["ratio"]) for item in ratios),
    }


def evaluate_manipulation(
    nominal: dict[str, Any], candidate: dict[str, Any], physical: dict[str, Any]
) -> dict[str, Any]:  # 使用分离进程artifact做24-seed配对与冻结floor判据。
    nominal_result = nominal.get("result") or {}
    candidate_result = candidate.get("result") or {}
    if not candidate_result.get("material_valid"):
        return {
            "status": "FAIL",
            "classification": "MATERIAL_INVALID",
            "reason": "manipulation_material_readback_invalid",
        }
    nominal_fingerprints = nominal_result.get("fingerprints", {})
    candidate_fingerprints = candidate_result.get("fingerprints", {})
    mismatches = {
        str(seed): {
            "nominal": nominal_fingerprints.get(str(seed)),
            "candidate": candidate_fingerprints.get(str(seed)),
        }
        for seed in PAIRED_EPISODE_SEEDS
        if nominal_fingerprints.get(str(seed)) != candidate_fingerprints.get(str(seed))
    }
    if mismatches:
        raise RuntimeError(f"Paired initial state mismatch: {mismatches}")
    nominal_metrics = nominal_result.get("metrics", {})
    candidate_metrics = candidate_result.get("metrics", {})
    if nominal_metrics.get("episodes") != 24 or candidate_metrics.get("episodes") != 24:
        raise RuntimeError(
            f"Manipulation episode count mismatch: nominal={nominal_metrics.get('episodes')}, "
            f"candidate={candidate_metrics.get('episodes')}"
        )
    time_to_drop_ratio = (
        float(candidate_metrics["median_time_to_drop_s"]) / max(float(nominal_metrics["median_time_to_drop_s"]), 1.0e-12)
    )
    c1 = time_to_drop_ratio >= 0.50
    c2 = float(candidate_metrics["early_drop_rate"]) <= 0.50
    c3 = float(candidate_metrics["median_j_rot_radps"]) > 0.0
    passed = c1 and c2 and c3
    return {
        "status": "PASS" if passed else "FAIL",
        "classification": "QUALIFIED" if passed else "FRICTION_CAUSES_GRASP_COLLAPSE",
        "friction": physical["friction"],
        "paired_seed_count": 24,
        "initial_state_pairing": "PASS",
        "nominal_metrics": nominal_metrics,
        "candidate_metrics": candidate_metrics,
        "criteria": {
            "median_time_to_drop_ratio_at_least_0_50": c1,
            "early_drop_rate_at_most_0_50": c2,
            "positive_median_j_rot": c3,
        },
        "time_to_drop_ratio": time_to_drop_ratio,
        "ppo_reward_used": False,
    }


def write_checksums(output_dir: Path, checksum_path: Path) -> dict[str, str]:  # 汇总全部已落盘artifact。
    artifacts = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.resolve() != checksum_path.resolve()
    )
    checksums = {path.relative_to(output_dir).as_posix(): sha256_file(path) for path in artifacts}
    with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
        for relative, digest in checksums.items():
            stream.write(f"{digest}  {relative}\n")
    return checksums


def orchestrate() -> int:  # 父进程仅串行编排和离线判定，不创建SimulationApp。
    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_output = PRE_ARGS.output_dir if PRE_ARGS.output_dir.is_absolute() else repo_root / PRE_ARGS.output_dir
    output_dir = (base_output / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    for friction in ALL_FRICTIONS:
        (output_dir / friction_tag(friction)).mkdir()
    protected_before = protected_hashes(repo_root)
    checkpoint_path = repo_root / REFERENCE_CHECKPOINT
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    fatal_error = None
    physical_workers: dict[str, dict[str, Any]] = {}
    manipulation_workers: dict[str, dict[str, Any]] = {}
    physical_results: dict[str, dict[str, Any]] = {}
    manipulation_results: dict[str, dict[str, Any]] = {}
    final_candidates: list[dict[str, Any]] = []
    final_conclusion = "FRICTION_PRIMARY_FACTOR_NOT_QUALIFIED"
    stop_rule = None
    try:
        if checkpoint_sha != REFERENCE_CHECKPOINT_SHA256:
            raise RuntimeError(
                f"Checkpoint SHA mismatch: actual={checkpoint_sha}, expected={REFERENCE_CHECKPOINT_SHA256}"
            )
        if validate_j_rot_semantics()["status"] != "PASS":
            raise RuntimeError("J_rot metric semantics validation failed")
        for friction in ALL_FRICTIONS:
            tag = friction_tag(friction)
            output_path = output_dir / tag / "physical_worker.json"
            worker = launch_worker(script_path, "physical", friction, output_path, PRE_ARGS.device)
            physical_workers[tag] = worker
            if worker.get("fatal_error") or worker.get("worker_return_code") != 0:
                raise RuntimeError(f"Physical worker {friction:.2f} failed: {worker.get('fatal_error')}")
            if not worker.get("protected_unchanged"):
                raise RuntimeError(f"Physical worker {friction:.2f} changed a protected file")
        nominal_physical = physical_workers[friction_tag(FRICTION_NOMINAL)]
        nominal_result = nominal_physical.get("result") or {}
        if not nominal_result.get("material_valid") or len(nominal_result.get("slip_repeats", [])) != REPEATS:
            raise RuntimeError("Nominal physical reference is invalid or incomplete")
        for friction in FRICTION_CANDIDATES:
            tag = friction_tag(friction)
            physical_results[tag] = evaluate_physical(nominal_physical, physical_workers[tag], friction)

        phase2_candidates = [
            friction
            for friction in FRICTION_CANDIDATES
            if physical_results[friction_tag(friction)]["slip_status"] == "PASS"
        ]
        if phase2_candidates:
            nominal_output = output_dir / friction_tag(FRICTION_NOMINAL) / "manipulation_worker.json"
            nominal_manipulation = launch_worker(
                script_path, "manipulation", FRICTION_NOMINAL, nominal_output, PRE_ARGS.device
            )
            manipulation_workers[friction_tag(FRICTION_NOMINAL)] = nominal_manipulation
            if nominal_manipulation.get("fatal_error") or nominal_manipulation.get("worker_return_code") != 0:
                raise RuntimeError(f"Nominal manipulation worker failed: {nominal_manipulation.get('fatal_error')}")
            if not nominal_manipulation.get("protected_unchanged"):
                raise RuntimeError("Nominal manipulation worker changed a protected file")
            for friction in phase2_candidates:
                tag = friction_tag(friction)
                output_path = output_dir / tag / "manipulation_worker.json"
                worker = launch_worker(script_path, "manipulation", friction, output_path, PRE_ARGS.device)
                manipulation_workers[tag] = worker
                if worker.get("fatal_error") or worker.get("worker_return_code") != 0:
                    raise RuntimeError(f"Manipulation worker {friction:.2f} failed: {worker.get('fatal_error')}")
                if not worker.get("protected_unchanged"):
                    raise RuntimeError(f"Manipulation worker {friction:.2f} changed a protected file")
                manipulation_results[tag] = evaluate_manipulation(
                    nominal_manipulation, worker, physical_results[tag]
                )

        for friction in FRICTION_CANDIDATES:
            tag = friction_tag(friction)
            physical = physical_results[tag]
            if physical["classification"] is not None:
                classification = physical["classification"]
            else:
                classification = manipulation_results[tag]["classification"]
            if classification not in SCIENTIFIC_CLASSES:
                raise RuntimeError(f"Unapproved scientific classification: {classification}")
            final_candidates.append(
                {
                    "friction": friction,
                    "physical": physical,
                    "manipulation": manipulation_results.get(tag),
                    "classification": classification,
                }
            )
        qualified = [item["friction"] for item in final_candidates if item["classification"] == "QUALIFIED"]
        if qualified:
            final_conclusion = "FRICTION_PRIMARY_FACTOR_QUALIFIED"
        else:
            final_conclusion = "FRICTION_PRIMARY_FACTOR_NOT_QUALIFIED"
            stop_rule = "FRICTION_FACTOR_NOT_SUITABLE_AS_PRIMARY_FACTOR"
    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}

    protected_after = protected_hashes(repo_root)
    all_pids = [
        worker.get("pid")
        for worker in [*physical_workers.values(), *manipulation_workers.values()]
        if worker.get("pid") is not None
    ]
    process_isolation = {
        "worker_count": len(all_pids),
        "unique_pid_count": len(set(all_pids)),
        "all_worker_pids_unique": len(all_pids) == len(set(all_pids)),
        "launch_policy": "strictly_sequential_subprocess_run",
        "simulation_app_lifecycle": "one_worker_one_SimulationApp_close",
    }
    if protected_before != protected_after and fatal_error is None:
        fatal_error = {"type": "PROTECTED_FILE_CHANGE", "message": "Protected hashes changed during P0-7F1"}
    overall_status = "P0-7F1 COMPLETE" if fatal_error is None else "P0-7F1 FAILED_TECHNICAL"
    summary = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "task_id": TASK_ID,
        "factor_schema": SCHEMA_ID,
        "overall_status": overall_status,
        "fatal_error": fatal_error,
        "final_conclusion": final_conclusion if fatal_error is None else None,
        "stop_rule": stop_rule if fatal_error is None else None,
        "automatic_candidate_selection": False,
        "qualified_candidates": [
            item["friction"] for item in final_candidates if item["classification"] == "QUALIFIED"
        ],
        "candidate_results": final_candidates,
        "nominal_physical_reference": (physical_workers.get(friction_tag(FRICTION_NOMINAL), {}).get("result")),
        "physical_workers": physical_workers,
        "manipulation_workers": manipulation_workers,
        "process_isolation": process_isolation,
        "frozen_protocol": {
            "nominal": FRICTION_NOMINAL,
            "candidates": list(FRICTION_CANDIDATES),
            "ratio_threshold": FRICTION_MAX_ONSET_RATIO,
            "paired_seeds": list(PAIRED_EPISODE_SEEDS),
            "manipulation_horizon_s": MANIPULATION_HORIZON_S,
            "ppo_reward_used": False,
            "training_performed": False,
            "B0_O0_entered": False,
        },
        "reference_checkpoint": {
            "path": str(REFERENCE_CHECKPOINT),
            "expected_sha256": REFERENCE_CHECKPOINT_SHA256,
            "actual_sha256": checkpoint_sha,
        },
        "j_rot_semantics": validate_j_rot_semantics(),
        "git_commit": git_text(repo_root, "rev-parse", "HEAD"),
        "git_branch": git_text(repo_root, "branch", "--show-current"),
        "script_sha256": sha256_file(script_path),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_unchanged": protected_before == protected_after,
    }
    summary_path = output_dir / "p0_friction_process_isolated_summary.json"
    write_json(summary_path, summary)
    checksum_path = output_dir / "checksums.sha256"
    write_checksums(output_dir, checksum_path)
    print(f"P0_7F1_OUTPUT_DIR={output_dir}", flush=True)
    print(f"P0_7F1_PROCESS_COUNT={process_isolation['worker_count']}", flush=True)
    print(f"P0_7F1_FINAL_CONCLUSION={summary['final_conclusion']}", flush=True)
    print(f"P0_7F1_STATUS={overall_status}", flush=True)
    if fatal_error is not None:
        print(f"P0_7F1_FATAL={fatal_error['type']}: {fatal_error['message']}", flush=True)
    return 1 if fatal_error is not None else 0


if __name__ == "__main__":
    if PRE_ARGS.worker_mode == "none":
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
