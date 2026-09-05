"""P0-7F friction cross-environment coupling diagnosis; no training or schema writes."""

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
from typing import Any

from isaaclab.app import AppLauncher


PROTOCOL_ID = "P0_7F_FRICTION_CROSS_ENVIRONMENT_COUPLING_DIAGNOSIS"  # diagnosis-only协议。
PARENT_QUALIFICATION_RUN_ID = "20260902_205231_823279"  # 第五次same-spec debug。
TASK_ID = "Isaac-Reorient-Cube-Leap-CRI"  # 同一CRI任务。
SCHEMA_ID = "CRI_FACTOR_SCHEMA_V1"  # 只读正式schema。
FACTOR_CODES = ("000", "100", "010", "001")  # 父4-env架构。
NUM_ENVS = 4
BASE_SEED = 42
REPEATS = 3
FRICTION_ENVS = (0, 2)
FRICTIONS = (0.80, 0.45, 0.40, 0.35, 0.30)
FORCE_MIN_N = 0.0
FORCE_MAX_N = 2.20
FORCE_STEP_N = 0.01
HOLD_STEPS = 8
SETTLE_STEPS = 120
VELOCITY_THRESHOLD_MPS = 0.01
DISPLACEMENT_THRESHOLD_M = 0.0005
ONSET_CONSECUTIVE_STEPS = 5
CLEARANCE_M = 0.001
ISOLATION_HEIGHT_M = 5.0
PROBE_ROOT_PATH = "/World/ground"
PROBE_TEMP_MATERIAL_PATH = "/World/P0_7F_audit_ground_material"
PROBE_FRICTION = 0.80
PROBE_COMBINE_MODE = "min"

PROTECTED_PATHS = (
    "scripts/p0_physics_validation.py",
    "scripts/p0_timing_audit.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/__init__.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/assets/leap.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/factor_conditions.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/leap_hand_cri_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/reorientation_cri_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml",
    "logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth",
)


def build_parser() -> argparse.ArgumentParser:  # 每次调用只运行一个diagnostic mode。
    parser = argparse.ArgumentParser(description="Run P0-7F friction coupling diagnosis.")
    parser.add_argument("--diagnostic_mode", choices=("multi_env", "isolated"), required=True)
    parser.add_argument("--friction", type=float, default=None)
    parser.add_argument("--manual_view", action="store_true")  # 仅保持GUI供人工检查，不产生qualification证据。
    parser.add_argument("--output_dir", type=Path, default=Path("logs/p0_friction_coupling_diagnosis"))
    parser.add_argument("--static_only", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS = build_parser().parse_args()


def validate_constants() -> None:  # Kit前拒绝冻结协议漂移。
    assert FACTOR_CODES == ("000", "100", "010", "001")
    assert FRICTION_ENVS == (0, 2)
    assert FRICTIONS == (0.80, 0.45, 0.40, 0.35, 0.30)
    assert (FORCE_MIN_N, FORCE_MAX_N, FORCE_STEP_N) == (0.0, 2.20, 0.01)
    assert HOLD_STEPS == 8 and SETTLE_STEPS == 120 and ONSET_CONSECUTIVE_STEPS == 5
    assert VELOCITY_THRESHOLD_MPS == 0.01 and DISPLACEMENT_THRESHOLD_M == 0.0005
    if ARGS.diagnostic_mode == "isolated" and ARGS.friction not in FRICTIONS:
        raise ValueError(f"Isolated friction must be frozen value, got {ARGS.friction}")
    if ARGS.manual_view and ARGS.diagnostic_mode != "multi_env":
        raise ValueError("--manual_view is only valid with --diagnostic_mode multi_env")
    if ARGS.manual_view and ARGS.friction is not None:
        raise ValueError("--manual_view does not accept --friction")
    if ARGS.manual_view and getattr(ARGS, "headless", False) and not ARGS.static_only:
        raise ValueError("--manual_view requires the Isaac Sim GUI; omit --headless")


validate_constants()
if ARGS.static_only:
    print("P0_7F_STATIC_CONTRACT_PASS")
    raise SystemExit(0)

# AppLauncher defaults to headless when no Kit visualizer is selected.  Manual
# inspection is the sole exception: request the local Kit GUI before app launch.
if ARGS.manual_view:
    ARGS.headless = False
    ARGS.visualizer = "kit"
    # Manual inspection is visualization-only.  On the 8 GiB RTX 5060 the
    # default balanced RTPT path consumed ~7.7 GiB at 100% utilization for this
    # four-environment scene.  Use Isaac Lab's official viewport-only
    # performance preset; qualification runs retain their original settings.
    ARGS.rendering_mode = "performance"


APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

import LEAP_Isaaclab.tasks  # noqa: E402,F401
import isaaclab  # noqa: E402
from isaaclab.sim.utils import get_current_stage, get_first_matching_child_prim  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg  # noqa: E402
from isaacsim.core.version import get_version as get_isaac_sim_version  # noqa: E402


def sha256_file(path: Path) -> str:  # 流式artifact哈希。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(repo_root: Path, *args: str) -> str:  # 不修改global safe.directory。
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root.as_posix()}", *args], cwd=repo_root, stderr=subprocess.STDOUT
    ).decode("utf-8", errors="replace").strip()


def protected_hashes(repo_root: Path) -> dict[str, str | None]:  # 保护文件逐项哈希。
    return {path: sha256_file(repo_root / path) if (repo_root / path).is_file() else None for path in PROTECTED_PATHS}


def ftext(value: float) -> str:  # CSV稳定浮点表示。
    return "NaN" if not math.isfinite(value) else f"{value:.12g}"


def json_safe(value: Any) -> Any:  # 非有限值写为JSON null。
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:  # create-only JSON。
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def create_env() -> gym.Env:  # 同一冻结task与4-env assignment。
    cfg = parse_env_cfg(TASK_ID, device=ARGS.device, num_envs=NUM_ENVS)
    cfg.factor_codes = FACTOR_CODES
    cfg.events = None
    cfg.enable_adr = False
    env = gym.make(TASK_ID, cfg=cfg)
    if tuple(env.unwrapped.factor_code_labels) != FACTOR_CODES:
        env.close()
        raise RuntimeError("Explicit factor assignment mismatch")
    return env


def reset_env(env: gym.Env) -> None:  # 严格复用父协议reset边界。
    torch.manual_seed(BASE_SEED)
    env.reset(seed=BASE_SEED)


def direct_step(core: Any, dt: float) -> None:  # write→physics step→state update。
    core.scene.write_data_to_sim()
    core.sim.step(render=False)
    core.scene.update(dt)


def set_object_states(core: Any, measured_envs: tuple[int, ...]) -> None:  # 非测量cube隔离到高处。
    all_ids = torch.arange(NUM_ENVS, dtype=torch.long, device=core.device)
    pose = core.object.data.default_root_pose.torch.clone()
    pose[:, :3] = core.scene.env_origins.clone()
    pose[:, 2] += ISOLATION_HEIGHT_M
    pose[:, 3:7] = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=pose.dtype, device=core.device)
    half_height = cube_half_height(core)
    for env_id in measured_envs:
        pose[env_id, :2] = core.scene.env_origins[env_id, :2]
        pose[env_id, 2] = half_height + CLEARANCE_M
    velocity = torch.zeros((NUM_ENVS, 6), dtype=pose.dtype, device=core.device)
    core.object.write_root_pose_to_sim_index(root_pose=pose, env_ids=all_ids)
    core.object.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=all_ids)


def cube_half_height(core: Any) -> float:  # 实际USD bound决定接触初始高度。
    prim = get_current_stage().GetPrimAtPath(f"{core.scene.env_prim_paths[0]}/object")
    if not prim.IsValid():
        raise RuntimeError("Cube prim missing")
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeLocalBound(prim).ComputeAlignedBox()
    return 0.5 * float(box.GetMax()[2] - box.GetMin()[2])


def set_wrench(core: Any, env_ids: tuple[int, ...], force_x: float) -> None:  # 仅选定cube施加world-x力。
    ids = torch.tensor(env_ids, dtype=torch.long, device=core.device)
    forces = torch.tensor((force_x, 0.0, 0.0), dtype=torch.float32, device=core.device).reshape(1, 1, 3).repeat(len(env_ids), 1, 1)
    core.object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces, torques=torch.zeros_like(forces), env_ids=ids, is_global=True
    )


def write_friction(core: Any, env_id: int, friction: float) -> dict[str, Any]:  # qualification override只作用指定env。
    ids = torch.tensor((env_id,), dtype=torch.long, device=core.device)
    values = torch.tensor((friction,), dtype=torch.float32, device=core.device)
    core._write_object_material(ids, values, values)
    actual = core._read_materials(core.object, "object")
    return {"static": float(actual[env_id, 0, 0]), "dynamic": float(actual[env_id, 0, 1]),
            "combine_mode": core._read_combine_mode(env_id, "object", "cri_object_physics_material")}


class GroundBinding:  # audit-only ground material，退出时严格恢复。
    def __init__(self):
        self.stage = get_current_stage()
        self.binding_api: Any = None
        self.prior_path: Any = None
        self.prior_strength: Any = None
        self.metadata: dict[str, Any] = {"probe_path": PROBE_ROOT_PATH, "restored": False}

    def __enter__(self) -> "GroundBinding":
        collider = get_first_matching_child_prim(
            PROBE_ROOT_PATH, predicate=lambda prim: prim.GetTypeName() == "Plane", stage=self.stage
        )
        if collider is None or not collider.IsValid():
            raise RuntimeError("Ground Plane collider missing")
        self.binding_api = UsdShade.MaterialBindingAPI(collider)
        prior = self.binding_api.GetDirectBinding("physics")
        self.prior_path = prior.GetMaterialPath()
        self.prior_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(prior.GetBindingRel())
        if not self.prior_path or self.stage.GetPrimAtPath(PROBE_TEMP_MATERIAL_PATH).IsValid():
            raise RuntimeError("Ground material is not safely replaceable")
        cfg = PhysxRigidBodyMaterialCfg(static_friction=PROBE_FRICTION, dynamic_friction=PROBE_FRICTION,
                                        restitution=0.0, friction_combine_mode=PROBE_COMBINE_MODE)
        temp_prim = cfg.func(PROBE_TEMP_MATERIAL_PATH, cfg)
        self.binding_api.Bind(UsdShade.Material(temp_prim), bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                              materialPurpose="physics")
        self.metadata.update({"collider_path": str(collider.GetPath()), "temporary_material_path": PROBE_TEMP_MATERIAL_PATH,
                              "prior_material_path": str(self.prior_path), "static_friction": float(UsdPhysics.MaterialAPI(temp_prim).GetStaticFrictionAttr().Get()),
                              "dynamic_friction": float(UsdPhysics.MaterialAPI(temp_prim).GetDynamicFrictionAttr().Get()),
                              "combine_mode": str(PhysxSchema.PhysxMaterialAPI(temp_prim).GetFrictionCombineModeAttr().Get())})
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.binding_api.UnbindDirectBinding("physics")
        prior_prim = self.stage.GetPrimAtPath(self.prior_path)
        self.binding_api.Bind(UsdShade.Material(prior_prim), bindingStrength=self.prior_strength,
                              materialPurpose="physics")
        if self.stage.GetPrimAtPath(PROBE_TEMP_MATERIAL_PATH).IsValid():
            self.stage.RemovePrim(PROBE_TEMP_MATERIAL_PATH)
        restored = self.binding_api.GetDirectBinding("physics")
        self.metadata["restored"] = restored.GetMaterialPath() == self.prior_path
        self.metadata["temporary_material_removed"] = not self.stage.GetPrimAtPath(PROBE_TEMP_MATERIAL_PATH).IsValid()


def prim_identity(prim: Any) -> dict[str, Any]:  # USD instance/prototype与prim-stack身份。
    if prim is None or not prim.IsValid():
        return {"valid": False}
    prototype = prim.GetPrototype() if prim.IsInstance() else None
    stack = [{"layer": spec.layer.identifier, "path": str(spec.path)} for spec in prim.GetPrimStack()]
    return {"valid": True, "path": str(prim.GetPath()), "type": prim.GetTypeName(), "python_object_id": id(prim),
            "is_instance": prim.IsInstance(), "is_instance_proxy": prim.IsInstanceProxy(),
            "prototype_path": str(prototype.GetPath()) if prototype is not None and prototype.IsValid() else None,
            "prim_stack": stack}


def binding_identity(prim: Any) -> dict[str, Any]:  # direct与computed physics binding均记录。
    api = UsdShade.MaterialBindingAPI(prim)
    direct = api.GetDirectBinding("physics")
    try:
        material, relation = api.ComputeBoundMaterial("physics")
        computed_path = str(material.GetPath()) if material and material.GetPrim().IsValid() else None
        relation_path = str(relation.GetPath()) if relation else None
    except Exception as exc:
        computed_path = None
        relation_path = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
    rel = direct.GetBindingRel()
    return {"prim_path": str(prim.GetPath()), "direct_material_path": str(direct.GetMaterialPath()),
            "direct_relation_path": str(rel.GetPath()),
            "binding_strength": str(UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(rel)),
            "computed_material_path": computed_path, "computed_relation_path": relation_path}


def material_snapshot(core: Any, moment: str) -> dict[str, Any]:  # env0/env2只读USD/PhysX审计。
    stage = get_current_stage()
    materials = core._read_materials(core.object, "object")
    result: dict[str, Any] = {"moment": moment, "root_view_python_object_id": id(core.object.root_view),
                              "physx_material_handle_identity": "API_NOT_EXPOSED", "envs": {}}
    for env_id in FRICTION_ENVS:
        object_path = f"{core.scene.env_prim_paths[env_id]}/object"
        material_path = f"{object_path}/cri_object_physics_material"
        object_prim = stage.GetPrimAtPath(object_path)
        material_prim = stage.GetPrimAtPath(material_path)
        colliders = [prim for prim in Usd.PrimRange(object_prim) if prim.HasAPI(UsdPhysics.CollisionAPI)]
        result["envs"][str(env_id)] = {
            "environment_id": env_id, "factor_code": FACTOR_CODES[env_id], "object_prim": prim_identity(object_prim),
            "collider_prim_paths": [str(prim.GetPath()) for prim in colliders], "physics_material_prim": prim_identity(material_prim),
            "material_usdshade_python_object_id": id(UsdShade.Material(material_prim)),
            "material_bindings": [binding_identity(object_prim)] + [binding_identity(prim) for prim in colliders],
            "static_friction_readback": float(materials[env_id, 0, 0]),
            "dynamic_friction_readback": float(materials[env_id, 0, 1]),
            "friction_combine_mode": core._read_combine_mode(env_id, "object", "cri_object_physics_material"),
            "material_api_target": "RigidObject.root_view.set_material_properties(env_indices)",
        }
    env0 = result["envs"]["0"]
    env2 = result["envs"]["2"]
    result["usd_material_paths_independent"] = env0["physics_material_prim"]["path"] != env2["physics_material_prim"]["path"]
    result["computed_binding_paths_independent"] = (
        {item["computed_material_path"] for item in env0["material_bindings"] if item["computed_material_path"]}
        != {item["computed_material_path"] for item in env2["material_bindings"] if item["computed_material_path"]}
    )
    result["shared_exposed_material_path"] = not result["usd_material_paths_independent"]
    return result


RAW_FIELDS = ("measurement_mode", "friction", "repeat", "condition", "env_id", "factor_code", "physics_step",
              "force_x_n", "x_m", "vx_mps", "displacement_m", "onset_flag", "onset_force_n",
              "static_friction_readback", "dynamic_friction_readback", "combine_mode", "process_id")


def measure(core: Any, env: gym.Env, measured: tuple[tuple[str, int, float], ...], writer: csv.DictWriter, mode: str) -> list[dict[str, Any]]:
    dt = float(core.cfg.sim.dt)
    levels = np.round(np.arange(FORCE_MIN_N, FORCE_MAX_N + 0.5 * FORCE_STEP_N, FORCE_STEP_N), 10)
    results = []
    for repeat in range(REPEATS):
        reset_env(env)
        core.object.permanent_wrench_composer.reset()
        readbacks = {env_id: write_friction(core, env_id, friction) for _, env_id, friction in measured}
        set_object_states(core, tuple(env_id for _, env_id, _ in measured))
        set_wrench(core, tuple(env_id for _, env_id, _ in measured), 0.0)
        for _ in range(SETTLE_STEPS):
            direct_step(core, dt)
        x0 = core.object.data.root_pos_w.torch[:, 0].clone()
        onset = {env_id: math.nan for _, env_id, _ in measured}
        found = {env_id: False for _, env_id, _ in measured}
        start_step = int(core.sim.get_physics_step_count())
        for force_x in levels:
            consecutive = {env_id: 0 for _, env_id, _ in measured}
            set_wrench(core, tuple(env_id for _, env_id, _ in measured), float(force_x))
            for _ in range(HOLD_STEPS):
                direct_step(core, dt)
                pos = core.object.data.root_pos_w.torch
                vel = core.object.data.root_lin_vel_w.torch
                for condition, env_id, friction in measured:
                    displacement = abs(float(pos[env_id, 0] - x0[env_id]))
                    active = abs(float(vel[env_id, 0])) >= VELOCITY_THRESHOLD_MPS and displacement >= DISPLACEMENT_THRESHOLD_M
                    if not found[env_id] and active:
                        consecutive[env_id] += 1
                    elif not found[env_id]:
                        consecutive[env_id] = 0
                    onset_flag = False
                    if not found[env_id] and consecutive[env_id] >= ONSET_CONSECUTIVE_STEPS:
                        found[env_id] = True
                        onset[env_id] = float(force_x)
                        onset_flag = True
                    writer.writerow({"measurement_mode": mode, "friction": ftext(friction), "repeat": repeat,
                                     "condition": condition, "env_id": env_id, "factor_code": FACTOR_CODES[env_id],
                                     "physics_step": int(core.sim.get_physics_step_count()) - start_step,
                                     "force_x_n": ftext(float(force_x)), "x_m": ftext(float(pos[env_id, 0])),
                                     "vx_mps": ftext(float(vel[env_id, 0])), "displacement_m": ftext(displacement),
                                     "onset_flag": int(onset_flag), "onset_force_n": ftext(onset[env_id]),
                                     "static_friction_readback": ftext(readbacks[env_id]["static"]),
                                     "dynamic_friction_readback": ftext(readbacks[env_id]["dynamic"]),
                                     "combine_mode": readbacks[env_id]["combine_mode"], "process_id": os.getpid()})
        core.object.permanent_wrench_composer.reset()
        results.append({"repeat": repeat, "conditions": {condition: {"env_id": env_id, "friction": friction,
                        "onset_n": onset[env_id], "material": readbacks[env_id]} for condition, env_id, friction in measured}})
    return results


def runtime_versions() -> dict[str, Any]:  # 精确记录仿真栈。
    return {"isaac_sim": get_isaac_sim_version(), "isaaclab_package": getattr(isaaclab, "__version__", None),
            "isaaclab_file": str(Path(isaaclab.__file__).resolve())}


def manual_view_snapshot(core: Any) -> list[dict[str, Any]]:  # 只读打印四环境物理状态，禁止用于科研判定。
    env_ids = torch.arange(NUM_ENVS, dtype=torch.long, device=core.device)
    core._read_back_factor_values(env_ids)
    snapshot = core.get_factor_snapshot()
    object_pos = core.object.data.root_pos_w.torch.detach().cpu()
    object_quat = core.object.data.root_quat_w.torch.detach().cpu()
    origins = core.scene.env_origins.detach().cpu()
    rows = []
    for env_id, factor_code in enumerate(FACTOR_CODES):
        effort = snapshot["actual_effort_limit"][env_id].detach().cpu().tolist()
        rows.append({
            "usage": "MANUAL_VISUAL_INSPECTION_ONLY",
            "qualification_eligible": False,
            "env_id": env_id,
            "factor_code": factor_code,
            "object_static_friction_readback": float(snapshot["actual_object_static_friction"][env_id]),
            "object_dynamic_friction_readback": float(snapshot["actual_object_dynamic_friction"][env_id]),
            "object_mass_kg": snapshot["actual_object_mass"][env_id].detach().cpu().tolist(),
            "motor_effort_limit_nm": effort,
            "env_origin_xyz_m": origins[env_id].tolist(),
            "object_world_position_xyz_m": object_pos[env_id].tolist(),
            "object_world_quaternion_wxyz": object_quat[env_id].tolist(),
        })
    return rows


def run_manual_view() -> int:  # 保持原始4-env viewport，直到窗口关闭或用户中断。
    print("MANUAL_VIEW_START", flush=True)
    print(f"simulation_app.is_running = {SIMULATION_APP.is_running()}", flush=True)
    print(f"headless = {ARGS.headless}", flush=True)
    print(f"device = {ARGS.device}", flush=True)
    print(f"renderer = {getattr(ARGS, 'rendering_mode', None)}", flush=True)
    # Keep the GUI visible while removing the RTPT/DLSS viewport path that
    # exhausts the 8 GiB GPU for the four-environment manual inspection scene.
    # Mode 2 is Kit's textured-diffuse MinimalRendering mode: geometry remains
    # visible, but it has no RTPT, reflections, shadows, AO, or DLSS pass.
    import carb

    render_settings = carb.settings.get_settings()
    render_settings.set_string("/rtx/rendermode", "MinimalRendering")
    render_settings.set_int("/rtx/minimal/mode", 2)
    SIMULATION_APP.update()
    # AppLauncher applies its persisted rendering preset during this first
    # update, so AA must be overridden afterwards for the manual viewport.
    render_settings.set_int("/rtx/post/aa/op", 0)
    print(f"manual_view_rtx_mode = {render_settings.get('/rtx/rendermode')}", flush=True)
    print(f"manual_view_minimal_mode = {render_settings.get('/rtx/minimal/mode')}", flush=True)
    print(f"manual_view_anti_aliasing = {render_settings.get('/rtx/post/aa/op')}", flush=True)
    env = create_env()
    core = env.unwrapped
    reset_env(env)
    print("MANUAL_VISUAL_INSPECTION_ONLY", flush=True)
    print("QUALIFICATION_ELIGIBLE=false", flush=True)
    print(json.dumps(manual_view_snapshot(core), ensure_ascii=False, indent=2), flush=True)
    print("MANUAL_VIEW_RUNNING=close the Isaac Sim window or press Ctrl+C to stop", flush=True)
    try:
        loop_started = False
        while SIMULATION_APP.is_running():
            if not loop_started:
                print("MANUAL_VIEW_LOOP_RUNNING", flush=True)
                loop_started = True
            core.scene.write_data_to_sim()
            core.sim.step(render=True)
            core.scene.update(float(core.cfg.sim.dt))
            SIMULATION_APP.update()
    except KeyboardInterrupt:
        print("MANUAL_VIEW_STOPPED_BY_USER", flush=True)
    return 0


def main() -> int:  # 单进程只执行一个diagnostic mode。
    if ARGS.manual_view:
        return run_manual_view()
    repo_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (repo_root / ARGS.output_dir / run_id).resolve() if not ARGS.output_dir.is_absolute() else (ARGS.output_dir / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "friction_raw.csv"
    material_path = output_dir / "material_isolation_audit.json"
    summary_path = output_dir / "p0_7f_summary.json"
    checksum_path = output_dir / "checksums.sha256"
    script_path = Path(__file__).resolve()
    git_before = git_text(repo_root, "rev-parse", "HEAD")
    protected_before = protected_hashes(repo_root)
    script_hash = sha256_file(script_path)
    env: gym.Env | None = None
    error = None
    results: dict[str, Any] = {}
    material_audit: list[dict[str, Any]] = []
    probe_metadata: dict[str, Any] = {}
    try:
        env = create_env()
        core = env.unwrapped
        dt = float(core.cfg.sim.dt)
        material_audit.append(material_snapshot(core, "A_TASK_CREATION"))
        reset_env(env)
        material_audit.append(material_snapshot(core, "B_PARENT_FACTOR_ASSIGNMENT"))
        with raw_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
            writer.writeheader()
            binding = GroundBinding()
            with binding:
                if ARGS.diagnostic_mode == "multi_env":
                    conditions = []
                    for friction in FRICTIONS[1:]:
                        reset_env(env)
                        write_friction(core, FRICTION_ENVS[0], 0.80)
                        write_friction(core, FRICTION_ENVS[1], friction)
                        material_audit.append(material_snapshot(core, f"AFTER_CANDIDATE_{friction:.2f}"))
                        measured = (("nominal", FRICTION_ENVS[0], 0.80), ("candidate", FRICTION_ENVS[1], friction))
                        conditions.append({"env2_friction": friction,
                                           "repeats": measure(core, env, measured, writer, "MULTI_ENV_SIMULTANEOUS")})
                    results = {"conditions": conditions}
                else:
                    assert ARGS.friction is not None
                    env_id = FRICTION_ENVS[0] if math.isclose(ARGS.friction, 0.80) else FRICTION_ENVS[1]
                    reset_env(env)
                    write_friction(core, env_id, ARGS.friction)
                    material_audit.append(material_snapshot(core, f"ISOLATED_{ARGS.friction:.2f}"))
                    measured = (("isolated", env_id, ARGS.friction),)
                    results = {"friction": ARGS.friction,
                               "repeats": measure(core, env, measured, writer, "PROCESS_ISOLATED_SINGLE_CONDITION")}
            probe_metadata = binding.metadata
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if env is not None:
            try:
                env.unwrapped.object.permanent_wrench_composer.reset()
            except Exception:
                pass
            env.close()
        protected_after = protected_hashes(repo_root)
        git_after = git_text(repo_root, "rev-parse", "HEAD")
        material_shared = any(item.get("shared_exposed_material_path") for item in material_audit)
        summary = {"protocol_id": PROTOCOL_ID, "run_id": run_id, "parent_qualification_run_id": PARENT_QUALIFICATION_RUN_ID,
                   "measurement_mode": "MULTI_ENV_SIMULTANEOUS" if ARGS.diagnostic_mode == "multi_env" else "PROCESS_ISOLATED_SINGLE_CONDITION",
                   "process_id": os.getpid(), "status": "FAILED_TECHNICAL" if error else "DATA_COLLECTED",
                   "error": error, "git_branch": git_text(repo_root, "branch", "--show-current"),
                   "git_commit": git_before, "git_commit_after": git_after, "script_sha256": script_hash,
                   "script_sha256_after": sha256_file(script_path), "protected_hashes_before": protected_before,
                   "protected_hashes_after": protected_after, "protected_unchanged": protected_before == protected_after,
                   "task_id": TASK_ID, "schema": SCHEMA_ID, "num_envs": NUM_ENVS, "factor_assignment": list(FACTOR_CODES),
                   "physics_dt": float(env.unwrapped.cfg.sim.dt) if env is not None else None,
                   "decimation": int(env.unwrapped.cfg.decimation) if env is not None else None,
                   "protocol_constants": {"frictions": list(FRICTIONS), "force_grid_n": [FORCE_MIN_N, FORCE_MAX_N, FORCE_STEP_N],
                                          "hold_steps": HOLD_STEPS, "settling_steps": SETTLE_STEPS,
                                          "velocity_threshold_mps": VELOCITY_THRESHOLD_MPS,
                                          "displacement_threshold_m": DISPLACEMENT_THRESHOLD_M,
                                          "consecutive_steps": ONSET_CONSECUTIVE_STEPS,
                                          "probe_friction": PROBE_FRICTION, "combine_mode": PROBE_COMBINE_MODE},
                   "material_audit": material_audit, "shared_exposed_material_path": material_shared,
                   "physx_material_handle_identity": "API_NOT_EXPOSED", "probe_metadata": probe_metadata,
                   "results": results, "versions": runtime_versions()}
        write_json(material_path, {"material_audit": material_audit, "shared_exposed_material_path": material_shared,
                                   "physx_material_handle_identity": "API_NOT_EXPOSED"})
        write_json(summary_path, summary)
        with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
            for path in (material_path, raw_path, summary_path):
                if path.is_file():
                    stream.write(f"{sha256_file(path)}  {path.name}\n")
        print(f"P0_7F_OUTPUT_DIR={output_dir}", flush=True)
        print(f"P0_7F_STATUS={summary['status']}", flush=True)
        if error:
            print(f"P0_7F_ERROR={error['type']}: {error['message']}", flush=True)
    return 0 if error is None else 1


if __name__ == "__main__":
    exit_code = main()
    if not ARGS.manual_view:
        SIMULATION_APP.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
