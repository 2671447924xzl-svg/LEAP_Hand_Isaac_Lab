"""隔离的 LEAP Hand CRI 科研任务配置。"""

from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import bind_physics_material, clone, make_uninstanceable
from isaaclab.utils.configclass import configclass

from LEAP_Isaaclab.tasks.leap_hand_reorient.leap_hand_env_cfg import LeapHandEnvCfg


HAND_PHYSICS_MATERIAL_PATH = "cri_hand_physics_material"  # 手部显式材料路径。
OBJECT_PHYSICS_MATERIAL_PATH = "cri_object_physics_material"  # 方块显式材料路径。


@clone
def spawn_from_usd_with_uninstanceable_physics_material(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Any:
    """先解除碰撞 prim 实例化，再执行 stronger-than-descendants 材料绑定。"""

    spawn_cfg = cfg.copy()
    spawn_cfg.physics_material = None  # 避免在 instanceable prim 上提前执行无效绑定。
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, spawn_cfg, translation, orientation)
    make_uninstanceable(prim_path)  # 官方工具明确用于逐实例材料修改。
    if cfg.physics_material is not None:
        if cfg.physics_material_path.startswith("/"):
            material_path = cfg.physics_material_path
        else:
            material_path = f"{prim_path}/{cfg.physics_material_path}"
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(prim_path, material_path, stronger_than_descendants=True)
    return prim


@configclass
class LeapHandCRIEnvCfg(LeapHandEnvCfg):
    """保持基线控制语义、仅增加 CRI 因子配置。"""

    enable_adr = False  # 关闭 ADR，不改变正常任务生命周期。
    events = None  # 关闭随机化事件管理器。
    factor_codes: tuple[str, ...] = ("000",)  # 单值只做确定性逐环境广播。

    def __post_init__(self) -> None:
        """在 configclass 实例字段生成后隔离替换 nested cfg。"""

        self.robot_cfg = self.robot_cfg.replace(
            spawn=self.robot_cfg.spawn.replace(
                func=spawn_from_usd_with_uninstanceable_physics_material,
                physics_material_path=HAND_PHYSICS_MATERIAL_PATH,
                physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
                    static_friction=0.80,
                    dynamic_friction=0.80,
                    restitution=0.0,
                    friction_combine_mode="min",
                ),
            )
        )  # 手部摩擦固定为名义值。
        self.object_cfg = self.object_cfg.replace(
            spawn=self.object_cfg.spawn.replace(
                func=spawn_from_usd_with_uninstanceable_physics_material,
                scale=(1.2, 1.2, 1.2),
                physics_material_path=OBJECT_PHYSICS_MATERIAL_PATH,
                physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
                    static_friction=0.80,
                    dynamic_friction=0.80,
                    restitution=0.0,
                    friction_combine_mode="min",
                ),
            )
        )  # 方块几何固定，摩擦仅由 CRI 写入改变。
