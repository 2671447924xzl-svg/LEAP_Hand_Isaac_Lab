"""LEAP Hand CRI 因子写入与 PhysX 直接回读环境。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import warp as wp
from isaaclab.sim.utils.stage import get_current_stage

from LEAP_Isaaclab.tasks.leap_hand_reorient.reorientation_env import ReorientationEnv

from .factor_conditions import (
    FACTOR_SCHEMA_VERSION,
    PROTOCOL_ID,
    TASK_ID,
    get_factor_condition,
    resolve_factor_codes,
)
from .leap_hand_cri_env_cfg import HAND_PHYSICS_MATERIAL_PATH, OBJECT_PHYSICS_MATERIAL_PATH, LeapHandCRIEnvCfg


class ReorientationCRIEnv(ReorientationEnv):
    """在基线生命周期上增加 episode-static 的显式 CRI 因子。"""

    cfg: LeapHandCRIEnvCfg

    def __init__(self, cfg: LeapHandCRIEnvCfg, render_mode: str | None = None, **kwargs: Any):
        super().__init__(cfg, render_mode, **kwargs)

        self.factor_code_labels = resolve_factor_codes(self.cfg.factor_codes, self.num_envs)  # 固定逐环境标签。
        self.factor_code_bits = torch.tensor(
            [get_factor_condition(code).bits for code in self.factor_code_labels],
            dtype=torch.int8,
            device=self.device,
        )  # 显式逐环境 assignment buffer。
        self._frozen_factor_code_bits = self.factor_code_bits.clone()  # 检测未授权运行时变化。

        self._nominal_object_mass = self._read_object_masses()  # 从 PhysX 捕获名义质量。
        self._nominal_object_inertia = self._read_object_inertias()  # 从 PhysX 捕获名义惯量。
        self._nominal_effort_limit = self._read_effort_limits()  # 从 PhysX 捕获名义容量。
        self._validate_nominal_shapes()  # 在首次 reset 前锁定形状契约。

        self.episode_id = torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device)  # 首次 reset 后为零。
        self.reset_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)  # 逐环境 reset 计数。
        self._allocate_factor_buffers()  # 请求值与实际值严格分离。

    def _allocate_factor_buffers(self) -> None:
        """分配科研日志所需的逐环境张量。"""

        nan = float("nan")  # 未完成回读时保留不可伪造标记。
        self.mass_scale = torch.full((self.num_envs,), nan, dtype=torch.float32, device=self.device)
        self.motor_capacity_scale = torch.full((self.num_envs,), nan, dtype=torch.float32, device=self.device)
        self.requested_object_mass = torch.full_like(self._nominal_object_mass, nan)
        self.actual_object_mass = torch.full_like(self._nominal_object_mass, nan)
        self.requested_object_inertia = torch.full_like(self._nominal_object_inertia, nan)
        self.actual_object_inertia = torch.full_like(self._nominal_object_inertia, nan)
        self.requested_object_static_friction = torch.full(
            (self.num_envs,), nan, dtype=torch.float32, device=self.device
        )
        self.requested_object_dynamic_friction = torch.full(
            (self.num_envs,), nan, dtype=torch.float32, device=self.device
        )
        self.actual_object_static_friction = torch.full(
            (self.num_envs,), nan, dtype=torch.float32, device=self.device
        )
        self.actual_object_dynamic_friction = torch.full(
            (self.num_envs,), nan, dtype=torch.float32, device=self.device
        )
        self.actual_hand_static_friction = torch.full(
            (self.num_envs,), nan, dtype=torch.float32, device=self.device
        )
        self.actual_hand_dynamic_friction = torch.full(
            (self.num_envs,), nan, dtype=torch.float32, device=self.device
        )
        self.requested_effort_limit = torch.full_like(self._nominal_effort_limit, nan)
        self.actual_effort_limit = torch.full_like(self._nominal_effort_limit, nan)
        self.actual_hand_combine_mode = [""] * self.num_envs  # USD 令牌不能存入张量。
        self.actual_object_combine_mode = [""] * self.num_envs  # USD 令牌独立回读。
        self.actual_availability = {
            field: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for field in ("mass", "inertia", "object_material", "hand_material", "effort_limit", "combine_mode")
        }  # 每个 actual 字段均有显式可用性。

    def _validate_nominal_shapes(self) -> None:
        """拒绝与批准协议不兼容的资产形状。"""

        if self._nominal_object_mass.ndim != 2 or self._nominal_object_mass.shape[0] != self.num_envs:
            raise RuntimeError(f"Unexpected object mass shape: {tuple(self._nominal_object_mass.shape)}")
        if self._nominal_object_mass.shape[1] != self.object.num_bodies:
            raise RuntimeError("Object mass body dimension does not match the rigid-object view.")
        expected_inertia = (self.num_envs, self.object.num_bodies, 9)
        if tuple(self._nominal_object_inertia.shape) != expected_inertia:
            raise RuntimeError(
                f"Unexpected object inertia shape: {tuple(self._nominal_object_inertia.shape)}, expected {expected_inertia}."
            )
        expected_effort = (self.num_envs, self.hand.num_joints)
        if tuple(self._nominal_effort_limit.shape) != expected_effort:
            raise RuntimeError(
                f"Unexpected effort-limit shape: {tuple(self._nominal_effort_limit.shape)}, expected {expected_effort}."
            )
        if self.hand.num_joints != 16:
            raise RuntimeError(f"CRI protocol requires 16 LEAP joints, found {self.hand.num_joints}.")
        expected_nominal_effort = torch.full_like(self._nominal_effort_limit, 0.50)
        torch.testing.assert_close(self._nominal_effort_limit, expected_nominal_effort, atol=1.0e-6, rtol=0.0)

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        """将 reset 索引标准化为设备上的一维 long 张量。"""

        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            normalized = env_ids.to(device=self.device, dtype=torch.long).flatten()
        else:
            normalized = torch.as_tensor(tuple(env_ids), dtype=torch.long, device=self.device).flatten()
        if normalized.numel() == 0:
            return normalized
        if int(normalized.min()) < 0 or int(normalized.max()) >= self.num_envs:
            raise IndexError(f"Environment indices out of range: {normalized.tolist()}")
        if torch.unique(normalized).numel() != normalized.numel():
            raise ValueError(f"Duplicate environment indices are not allowed: {normalized.tolist()}")
        return normalized

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """执行基线 reset 后应用固定的逐环境因子。"""

        normalized = self._normalize_env_ids(env_ids)  # reset 和因子写入共享同一索引。
        super()._reset_idx(normalized)
        if not torch.equal(self.factor_code_bits, self._frozen_factor_code_bits):
            raise RuntimeError("factor_code assignment buffer changed after construction.")
        if normalized.numel() == 0:
            return
        self._apply_factor_conditions(normalized)  # 只写本次 reset 的环境。
        self.reset_count[normalized] += 1
        self.episode_id[normalized] += 1

    def _apply_factor_conditions(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """按显式 assignment buffer 执行选定环境的索引写入。"""

        normalized = self._normalize_env_ids(env_ids)
        if normalized.numel() == 0:
            return
        codes = [self.factor_code_labels[index] for index in normalized.tolist()]  # 不进行 reset 随机采样。
        conditions = [get_factor_condition(code) for code in codes]
        mass_scales = torch.tensor(
            [condition.mass_scale for condition in conditions], dtype=torch.float32, device=self.device
        )
        motor_scales = torch.tensor(
            [condition.motor_capacity_scale for condition in conditions], dtype=torch.float32, device=self.device
        )
        static_friction = torch.tensor(
            [condition.object_static_friction for condition in conditions], dtype=torch.float32, device=self.device
        )
        dynamic_friction = torch.tensor(
            [condition.object_dynamic_friction for condition in conditions], dtype=torch.float32, device=self.device
        )

        requested_mass = self._nominal_object_mass[normalized] * mass_scales[:, None]
        requested_inertia = self._nominal_object_inertia[normalized] * mass_scales[:, None, None]
        requested_effort = self._nominal_effort_limit[normalized] * motor_scales[:, None]

        self.mass_scale[normalized] = mass_scales
        self.motor_capacity_scale[normalized] = motor_scales
        self.requested_object_mass[normalized] = requested_mass
        self.requested_object_inertia[normalized] = requested_inertia
        self.requested_object_static_friction[normalized] = static_friction
        self.requested_object_dynamic_friction[normalized] = dynamic_friction
        self.requested_effort_limit[normalized] = requested_effort

        self.object.set_masses_index(masses=requested_mass, env_ids=normalized)
        self.object.set_inertias_index(inertias=requested_inertia, env_ids=normalized)
        self._write_object_material(normalized, static_friction, dynamic_friction)
        self.hand.write_joint_effort_limit_to_sim_index(limits=requested_effort, env_ids=normalized)
        self._read_back_factor_values(normalized)  # actual 只来自 PhysX/USD getter。
        self._assert_requested_matches_actual(normalized)

    def _read_object_masses(self) -> torch.Tensor:
        """从 PhysX rigid-object view 读取质量。"""

        return wp.to_torch(self.object.root_view.get_masses()).to(device=self.device).clone()

    def _read_object_inertias(self) -> torch.Tensor:
        """从 PhysX rigid-object view 读取并恢复 3x3 惯量展平形状。"""

        inertias = wp.to_torch(self.object.root_view.get_inertias()).to(device=self.device)
        return inertias.reshape(self.num_envs, self.object.num_bodies, 9).clone()

    def _read_effort_limits(self) -> torch.Tensor:
        """从 PhysX articulation view 读取关节最大力。"""

        return wp.to_torch(self.hand.root_view.get_dof_max_forces()).to(device=self.device).clone()

    def _read_materials(self, asset: Any, asset_name: str) -> torch.Tensor:
        """从 PhysX view 读取所有碰撞 shape 的材料三元组。"""

        materials = wp.to_torch(asset.root_view.get_material_properties()).to(device=self.device)
        if materials.ndim == 2 and materials.shape[-1] == 3:
            materials = materials.reshape(self.num_envs, -1, 3)
        if materials.ndim != 3 or materials.shape[0] != self.num_envs or materials.shape[-1] != 3:
            raise RuntimeError(f"Unexpected {asset_name} material shape: {tuple(materials.shape)}")
        if materials.shape[1] != asset.root_view.max_shapes:
            raise RuntimeError(
                f"{asset_name} material shape count {materials.shape[1]} does not match {asset.root_view.max_shapes}."
            )
        return materials.clone()

    def _write_object_material(
        self, env_ids: torch.Tensor, static_friction: torch.Tensor, dynamic_friction: torch.Tensor
    ) -> None:
        """只修改选定环境的物体侧材料系数。"""

        materials = self._read_materials(self.object, "object").cpu()
        cpu_ids = env_ids.to(device="cpu", dtype=torch.int64)
        materials[cpu_ids, :, 0] = static_friction.cpu()[:, None]
        materials[cpu_ids, :, 1] = dynamic_friction.cpu()[:, None]
        indices = cpu_ids.to(dtype=torch.int32)
        self.object.root_view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32), wp.from_torch(indices, dtype=wp.int32)
        )

    def _read_combine_mode(self, env_id: int, asset_prim: str, material_path: str) -> str:
        """直接读取指定环境材料 prim 的 PhysX combine mode。"""

        stage = get_current_stage()
        prim_path = f"{self.scene.env_prim_paths[env_id]}/{asset_prim}/{material_path}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Physics material prim is missing: {prim_path}")
        value = prim.GetAttribute("physxMaterial:frictionCombineMode").Get()
        if value is None:
            raise RuntimeError(f"frictionCombineMode is unavailable: {prim_path}")
        return str(value)

    def _read_back_factor_values(self, env_ids: torch.Tensor) -> None:
        """独立回读所有实际因子并设置可用性标记。"""

        masses = self._read_object_masses()
        inertias = self._read_object_inertias()
        object_materials = self._read_materials(self.object, "object")
        hand_materials = self._read_materials(self.hand, "hand")
        effort_limits = self._read_effort_limits()

        for env_id in env_ids.tolist():
            object_values = object_materials[env_id]
            hand_values = hand_materials[env_id]
            torch.testing.assert_close(
                object_values, object_values[0].expand_as(object_values), atol=1.0e-6, rtol=0.0
            )
            torch.testing.assert_close(hand_values, hand_values[0].expand_as(hand_values), atol=1.0e-6, rtol=0.0)
            self.actual_object_static_friction[env_id] = object_values[0, 0]
            self.actual_object_dynamic_friction[env_id] = object_values[0, 1]
            self.actual_hand_static_friction[env_id] = hand_values[0, 0]
            self.actual_hand_dynamic_friction[env_id] = hand_values[0, 1]
            self.actual_hand_combine_mode[env_id] = self._read_combine_mode(
                env_id, "Robot", HAND_PHYSICS_MATERIAL_PATH
            )
            self.actual_object_combine_mode[env_id] = self._read_combine_mode(
                env_id, "object", OBJECT_PHYSICS_MATERIAL_PATH
            )

        self.actual_object_mass[env_ids] = masses[env_ids]
        self.actual_object_inertia[env_ids] = inertias[env_ids]
        self.actual_effort_limit[env_ids] = effort_limits[env_ids]
        for field in self.actual_availability.values():
            field[env_ids] = True

    def _assert_requested_matches_actual(self, env_ids: torch.Tensor) -> None:
        """在 reset 边界拒绝 requested/actual 不一致。"""

        torch.testing.assert_close(
            self.actual_object_mass[env_ids], self.requested_object_mass[env_ids], atol=1.0e-6, rtol=1.0e-5
        )
        torch.testing.assert_close(
            self.actual_object_inertia[env_ids],
            self.requested_object_inertia[env_ids],
            atol=1.0e-7,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            self.actual_object_static_friction[env_ids],
            self.requested_object_static_friction[env_ids],
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            self.actual_object_dynamic_friction[env_ids],
            self.requested_object_dynamic_friction[env_ids],
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            self.actual_effort_limit[env_ids], self.requested_effort_limit[env_ids], atol=1.0e-6, rtol=0.0
        )
        expected_hand = torch.full_like(self.actual_hand_static_friction[env_ids], 0.80)
        torch.testing.assert_close(self.actual_hand_static_friction[env_ids], expected_hand, atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(self.actual_hand_dynamic_friction[env_ids], expected_hand, atol=1.0e-6, rtol=0.0)
        for env_id in env_ids.tolist():
            if self.actual_hand_combine_mode[env_id] != "min" or self.actual_object_combine_mode[env_id] != "min":
                raise RuntimeError(
                    f"Environment {env_id} combine modes are hand={self.actual_hand_combine_mode[env_id]!r}, "
                    f"object={self.actual_object_combine_mode[env_id]!r}; expected 'min'."
                )

    def get_factor_snapshot(self) -> dict[str, Any]:
        """返回可安全用于 smoke 日志的只读副本。"""

        return {
            "task_id": TASK_ID,
            "protocol_id": PROTOCOL_ID,
            "factor_schema_version": FACTOR_SCHEMA_VERSION,
            "factor_codes": tuple(self.factor_code_labels),
            "factor_code_bits": self.factor_code_bits.clone(),
            "friction_conditions": tuple(
                get_factor_condition(code).friction_condition for code in self.factor_code_labels
            ),
            "mass_scale": self.mass_scale.clone(),
            "motor_capacity_scale": self.motor_capacity_scale.clone(),
            "requested_object_mass": self.requested_object_mass.clone(),
            "actual_object_mass": self.actual_object_mass.clone(),
            "requested_object_inertia": self.requested_object_inertia.clone(),
            "actual_object_inertia": self.actual_object_inertia.clone(),
            "requested_object_static_friction": self.requested_object_static_friction.clone(),
            "actual_object_static_friction": self.actual_object_static_friction.clone(),
            "requested_object_dynamic_friction": self.requested_object_dynamic_friction.clone(),
            "actual_object_dynamic_friction": self.actual_object_dynamic_friction.clone(),
            "actual_hand_static_friction": self.actual_hand_static_friction.clone(),
            "actual_hand_dynamic_friction": self.actual_hand_dynamic_friction.clone(),
            "requested_effort_limit": self.requested_effort_limit.clone(),
            "actual_effort_limit": self.actual_effort_limit.clone(),
            "actual_hand_combine_mode": tuple(self.actual_hand_combine_mode),
            "actual_object_combine_mode": tuple(self.actual_object_combine_mode),
            "actual_availability": {key: value.clone() for key, value in self.actual_availability.items()},
            "episode_id": self.episode_id.clone(),
            "reset_count": self.reset_count.clone(),
        }
