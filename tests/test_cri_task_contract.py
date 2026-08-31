"""CRI task 注册与配置隔离的静态 contract tests。"""

from __future__ import annotations

import unittest

import gymnasium as gym

import LEAP_Isaaclab.tasks  # noqa: F401  # 触发项目 task 自动注册。
from LEAP_Isaaclab.tasks.leap_hand_reorient.leap_hand_env_cfg import LeapHandEnvCfg
from LEAP_Isaaclab.tasks.leap_hand_reorient_cri.factor_conditions import resolve_factor_codes
from LEAP_Isaaclab.tasks.leap_hand_reorient_cri.leap_hand_cri_env_cfg import (
    LeapHandCRIEnvCfg,
    spawn_from_usd_with_uninstanceable_physics_material,
)
from LEAP_Isaaclab.tasks.leap_hand_reorient_cri.reorientation_cri_env import ReorientationCRIEnv


class CRITaskContractTests(unittest.TestCase):
    """证明 CRI 注册独立且不覆盖基线语义。"""

    def test_baseline_and_cri_registration_are_isolated(self) -> None:
        baseline = gym.spec("Isaac-Reorient-Cube-Leap")
        cri = gym.spec("Isaac-Reorient-Cube-Leap-CRI")
        self.assertEqual(
            baseline.entry_point,
            "LEAP_Isaaclab.tasks.leap_hand_reorient.reorientation_env:ReorientationEnv",
        )
        self.assertIn("rl_games_cfg_entry_point", baseline.kwargs)
        self.assertEqual(
            cri.entry_point,
            "LEAP_Isaaclab.tasks.leap_hand_reorient_cri.reorientation_cri_env:ReorientationCRIEnv",
        )
        self.assertEqual(set(cri.kwargs), {"env_cfg_entry_point"})  # CRI 不暴露 PPO 配置。

    def test_cri_configuration_matches_frozen_contract(self) -> None:
        cfg = LeapHandCRIEnvCfg()
        self.assertIsNone(cfg.events)
        self.assertFalse(cfg.enable_adr)
        self.assertEqual(cfg.factor_codes, ("000",))
        self.assertEqual(tuple(cfg.object_cfg.spawn.scale), (1.2, 1.2, 1.2))
        self.assertEqual(cfg.robot_cfg.spawn.physics_material.static_friction, 0.80)
        self.assertEqual(cfg.robot_cfg.spawn.physics_material.dynamic_friction, 0.80)
        self.assertEqual(cfg.robot_cfg.spawn.physics_material.friction_combine_mode, "min")
        self.assertEqual(cfg.object_cfg.spawn.physics_material.static_friction, 0.80)
        self.assertEqual(cfg.object_cfg.spawn.physics_material.dynamic_friction, 0.80)
        self.assertEqual(cfg.object_cfg.spawn.physics_material.friction_combine_mode, "min")
        self.assertIs(cfg.robot_cfg.spawn.func, spawn_from_usd_with_uninstanceable_physics_material)
        self.assertIs(cfg.object_cfg.spawn.func, spawn_from_usd_with_uninstanceable_physics_material)

    def test_baseline_configuration_remains_unchanged(self) -> None:
        baseline = LeapHandEnvCfg()
        self.assertIsNotNone(baseline.events)
        self.assertTrue(baseline.enable_adr)
        self.assertEqual(tuple(baseline.object_cfg.spawn.scale), (1.2, 1.2, 1.2))
        self.assertIsNone(baseline.robot_cfg.spawn.physics_material)
        self.assertIsNone(baseline.object_cfg.spawn.physics_material)

    def test_smoke_assignment_is_explicit(self) -> None:
        assignment = ("000", "100", "010", "001")
        cfg = LeapHandCRIEnvCfg()
        cfg.scene.num_envs = 4
        cfg.factor_codes = assignment
        self.assertEqual(resolve_factor_codes(cfg.factor_codes, cfg.scene.num_envs), assignment)

    def test_cri_does_not_override_control_or_task_semantics(self) -> None:
        forbidden_overrides = {
            "_pre_physics_step",
            "_apply_action",
            "_get_observations",
            "_get_rewards",
            "_get_dones",
        }  # 这些符号必须继续来自 baseline。
        self.assertTrue(forbidden_overrides.isdisjoint(ReorientationCRIEnv.__dict__))


if __name__ == "__main__":
    unittest.main()
