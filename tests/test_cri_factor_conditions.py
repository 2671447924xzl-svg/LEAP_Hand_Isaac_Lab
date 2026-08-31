"""CRI 因子规范的纯 Python 单元测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "LEAP_Isaaclab"
    / "LEAP_Isaaclab"
    / "tasks"
    / "leap_hand_reorient_cri"
    / "factor_conditions.py"
)  # 绕过 Isaac package 初始化，保持测试纯净。
SPEC = importlib.util.spec_from_file_location("cri_factor_conditions_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load factor module from {MODULE_PATH}.")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE  # dataclass 解析注解时需要模块已注册。
SPEC.loader.exec_module(MODULE)


class FactorConditionTests(unittest.TestCase):
    """锁定 CRI_FACTOR_SCHEMA_V1 的全部离散条件。"""

    def test_canonical_order_and_values(self) -> None:
        self.assertEqual(MODULE.FACTOR_CODE_ORDER, ("000", "100", "010", "001"))
        expected = {
            "000": ((0, 0, 0), 1.00, 0.80, 0.80, 1.00, 0.50),
            "100": ((1, 0, 0), 1.35, 0.80, 0.80, 1.00, 0.50),
            "010": ((0, 1, 0), 1.00, 0.45, 0.45, 1.00, 0.50),
            "001": ((0, 0, 1), 1.00, 0.80, 0.80, 0.80, 0.40),
        }  # 审核批准的唯一数值表。
        for code, values in expected.items():
            condition = MODULE.get_factor_condition(code)
            actual = (
                condition.bits,
                condition.mass_scale,
                condition.object_static_friction,
                condition.object_dynamic_friction,
                condition.motor_capacity_scale,
                condition.effective_effort_limit,
            )
            self.assertEqual(actual, values)

    def test_default_code_broadcasts_without_sampling(self) -> None:
        self.assertEqual(MODULE.resolve_factor_codes(("000",), 4), ("000", "000", "000", "000"))

    def test_explicit_smoke_assignment_is_preserved(self) -> None:
        codes = ("000", "100", "010", "001")
        self.assertEqual(MODULE.resolve_factor_codes(codes, 4), codes)

    def test_invalid_code_and_length_fail(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.resolve_factor_codes(("111",), 4)
        with self.assertRaises(ValueError):
            MODULE.resolve_factor_codes(("000", "100"), 4)
        with self.assertRaises(ValueError):
            MODULE.resolve_factor_codes((), 4)
        with self.assertRaises(ValueError):
            MODULE.resolve_factor_codes(("000",), 0)

    def test_mapping_is_immutable(self) -> None:
        with self.assertRaises(TypeError):
            MODULE.FACTOR_CONDITIONS["111"] = MODULE.FACTOR_CONDITIONS["000"]


if __name__ == "__main__":
    unittest.main()
