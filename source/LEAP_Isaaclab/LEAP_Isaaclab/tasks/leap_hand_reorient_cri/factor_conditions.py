"""CRI_FACTOR_SCHEMA_V1 的确定性条件表。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping, Sequence


FACTOR_SCHEMA_VERSION: Final = "CRI_FACTOR_SCHEMA_V1"  # 固定因子语义版本。
PROTOCOL_ID: Final = "CRI_RESEARCH_TASK_CORE_IMPLEMENTATION"  # 固定实现协议。
TASK_ID: Final = "Isaac-Reorient-Cube-Leap-CRI"  # 独立科研任务标识。
NOMINAL_EFFORT_LIMIT: Final = 0.50  # LEAP 基线关节力矩上限。


@dataclass(frozen=True)
class FactorCondition:
    """一个不可变的三因子条件。"""

    code: str  # 位序为质量、摩擦、电机容量。
    bits: tuple[int, int, int]  # 显式二进制因子编码。
    mass_scale: float  # 相对名义质量与惯量的缩放。
    friction_condition: str  # 便于科研日志读取的摩擦标签。
    object_static_friction: float  # 物体侧静摩擦系数。
    object_dynamic_friction: float  # 物体侧动摩擦系数。
    motor_capacity_scale: float  # 全部十六关节的容量缩放。

    @property
    def effective_effort_limit(self) -> float:
        """返回该条件的 PhysX 关节力矩上限。"""

        return NOMINAL_EFFORT_LIMIT * self.motor_capacity_scale


_CONDITIONS = {
    "000": FactorCondition("000", (0, 0, 0), 1.00, "nominal", 0.80, 0.80, 1.00),
    "100": FactorCondition("100", (1, 0, 0), 1.35, "nominal", 0.80, 0.80, 1.00),
    "010": FactorCondition("010", (0, 1, 0), 1.00, "low", 0.45, 0.45, 1.00),
    "001": FactorCondition("001", (0, 0, 1), 1.00, "nominal", 0.80, 0.80, 0.80),
}  # 唯一合法条件集合。

FACTOR_CONDITIONS: Final[Mapping[str, FactorCondition]] = MappingProxyType(_CONDITIONS)  # 阻止运行时改表。
FACTOR_CODE_ORDER: Final = tuple(_CONDITIONS)  # 保持日志和测试顺序稳定。


def get_factor_condition(code: str) -> FactorCondition:
    """按规范代码返回条件，拒绝隐式回退。"""

    try:
        return FACTOR_CONDITIONS[code]
    except KeyError as exc:
        raise ValueError(f"Unsupported {FACTOR_SCHEMA_VERSION} factor code: {code!r}") from exc


def resolve_factor_codes(codes: Sequence[str], num_envs: int) -> tuple[str, ...]:
    """将显式配置解析为逐环境代码，不执行随机采样。"""

    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")
    explicit_codes = tuple(codes)  # 复制配置，避免后续外部修改。
    if len(explicit_codes) == 1:
        explicit_codes = explicit_codes * num_envs  # 仅允许显式单值广播。
    elif len(explicit_codes) != num_envs:
        raise ValueError(
            f"factor_codes must contain one code or exactly num_envs={num_envs} codes; got {len(explicit_codes)}."
        )
    for code in explicit_codes:
        get_factor_condition(code)  # 逐项验证，不使用默认替代非法值。
    return explicit_codes


def validate_schema_invariants() -> None:
    """验证四个条件只改变批准的主因子。"""

    nominal = get_factor_condition("000")  # 名义条件作为唯一参照。
    if FACTOR_CODE_ORDER != ("000", "100", "010", "001"):
        raise AssertionError("Canonical factor-code order changed.")
    expected_changed_fields = {
        "100": {"mass_scale"},
        "010": {"friction_condition", "object_static_friction", "object_dynamic_friction"},
        "001": {"motor_capacity_scale"},
    }  # 每个非名义条件的批准差异。
    comparable_fields = (
        "mass_scale",
        "friction_condition",
        "object_static_friction",
        "object_dynamic_friction",
        "motor_capacity_scale",
    )
    for code, expected in expected_changed_fields.items():
        condition = get_factor_condition(code)
        changed = {field for field in comparable_fields if getattr(condition, field) != getattr(nominal, field)}
        if changed != expected:
            raise AssertionError(f"{code} changed fields {changed}, expected {expected}.")
    expected_effort_limits = {"000": 0.50, "100": 0.50, "010": 0.50, "001": 0.40}
    for code, expected in expected_effort_limits.items():
        actual = get_factor_condition(code).effective_effort_limit
        if abs(actual - expected) > 1.0e-12:
            raise AssertionError(f"{code} effort limit {actual} does not match {expected}.")


validate_schema_invariants()  # 导入时立即锁定规范不变量。
