"""LEAP Hand CRI 科研任务注册。"""

import gymnasium as gym

from .factor_conditions import TASK_ID


_TASK_PACKAGE = "LEAP_Isaaclab.tasks.leap_hand_reorient_cri"  # 隔离的任务包入口。

gym.register(
    id=TASK_ID,
    entry_point=f"{_TASK_PACKAGE}.reorientation_cri_env:ReorientationCRIEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_cri_env_cfg:LeapHandCRIEnvCfg",
    },
)  # 不暴露任何 PPO 配置入口。
