"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
try:
    from .ui_extension_example import *
except ModuleNotFoundError as exc:  # Python-only 路径没有 Kit 提供的 omni 模块。
    if exc.name not in ("omni", "omni.ext"):
        raise
