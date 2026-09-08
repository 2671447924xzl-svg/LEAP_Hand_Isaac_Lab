from pathlib import Path
import os
import sys

# Repository layout expected on both machines:
#
# <LEAP_ROOT>/
# ├── LEAP_Hand_Isaac_Lab
# └── LEAP_Hand_API
#
# Optional override:
#   LEAP_HAND_API_PYTHON=/custom/path/to/LEAP_Hand_API/python

REPO_ROOT = Path(__file__).resolve().parents[2]

default_api_root = REPO_ROOT.parent / "LEAP_Hand_API" / "python"

env_api_root = os.environ.get("LEAP_HAND_API_PYTHON")

if env_api_root:
    API_ROOT = Path(env_api_root).expanduser()
else:
    API_ROOT = default_api_root

API_ROOT = API_ROOT.resolve()

print("Research repo:", REPO_ROOT)
print("LEAP API source path:", API_ROOT)

if not API_ROOT.exists():
    raise SystemExit(
        f"FAIL: LEAP API path not found: {API_ROOT}\n"
        "Set LEAP_HAND_API_PYTHON if your API repository is elsewhere."
    )

sys.path.insert(0, str(API_ROOT))

import leap_hand_utils.leap_hand_utils as lhu
from leap_hand_utils.dynamixel_client import DynamixelClient

print("leap_hand_utils import: PASS")
print("DynamixelClient import: PASS")

sim_min, sim_max = lhu.LEAPsim_limits()

assert len(sim_min) == 16
assert len(sim_max) == 16
assert (sim_max > sim_min).all()

print("joint limit dimension:", len(sim_min))
print("16-DOF API shape: PASS")

print()
print("IMPORTANT:")
print("DRY RUN ONLY")
print("No serial port opened.")
print("No DynamixelClient instance created.")
print("No torque enabled.")
print("No motor command sent.")

print()
print("LEAP_API_OFFLINE_CHECK_PASS")
