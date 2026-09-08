from pathlib import Path
import sys

API_ROOT = Path(r"D:\Research\LEAP\LEAP_Hand_API\python")

if not API_ROOT.exists():
    raise SystemExit(f"FAIL: API path not found: {API_ROOT}")

sys.path.insert(0, str(API_ROOT))

import leap_hand_utils.leap_hand_utils as lhu
from leap_hand_utils.dynamixel_client import DynamixelClient

print("LEAP API source path:", API_ROOT)
print("leap_hand_utils import: PASS")
print("DynamixelClient import: PASS")

sim_min, sim_max = lhu.LEAPsim_limits()

assert len(sim_min) == 16
assert len(sim_max) == 16

print("joint limit dimension:", len(sim_min))
print("16-DOF API shape: PASS")

print("")
print("IMPORTANT:")
print("DRY RUN ONLY")
print("No serial port opened.")
print("No DynamixelClient instance created.")
print("No torque enabled.")
print("No motor command sent.")
