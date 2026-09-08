DRY_RUN
- No serial connection
- No motor writes
- No torque

READ_ONLY
- Serial connection allowed
- State reading only
- Torque must remain disabled

TORQUE_DISABLED
- Explicit torque-off state
- Calibration/read tests only

LOW_AUTHORITY
- Small bounded commands
- Only after H0-H4 PASS

DEPLOYMENT
- Full policy command path
- Only after H6/H7 qualification
