# Policy output contract (README "Preserve RobotAction contract"): mirrors
# control.action.RobotAction exactly -- 7 joint targets (radians) + 1
# gripper target in [0, 1]. No pickles, no opaque blobs -- every field is
# a primitive/array type.

builtin_interfaces/Time stamp
float64[7] joint_targets
float64 gripper_target
