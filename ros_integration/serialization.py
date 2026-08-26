"""``Observation`` / ``RobotAction`` <-> ROS2-message-shaped field dicts.

Each function here produces or consumes a plain dict whose keys/types
match exactly what would be set on the corresponding real ROS2 message
field (``sensor_msgs/Image``, ``sensor_msgs/JointState``, the custom
``VLARobotAction`` message -- README "Message strategy"). Kept as plain
dicts (not actual message objects) so this module has zero ``rclpy``
dependency and is fully unit-testable without ROS2 installed; the thin
``rclpy`` node wrappers under ``ros2_ws/`` just copy these dict values
onto real message instances and vice versa.

No pickles, no opaque blobs -- every field is a primitive/array type a
real ROS2 message could hold (README "Do not embed Python pickles in
ROS2 messages").
"""

import numpy as np

from control.action import RobotAction
from observations.observation import Observation

IMAGE_ENCODING = "rgb8"
NUM_JOINTS = 7

# Fixed field order for the 23D state vector when carried over
# sensor_msgs/JointState (README "state_dim=23 exactly the RobotState
# layout" -- see observations/robot_state.py's *_SLICE constants). Not a
# perfect semantic fit (JointState is meant for joint angles specifically)
# but avoids inventing a second custom message purely to carry an array
# ROS2 already has a standard-message home for.
STATE_FIELD_NAMES = (
    [f"joint_position_{i}" for i in range(7)]
    + [f"joint_velocity_{i}" for i in range(7)]
    + ["eef_x", "eef_y", "eef_z"]
    + ["eef_qw", "eef_qx", "eef_qy", "eef_qz"]
    + ["gripper_finger_1", "gripper_finger_2"]
)


def rgb_to_image_fields(rgb: np.ndarray) -> dict:
    """``sensor_msgs/Image`` field values for one RGB frame."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    height, width, channels = rgb.shape
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    return {
        "height": int(height),
        "width": int(width),
        "encoding": IMAGE_ENCODING,
        "is_bigendian": 0,
        "step": int(width * channels),
        "data": rgb.tobytes(),
    }


def image_fields_to_rgb(fields: dict) -> np.ndarray:
    if fields["encoding"] != IMAGE_ENCODING:
        raise ValueError(f"unsupported encoding {fields['encoding']!r}, expected {IMAGE_ENCODING!r}")
    height, width = fields["height"], fields["width"]
    return np.frombuffer(fields["data"], dtype=np.uint8).reshape(height, width, 3).copy()


def state_to_joint_state_fields(state: np.ndarray, timestamp_sec: float) -> dict:
    """``sensor_msgs/JointState`` field values for the 23D proprioceptive state."""
    if state.shape != (len(STATE_FIELD_NAMES),):
        raise ValueError(f"state must have shape ({len(STATE_FIELD_NAMES)},), got {state.shape}")
    return {"name": list(STATE_FIELD_NAMES), "position": state.astype(np.float64).tolist(), "stamp_sec": float(timestamp_sec)}


def joint_state_fields_to_state(fields: dict) -> np.ndarray:
    if list(fields["name"]) != list(STATE_FIELD_NAMES):
        raise ValueError("JointState field 'name' does not match the expected state layout")
    return np.array(fields["position"], dtype=np.float64)


def observation_to_message_fields(observation: Observation) -> dict:
    """Convenience: both the image and joint-state fields for one Observation."""
    return {
        "image": rgb_to_image_fields(observation.rgb),
        "joint_state": state_to_joint_state_fields(observation.state, observation.timestamp),
        "timestamp": float(observation.timestamp),
    }


def message_fields_to_observation(fields: dict) -> Observation:
    rgb = image_fields_to_rgb(fields["image"])
    state = joint_state_fields_to_state(fields["joint_state"])
    return Observation(rgb=rgb, state=state, timestamp=float(fields["timestamp"]))


def action_to_vla_action_fields(action: RobotAction, stamp_sec: float) -> dict:
    """Custom ``vla_robot_control_msgs/VLARobotAction`` field values."""
    if action.joint_targets.shape != (NUM_JOINTS,):
        raise ValueError(f"joint_targets must have shape ({NUM_JOINTS},), got {action.joint_targets.shape}")
    return {
        "joint_targets": action.joint_targets.astype(np.float64).tolist(),
        "gripper_target": float(action.gripper_target),
        "stamp_sec": float(stamp_sec),
    }


def vla_action_fields_to_action(fields: dict) -> RobotAction:
    """Inverse of ``action_to_vla_action_fields``. Raises via
    ``RobotAction.__post_init__`` if the fields describe an invalid action
    (wrong shape, non-finite, gripper out of range) -- callers on the
    receiving/backend side should validate with
    ``ros_integration.command_validator`` BEFORE calling this, so a
    malformed message is rejected with a clear reason rather than an
    uncaught exception (README "Action validation")."""
    return RobotAction(
        joint_targets=np.array(fields["joint_targets"], dtype=np.float64),
        gripper_target=float(fields["gripper_target"]),
    )
