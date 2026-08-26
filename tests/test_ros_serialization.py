"""Tests for ros_integration/serialization.py -- Observation/RobotAction
<-> ROS2-message-shaped field dicts. No rclpy required."""

import numpy as np
import pytest

from control.action import RobotAction
from observations.observation import Observation
from ros_integration.serialization import (
    STATE_FIELD_NAMES,
    action_to_vla_action_fields,
    image_fields_to_rgb,
    joint_state_fields_to_state,
    message_fields_to_observation,
    observation_to_message_fields,
    rgb_to_image_fields,
    state_to_joint_state_fields,
    vla_action_fields_to_action,
)


def make_observation(seed=0):
    rgb = np.random.default_rng(seed).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    state = np.random.default_rng(seed + 1).normal(size=23)
    return Observation(rgb=rgb, state=state, timestamp=1.5)


def test_rgb_round_trip():
    rgb = np.random.default_rng(0).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    fields = rgb_to_image_fields(rgb)
    assert fields["encoding"] == "rgb8"
    assert fields["height"] == 48 and fields["width"] == 64
    assert isinstance(fields["data"], bytes)
    restored = image_fields_to_rgb(fields)
    assert np.array_equal(rgb, restored)


def test_image_fields_reject_wrong_encoding():
    fields = rgb_to_image_fields(np.zeros((4, 4, 3), dtype=np.uint8))
    fields["encoding"] = "bgr8"
    with pytest.raises(ValueError):
        image_fields_to_rgb(fields)


def test_rgb_to_image_fields_rejects_wrong_shape():
    with pytest.raises(ValueError):
        rgb_to_image_fields(np.zeros((4, 4), dtype=np.uint8))


def test_state_round_trip():
    state = np.random.default_rng(2).normal(size=23)
    fields = state_to_joint_state_fields(state, timestamp_sec=3.0)
    assert fields["name"] == list(STATE_FIELD_NAMES)
    assert fields["stamp_sec"] == 3.0
    restored = joint_state_fields_to_state(fields)
    assert np.allclose(state, restored)


def test_state_field_name_mismatch_rejected():
    fields = state_to_joint_state_fields(np.zeros(23), 0.0)
    fields["name"] = ["wrong"] * 23
    with pytest.raises(ValueError):
        joint_state_fields_to_state(fields)


def test_state_wrong_shape_rejected():
    with pytest.raises(ValueError):
        state_to_joint_state_fields(np.zeros(10), 0.0)


def test_observation_round_trip():
    obs = make_observation()
    fields = observation_to_message_fields(obs)
    restored = message_fields_to_observation(fields)
    assert np.array_equal(obs.rgb, restored.rgb)
    assert np.allclose(obs.state, restored.state)
    assert obs.timestamp == restored.timestamp


def test_action_round_trip():
    action = RobotAction(joint_targets=np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.0]), gripper_target=0.75)
    fields = action_to_vla_action_fields(action, stamp_sec=10.0)
    assert fields["stamp_sec"] == 10.0
    restored = vla_action_fields_to_action(fields)
    assert np.allclose(action.joint_targets, restored.joint_targets)
    assert action.gripper_target == pytest.approx(restored.gripper_target)


def test_action_wrong_shape_rejected_on_deserialization():
    """RobotAction's own __post_init__ enforces shape -- this asserts the
    round trip surfaces that as a clear error, not silent corruption."""
    action = RobotAction(joint_targets=np.zeros(7), gripper_target=0.5)
    fields = action_to_vla_action_fields(action, 0.0)
    fields["joint_targets"] = fields["joint_targets"][:5]
    with pytest.raises(ValueError):
        vla_action_fields_to_action(fields)


def test_serialization_module_never_imports_rclpy():
    import inspect

    import ros_integration.serialization as module

    assert not hasattr(module, "rclpy")
    assert "import rclpy" not in inspect.getsource(module)
