"""Tests for ros_integration/command_validator.py."""

import numpy as np
import pytest

from control.action import RobotAction
from ros_integration.command_validator import CommandRejected, CommandValidator


def test_valid_command_accepted():
    validator = CommandValidator()
    result = validator.validate(np.zeros(7), 0.5)
    assert result.valid
    assert result.reason is None


def test_wrong_joint_count_rejected():
    validator = CommandValidator()
    result = validator.validate(np.zeros(6), 0.5)
    assert not result.valid
    assert "joint count" in result.reason


def test_nan_joint_targets_rejected():
    validator = CommandValidator()
    joints = np.zeros(7)
    joints[3] = float("nan")
    result = validator.validate(joints, 0.5)
    assert not result.valid
    assert "non-finite" in result.reason


def test_inf_joint_targets_rejected():
    validator = CommandValidator()
    joints = np.zeros(7)
    joints[0] = float("inf")
    result = validator.validate(joints, 0.5)
    assert not result.valid


def test_gripper_out_of_range_rejected():
    validator = CommandValidator()
    assert not validator.validate(np.zeros(7), 1.5).valid
    assert not validator.validate(np.zeros(7), -0.1).valid


def test_nan_gripper_rejected():
    validator = CommandValidator()
    result = validator.validate(np.zeros(7), float("nan"))
    assert not result.valid


def test_extreme_joint_delta_rejected_when_configured():
    validator = CommandValidator(max_joint_delta=0.2)
    previous = np.zeros(7)
    target = np.full(7, 1.0)  # 1.0 rad jump, way past 0.2
    result = validator.validate(target, 0.5, previous_joint_positions=previous)
    assert not result.valid
    assert "extreme joint delta" in result.reason


def test_small_joint_delta_accepted_when_configured():
    validator = CommandValidator(max_joint_delta=0.2)
    previous = np.zeros(7)
    target = np.full(7, 0.05)
    result = validator.validate(target, 0.5, previous_joint_positions=previous)
    assert result.valid


def test_delta_check_skipped_when_no_previous_given():
    validator = CommandValidator(max_joint_delta=0.01)
    result = validator.validate(np.full(7, 5.0), 0.5, previous_joint_positions=None)
    assert result.valid  # no previous reference -> delta check inapplicable


def test_build_action_returns_robot_action_for_valid_input():
    validator = CommandValidator()
    action = validator.build_action(np.zeros(7), 0.5)
    assert isinstance(action, RobotAction)


def test_build_action_raises_command_rejected_for_invalid_input():
    validator = CommandValidator()
    with pytest.raises(CommandRejected):
        validator.build_action(np.zeros(6), 0.5)


def test_command_validator_module_never_imports_rclpy():
    import inspect

    import ros_integration.command_validator as module

    assert not hasattr(module, "rclpy")
    assert "import rclpy" not in inspect.getsource(module)
