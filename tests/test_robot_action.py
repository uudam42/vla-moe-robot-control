"""Unit tests for the RobotAction data structure itself (no MuJoCo)."""

import numpy as np
import pytest

from control.action import RobotAction


def test_valid_action_constructs():
    action = RobotAction(joint_targets=np.zeros(7), gripper_target=0.5)
    assert action.joint_targets.shape == (7,)
    assert action.gripper_target == 0.5


def test_rejects_wrong_joint_target_length():
    with pytest.raises(ValueError):
        RobotAction(joint_targets=np.zeros(6), gripper_target=0.5)


def test_rejects_non_array_joint_targets():
    with pytest.raises(TypeError):
        RobotAction(joint_targets=[0.0] * 7, gripper_target=0.5)


def test_rejects_non_finite_joint_targets():
    targets = np.zeros(7)
    targets[3] = np.nan
    with pytest.raises(ValueError):
        RobotAction(joint_targets=targets, gripper_target=0.5)

    targets = np.zeros(7)
    targets[3] = np.inf
    with pytest.raises(ValueError):
        RobotAction(joint_targets=targets, gripper_target=0.5)


def test_rejects_gripper_target_out_of_range():
    with pytest.raises(ValueError):
        RobotAction(joint_targets=np.zeros(7), gripper_target=1.5)
    with pytest.raises(ValueError):
        RobotAction(joint_targets=np.zeros(7), gripper_target=-0.1)


def test_rejects_non_finite_gripper_target():
    with pytest.raises(ValueError):
        RobotAction(joint_targets=np.zeros(7), gripper_target=float("nan"))


def test_gripper_target_boundaries_are_valid():
    RobotAction(joint_targets=np.zeros(7), gripper_target=0.0)
    RobotAction(joint_targets=np.zeros(7), gripper_target=1.0)
