"""Tests for ros_integration/bridge_node_core.py -- the MuJoCo bridge
node's rclpy-independent command handling/execution/watchdog logic."""

import numpy as np

from robot_backend.fake_backend import FakeRobotBackend
from ros_integration.bridge_node_core import MuJoCoBridgeNodeCore
from ros_integration.command_validator import CommandValidator
from ros_integration.watchdog import CommandWatchdog


def test_valid_action_received_then_executed_on_tick():
    core = MuJoCoBridgeNodeCore(FakeRobotBackend())
    result = core.on_action_received(np.zeros(7), 0.5, now=1.0)
    assert result.valid
    assert core.stats.commands_received == 1
    assert core.stats.commands_rejected == 0

    action, timed_out = core.tick(now=1.01)
    assert timed_out is False
    assert action is not None
    assert core.backend.executed_actions == [action]
    assert core.stats.ticks_executed == 1


def test_invalid_action_rejected_and_not_executed():
    core = MuJoCoBridgeNodeCore(FakeRobotBackend())
    result = core.on_action_received(np.zeros(6), 0.5, now=1.0)  # wrong joint count
    assert not result.valid
    assert core.stats.commands_rejected == 1

    action, timed_out = core.tick(now=1.01)
    assert action is None  # nothing was ever validly received
    assert core.backend.executed_actions == []


def test_nan_action_rejected():
    core = MuJoCoBridgeNodeCore(FakeRobotBackend())
    joints = np.zeros(7)
    joints[2] = float("nan")
    result = core.on_action_received(joints, 0.5, now=1.0)
    assert not result.valid
    assert core.backend.executed_actions == []


def test_watchdog_holds_last_action_when_commands_stop():
    core = MuJoCoBridgeNodeCore(FakeRobotBackend(), watchdog=CommandWatchdog(timeout_sec=0.2))
    core.on_action_received(np.full(7, 0.1), 0.5, now=1.0)
    core.tick(now=1.01)  # executes fresh command

    action, timed_out = core.tick(now=1.5)  # way past the 0.2s watchdog timeout
    assert timed_out is True
    assert action is not None  # still holds the last safe action
    assert core.stats.watchdog_holds == 1
    assert len(core.backend.executed_actions) == 2  # held action still executed (simulated hold-in-place)


def test_missing_command_never_executes_anything():
    core = MuJoCoBridgeNodeCore(FakeRobotBackend())
    action, timed_out = core.tick(now=1.0)
    assert action is None
    assert timed_out is False
    assert core.backend.executed_actions == []


def test_reset_resets_backend_and_watchdog():
    core = MuJoCoBridgeNodeCore(FakeRobotBackend(), watchdog=CommandWatchdog(timeout_sec=0.2))
    core.on_action_received(np.zeros(7), 0.5, now=1.0)
    core.tick(now=1.01)
    core.reset()

    action, timed_out = core.tick(now=1.02)
    assert action is None  # watchdog has no command again after reset
    assert core.backend.executed_actions == []  # backend.reset() cleared history too


def test_extreme_joint_delta_rejected_against_last_accepted_command():
    core = MuJoCoBridgeNodeCore(FakeRobotBackend(), validator=CommandValidator(max_joint_delta=0.3))
    core.on_action_received(np.zeros(7), 0.5, now=1.0)
    result = core.on_action_received(np.full(7, 5.0), 0.5, now=1.1)  # huge jump from last accepted
    assert not result.valid
    assert "extreme joint delta" in result.reason


def test_bridge_node_core_never_imports_mujoco_or_rclpy():
    import inspect

    import ros_integration.bridge_node_core as module

    assert not hasattr(module, "mujoco")
    assert not hasattr(module, "rclpy")
    assert "import mujoco" not in inspect.getsource(module)
    assert "import rclpy" not in inspect.getsource(module)
