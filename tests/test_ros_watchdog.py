"""Tests for ros_integration/watchdog.py."""

import numpy as np
import pytest

from control.action import RobotAction
from ros_integration.watchdog import CommandWatchdog


def make_action(value=0.0):
    return RobotAction(joint_targets=np.full(7, value), gripper_target=0.5)


def test_no_command_yet_returns_none_not_timed_out():
    watchdog = CommandWatchdog(timeout_sec=0.5)
    action, timed_out = watchdog.get_action(now=100.0)
    assert action is None
    assert timed_out is False
    assert watchdog.timeout_count == 0


def test_fresh_command_returned_untimed_out():
    watchdog = CommandWatchdog(timeout_sec=0.5)
    action = make_action(1.0)
    watchdog.record_command(action, now=100.0)
    returned, timed_out = watchdog.get_action(now=100.1)
    assert returned is action
    assert timed_out is False


def test_stale_command_triggers_hold_and_timeout_flag():
    watchdog = CommandWatchdog(timeout_sec=0.5)
    action = make_action(1.0)
    watchdog.record_command(action, now=100.0)
    returned, timed_out = watchdog.get_action(now=100.6)  # 0.6s > 0.5s timeout
    assert returned is action  # holds the LAST SAFE action, not None/fresh
    assert timed_out is True


def test_timeout_count_increments_once_per_transition_not_per_tick():
    watchdog = CommandWatchdog(timeout_sec=0.5)
    watchdog.record_command(make_action(), now=100.0)
    for now in [100.6, 100.7, 100.8, 100.9]:
        watchdog.get_action(now=now)
    assert watchdog.timeout_count == 1  # not 4


def test_new_command_after_timeout_clears_timed_out_state():
    watchdog = CommandWatchdog(timeout_sec=0.5)
    watchdog.record_command(make_action(1.0), now=100.0)
    watchdog.get_action(now=100.6)  # times out, count=1
    watchdog.record_command(make_action(2.0), now=101.0)  # fresh command arrives
    returned, timed_out = watchdog.get_action(now=101.1)
    assert timed_out is False
    assert watchdog.timeout_count == 1

    watchdog.get_action(now=101.7)  # times out again -> new transition
    assert watchdog.timeout_count == 2


def test_reset_clears_history_but_not_cumulative_count():
    watchdog = CommandWatchdog(timeout_sec=0.5)
    watchdog.record_command(make_action(), now=100.0)
    watchdog.get_action(now=100.6)
    assert watchdog.timeout_count == 1

    watchdog.reset()
    action, timed_out = watchdog.get_action(now=200.0)
    assert action is None
    assert timed_out is False
    assert watchdog.timeout_count == 1  # cumulative diagnostic survives reset


def test_invalid_timeout_rejected():
    with pytest.raises(ValueError):
        CommandWatchdog(timeout_sec=0.0)
    with pytest.raises(ValueError):
        CommandWatchdog(timeout_sec=-1.0)


def test_watchdog_module_never_imports_rclpy():
    import inspect

    import ros_integration.watchdog as module

    assert not hasattr(module, "rclpy")
    assert "import rclpy" not in inspect.getsource(module)
