"""Tests for safety/supervisor.py -- the Step 10 SafetySupervisor."""

import numpy as np
import pytest

from control.action import RobotAction
from safety.supervisor import SafetyDecision, SafetyReason, SafetySupervisor


def make_action(joints=None, gripper=0.5):
    return RobotAction(joint_targets=np.zeros(7) if joints is None else np.asarray(joints), gripper_target=gripper)


def test_accept_valid_action():
    supervisor = SafetySupervisor()
    safe, event = supervisor.process(make_action())
    assert event.decision == SafetyDecision.ACCEPT
    assert event.reason == SafetyReason.NONE
    assert np.allclose(safe.joint_targets, 0.0)


def test_clamp_gripper_out_of_range():
    supervisor = SafetySupervisor()
    action = RobotAction.__new__(RobotAction)  # bypass __post_init__ validation to construct an "invalid" action
    object.__setattr__(action, "joint_targets", np.zeros(7))
    object.__setattr__(action, "gripper_target", 1.7)

    safe, event = supervisor.process(action)
    assert event.decision == SafetyDecision.CLAMP
    assert event.reason == SafetyReason.INVALID_GRIPPER
    assert safe.gripper_target == pytest.approx(1.0)


def test_reject_nonfinite_action():
    supervisor = SafetySupervisor()
    action = RobotAction.__new__(RobotAction)
    joints = np.zeros(7)
    joints[3] = float("nan")
    object.__setattr__(action, "joint_targets", joints)
    object.__setattr__(action, "gripper_target", 0.5)

    safe, event = supervisor.process(action)
    assert event.decision == SafetyDecision.REJECT
    assert event.reason == SafetyReason.NONFINITE_ACTION
    assert safe is None  # no prior safe action established yet


def test_reject_falls_back_to_last_safe_action():
    supervisor = SafetySupervisor()
    supervisor.process(make_action(joints=np.full(7, 0.1)))  # establishes a safe action

    action = RobotAction.__new__(RobotAction)
    object.__setattr__(action, "joint_targets", np.zeros(6))  # wrong shape
    object.__setattr__(action, "gripper_target", 0.5)
    safe, event = supervisor.process(action)

    assert event.decision == SafetyDecision.REJECT
    assert event.reason == SafetyReason.INVALID_SHAPE
    assert np.allclose(safe.joint_targets, 0.1)  # held the previous safe action


def test_clamp_max_joint_delta():
    supervisor = SafetySupervisor(max_joint_delta=0.2)
    supervisor.process(make_action(joints=np.zeros(7)))  # establish previous=0

    safe, event = supervisor.process(make_action(joints=np.full(7, 5.0)))
    assert event.decision == SafetyDecision.CLAMP
    assert event.reason == SafetyReason.MAX_JOINT_DELTA
    assert np.allclose(safe.joint_targets, 0.2)  # clamped to +0.2 from previous 0.0


def test_clamp_joint_limit_using_backend_supplied_range():
    joint_range = np.stack([np.full(7, -1.0), np.full(7, 1.0)], axis=1)  # (7, 2)
    supervisor = SafetySupervisor(joint_range=joint_range, max_joint_delta=10.0)

    safe, event = supervisor.process(make_action(joints=np.full(7, 5.0)))
    assert event.decision == SafetyDecision.CLAMP
    assert event.reason == SafetyReason.JOINT_LIMIT
    assert np.allclose(safe.joint_targets, 1.0)


def test_joint_limit_disabled_when_no_range_given():
    supervisor = SafetySupervisor(joint_range=None, max_joint_delta=100.0)
    safe, event = supervisor.process(make_action(joints=np.full(7, 999.0)))
    assert event.decision == SafetyDecision.ACCEPT
    assert np.allclose(safe.joint_targets, 999.0)


def test_hold_on_stale_observation():
    supervisor = SafetySupervisor(stale_observation_sec=0.5)
    supervisor.process(make_action(joints=np.full(7, 0.3)), now=100.0)

    safe, event = supervisor.process(make_action(joints=np.full(7, 0.9)), observation_timestamp=100.0, now=100.6)
    assert event.decision == SafetyDecision.HOLD
    assert event.reason == SafetyReason.STALE_OBSERVATION
    assert np.allclose(safe.joint_targets, 0.3)  # held, not the fresh (but stale-observation-derived) 0.9


def test_hold_on_command_timeout():
    supervisor = SafetySupervisor(command_timeout_sec=0.05)
    supervisor.process(make_action(joints=np.full(7, 0.3)))

    safe, event = supervisor.process(make_action(joints=np.full(7, 0.9)), inference_latency_sec=0.2)
    assert event.decision == SafetyDecision.HOLD
    assert event.reason == SafetyReason.COMMAND_TIMEOUT


def test_hold_on_backend_not_ready():
    supervisor = SafetySupervisor()
    safe, event = supervisor.process(make_action(), backend_ready=False)
    assert event.decision == SafetyDecision.HOLD
    assert event.reason == SafetyReason.BACKEND_NOT_READY


def test_stop_episode_after_repeated_interventions():
    supervisor = SafetySupervisor(max_consecutive_interventions=3)
    bad_action = RobotAction.__new__(RobotAction)
    object.__setattr__(bad_action, "joint_targets", np.zeros(6))  # always wrong shape
    object.__setattr__(bad_action, "gripper_target", 0.5)

    decisions = []
    for _ in range(5):
        _, event = supervisor.process(bad_action)
        decisions.append(event.decision)

    assert decisions[:3] == [SafetyDecision.REJECT] * 3
    assert decisions[3] == SafetyDecision.STOP_EPISODE
    assert decisions[4] == SafetyDecision.STOP_EPISODE


def test_reset_clears_history_not_event_log():
    supervisor = SafetySupervisor()
    supervisor.process(make_action(joints=np.full(7, 0.5)))
    assert len(supervisor.events) == 1

    supervisor.reset()
    assert supervisor._last_safe_action is None
    assert len(supervisor.events) == 1  # event log is NOT cleared by reset()


def test_safety_event_to_dict_is_json_serializable():
    import json

    supervisor = SafetySupervisor()
    _, event = supervisor.process(make_action())
    serialized = json.dumps(event.to_dict())
    restored = json.loads(serialized)
    assert restored["decision"] == "ACCEPT"
    assert restored["reason"] == "NONE"


def test_safety_module_never_imports_mujoco():
    import inspect

    import safety.supervisor as module

    assert not hasattr(module, "mujoco")
    assert "import mujoco" not in inspect.getsource(module)
