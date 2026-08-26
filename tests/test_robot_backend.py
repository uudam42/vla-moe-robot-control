"""Tests for the Step 9 ``RobotBackend`` abstraction: the ABC itself,
``MuJoCoBackend``, and ``FakeRobotBackend``.
"""

import numpy as np
import pytest
import torch

from control.action import RobotAction
from observations.observation import Observation
from robot_backend.base import RobotBackend
from robot_backend.fake_backend import FakeRobotBackend
from robot_backend.mujoco_backend import MuJoCoBackend
from simulation.environment import SimulationEnvironment


def test_robot_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        RobotBackend()


def test_fake_backend_is_a_robot_backend():
    backend = FakeRobotBackend()
    assert isinstance(backend, RobotBackend)


def test_fake_backend_returns_valid_observation():
    backend = FakeRobotBackend()
    obs = backend.get_observation()
    assert isinstance(obs, Observation)
    assert obs.rgb.shape == (64, 64, 3)
    assert obs.state.shape == (23,)


def test_fake_backend_execute_action_advances_tick_and_records():
    backend = FakeRobotBackend()
    obs_before = backend.get_observation()
    action = RobotAction(joint_targets=np.zeros(7), gripper_target=1.0)
    backend.execute_action(action)
    obs_after = backend.get_observation()

    assert obs_after.timestamp > obs_before.timestamp
    assert np.all(obs_after.state > obs_before.state)
    assert backend.executed_actions == [action]


def test_fake_backend_execute_action_rejects_non_robot_action():
    backend = FakeRobotBackend()
    with pytest.raises(TypeError):
        backend.execute_action("not an action")


def test_fake_backend_reset_clears_ticks_and_history():
    backend = FakeRobotBackend()
    backend.execute_action(RobotAction(joint_targets=np.zeros(7), gripper_target=1.0))
    backend.reset()
    assert backend.get_observation().timestamp == 0.0
    assert backend.executed_actions == []


def test_fake_backend_close_marks_closed():
    backend = FakeRobotBackend()
    assert not backend.closed
    backend.close()
    assert backend.closed


def test_fake_backend_context_manager_closes():
    with FakeRobotBackend() as backend:
        assert not backend.closed
    assert backend.closed


# --- MuJoCoBackend: wraps SimulationEnvironment, no physics duplicated ---


def test_mujoco_backend_is_a_robot_backend():
    with MuJoCoBackend() as backend:
        assert isinstance(backend, RobotBackend)


def test_mujoco_backend_get_observation_matches_env_directly():
    with SimulationEnvironment() as env:
        backend = MuJoCoBackend(env=env)
        direct_obs = env.get_observation()
        backend_obs = backend.get_observation()
        # Not the same object (each call re-renders), but same shapes/dtype
        # and the robot hasn't moved between the two calls.
        assert np.allclose(direct_obs.state, backend_obs.state)
        assert direct_obs.rgb.shape == backend_obs.rgb.shape


def test_mujoco_backend_execute_action_steps_the_environment():
    with SimulationEnvironment() as env:
        backend = MuJoCoBackend(env=env)
        state_before = env.get_robot_state().joint_positions.copy()
        action = RobotAction(joint_targets=state_before + 0.05, gripper_target=1.0)
        backend.execute_action(action)
        state_after = env.get_robot_state().joint_positions
        assert not np.allclose(state_before, state_after)


def test_mujoco_backend_reset_restores_initial_state():
    with SimulationEnvironment() as env:
        backend = MuJoCoBackend(env=env)
        initial = env.get_robot_state().joint_positions.copy()
        backend.execute_action(RobotAction(joint_targets=initial + 0.1, gripper_target=1.0))
        assert not np.allclose(env.get_robot_state().joint_positions, initial)
        backend.reset()
        assert np.allclose(env.get_robot_state().joint_positions, initial)


def test_mujoco_backend_close_only_closes_owned_env():
    with SimulationEnvironment() as env:
        backend = MuJoCoBackend(env=env)
        backend.close()  # should NOT close `env` since backend did not construct it
        env.get_observation()  # still usable


def test_mujoco_backend_constructs_own_env_when_none_given():
    backend = MuJoCoBackend()
    try:
        obs = backend.get_observation()
        assert isinstance(obs, Observation)
    finally:
        backend.close()


def test_mujoco_backend_privileged_passthrough_not_on_the_abc():
    """get_object_position etc. are intentionally NOT part of RobotBackend --
    only a concrete backend (MuJoCoBackend) may expose them."""
    assert not hasattr(RobotBackend, "get_object_position")
    with MuJoCoBackend() as backend:
        position = backend.get_object_position("red_cube")
        assert position.shape == (3,)


# --- Policy code works against FakeRobotBackend with no MuJoCo at all ---


def test_temporal_policy_predicts_against_fake_backend(tiny_temporal_vla_checkpoint):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    with FakeRobotBackend() as backend:
        policy.reset()
        for _ in range(5):
            observation = backend.get_observation()
            action = policy.predict(observation=observation, instruction="Pick up the red cube.")
            backend.execute_action(action)
        assert len(backend.executed_actions) == 5
        assert all(np.all(np.isfinite(a.joint_targets)) for a in backend.executed_actions)
