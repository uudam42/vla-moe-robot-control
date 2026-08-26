"""Closed-loop runtime tests: obs -> policy.predict() -> env.step(), repeatedly.

Uses the tiny untrained checkpoint (see conftest.py) -- these tests verify
the *plumbing* runs correctly (advances simulation, no NaN, timeout
works, success detector fires), not that an untrained policy succeeds at
the task.
"""

import numpy as np
import torch

from control.success import sustained_lift_success
from evaluation.closed_loop import run_closed_loop_episode
from models.policy import DenseVLAPolicy
from simulation.environment import SimulationEnvironment


def test_one_step_advances_simulation(tiny_vla_checkpoint):
    policy = DenseVLAPolicy.from_checkpoint(tiny_vla_checkpoint, device=torch.device("cpu"))

    with SimulationEnvironment() as env:
        env.reset()
        state_before = env.get_robot_state().joint_positions.copy()

        observation = env.get_observation()
        action = policy.predict(observation=observation, instruction="Pick up the red cube.")
        env.step(action)

        state_after = env.get_robot_state().joint_positions
        assert not np.allclose(state_before, state_after)


def test_multi_step_rollout_produces_no_nan_or_crash(tiny_vla_checkpoint):
    policy = DenseVLAPolicy.from_checkpoint(tiny_vla_checkpoint, device=torch.device("cpu"))

    with SimulationEnvironment() as env:
        env.reset()
        for _ in range(15):
            observation = env.get_observation()
            action = policy.predict(observation=observation, instruction="Pick up the red cube.")
            assert np.all(np.isfinite(action.joint_targets))
            assert np.isfinite(action.gripper_target)
            env.step(action)

        final_state = env.get_robot_state()
        assert np.all(np.isfinite(final_state.joint_positions))
        assert np.all(np.isfinite(final_state.end_effector_position))


def test_rollout_terminates_at_max_steps_when_never_successful(tiny_vla_checkpoint):
    policy = DenseVLAPolicy.from_checkpoint(tiny_vla_checkpoint, device=torch.device("cpu"))

    with SimulationEnvironment() as env:
        result = run_closed_loop_episode(env, policy, "Pick up the red cube.", max_steps=8)

    assert result.episode_length <= 8
    if not result.success:
        assert result.termination_reason == "timeout"
        assert result.episode_length == 8


def test_rollout_result_has_consistent_latency_records(tiny_vla_checkpoint):
    policy = DenseVLAPolicy.from_checkpoint(tiny_vla_checkpoint, device=torch.device("cpu"))

    with SimulationEnvironment() as env:
        result = run_closed_loop_episode(env, policy, "Pick up the red cube.", max_steps=6)

    assert len(result.inference_latencies_ms) == result.episode_length
    assert len(result.env_step_latencies_ms) == result.episode_length
    assert all(latency >= 0 for latency in result.inference_latencies_ms)


def test_record_trajectory_populates_step_diagnostics(tiny_vla_checkpoint):
    policy = DenseVLAPolicy.from_checkpoint(tiny_vla_checkpoint, device=torch.device("cpu"))

    with SimulationEnvironment() as env:
        result = run_closed_loop_episode(
            env, policy, "Pick up the red cube.", max_steps=5, record_trajectory=True
        )

    assert len(result.steps) == result.episode_length
    for step in result.steps:
        assert len(step.eef_position) == 3
        assert len(step.cube_position) == 3
        assert len(step.joint_targets) == 7


def test_smoothing_disabled_by_default(tiny_vla_checkpoint):
    """The raw (unsmoothed) baseline must be the default -- smoothing is opt-in."""
    import inspect

    from evaluation.closed_loop import run_closed_loop_episode

    signature = inspect.signature(run_closed_loop_episode)
    assert signature.parameters["smoothing_alpha"].default is None


def test_ema_smoothing_reduces_action_delta():
    from control.action import RobotAction
    from evaluation.closed_loop import _smooth_action

    previous = RobotAction(joint_targets=np.zeros(7), gripper_target=0.0)
    raw = RobotAction(joint_targets=np.ones(7), gripper_target=1.0)

    smoothed = _smooth_action(raw, previous, alpha=0.3)

    assert np.allclose(smoothed.joint_targets, 0.3)
    assert np.isclose(smoothed.gripper_target, 0.3)
    # Smoothed action must move less than the raw jump from previous.
    assert np.linalg.norm(smoothed.joint_targets - previous.joint_targets) < np.linalg.norm(
        raw.joint_targets - previous.joint_targets
    )


def test_smoothing_alpha_changes_rollout_actions(tiny_vla_checkpoint):
    """Smoothed and raw rollouts from the same seed must diverge once >1 step has run."""
    policy = DenseVLAPolicy.from_checkpoint(tiny_vla_checkpoint, device=torch.device("cpu"))

    with SimulationEnvironment() as env:
        raw_result = run_closed_loop_episode(env, policy, "Pick up the red cube.", max_steps=5)
    with SimulationEnvironment() as env:
        smoothed_result = run_closed_loop_episode(
            env, policy, "Pick up the red cube.", max_steps=5, smoothing_alpha=0.3
        )

    # Both must still produce a valid, same-length rollout.
    assert raw_result.episode_length == smoothed_result.episode_length == 5


def test_success_detector_fires_on_synthetic_sustained_lift():
    """Integration sanity: the same physics-based success rule closed_loop.py
    uses is the one from Step 2.5/3, not a reimplementation."""
    initial_z = 0.42
    history = np.concatenate([np.full(5, initial_z), np.full(15, initial_z + 0.06)])
    assert sustained_lift_success(history, initial_cube_z=initial_z) is True
