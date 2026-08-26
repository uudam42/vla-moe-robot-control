"""Step 7 boundary tests, mirroring test_no_privileged_vla_inputs.py /
test_moe_no_privileged_inputs.py: the closed-loop runner must call the
temporal policy with ONLY (observation, instruction), and history must
never leak across episodes.
"""

import inspect

import numpy as np
import torch

from evaluation import closed_loop
from models.temporal_policy import TemporalDenseVLAPolicy
from observations.observation import Observation
from observations.robot_state import STATE_DIM
from simulation.environment import SimulationEnvironment


def test_predict_is_called_with_only_observation_and_instruction(tiny_temporal_vla_checkpoint, monkeypatch):
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    calls = []
    original_predict = TemporalDenseVLAPolicy.predict

    def spy_predict(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_predict(self, *args, **kwargs)

    monkeypatch.setattr(TemporalDenseVLAPolicy, "predict", spy_predict)

    with SimulationEnvironment() as env:
        closed_loop.run_closed_loop_episode(env, policy, "Pick up the red cube.", max_steps=5)

    assert len(calls) == 5
    for args, kwargs in calls:
        allowed_keys = {"observation", "instruction"}
        assert set(kwargs) <= allowed_keys
        assert len(args) + len(kwargs) == 2
        observation = kwargs.get("observation", args[0] if args else None)
        instruction = kwargs.get("instruction", args[1] if len(args) > 1 else None)
        assert isinstance(observation, Observation)
        assert observation.state.shape == (STATE_DIM,)
        assert isinstance(instruction, str)


def test_observation_state_excludes_cube_position(tiny_temporal_vla_checkpoint, monkeypatch):
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    observed_states = []
    original_predict = TemporalDenseVLAPolicy.predict

    def spy_predict(self, observation, instruction):
        observed_states.append(observation.state.copy())
        return original_predict(self, observation, instruction)

    monkeypatch.setattr(TemporalDenseVLAPolicy, "predict", spy_predict)

    with SimulationEnvironment() as env:
        closed_loop.run_closed_loop_episode(
            env, policy, "Pick up the red cube.", cube_xy_offset=np.array([0.02, -0.01]), max_steps=5
        )
        cube_position = env.get_object_position("red_cube")

    for state in observed_states:
        assert not np.any(np.isclose(state, cube_position[0], atol=1e-9))
        assert not np.any(np.isclose(state, cube_position[1], atol=1e-9))


def test_closed_loop_episode_resets_policy_history(tiny_temporal_vla_checkpoint):
    """README "Episode reset": run_closed_loop_episode must call
    policy.reset() -- history from one call must not leak into the next."""
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))

    with SimulationEnvironment() as env:
        closed_loop.run_closed_loop_episode(env, policy, "Pick up the red cube.", max_steps=6)
        ticks_after_first_episode = policy._tick
        assert ticks_after_first_episode == 6

        closed_loop.run_closed_loop_episode(env, policy, "Pick up the red cube.", max_steps=3)
        # If history leaked, _tick would continue climbing (e.g. reach 9);
        # a correct reset means the second episode's own tick count is 3.
        assert policy._tick == 3
        assert len(policy._rgb_history) <= policy.history_length


def test_multi_step_temporal_rollout_produces_no_nan_or_crash(tiny_temporal_vla_checkpoint):
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))

    with SimulationEnvironment() as env:
        env.reset()
        policy.reset()
        for _ in range(15):
            observation = env.get_observation()
            action = policy.predict(observation=observation, instruction="Pick up the red cube.")
            assert np.all(np.isfinite(action.joint_targets))
            assert np.isfinite(action.gripper_target)
            env.step(action)

        final_state = env.get_robot_state()
        assert np.all(np.isfinite(final_state.joint_positions))


def test_closed_loop_module_reused_unmodified_for_temporal(tiny_temporal_vla_checkpoint):
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    with SimulationEnvironment() as env:
        result = closed_loop.run_closed_loop_episode(env, policy, "Pick up the red cube.", max_steps=6)
    assert result.episode_length <= 6


def test_predict_never_imports_or_calls_mujoco():
    import models.temporal_policy as policy_module

    assert not hasattr(policy_module, "mujoco")
    assert "import mujoco" not in inspect.getsource(policy_module)
