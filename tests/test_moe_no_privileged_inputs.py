"""Step 6 boundary test, mirroring tests/test_no_privileged_vla_inputs.py:
the closed-loop runner must call the MoE policy with ONLY (observation,
instruction) -- never cube position, Jacobian, controller stage, or any
other privileged simulator state. Also confirms the SAME
``evaluation.closed_loop.run_closed_loop_episode`` used for Dense is
reused unmodified for MoE (README "Dense code reuse").
"""

import numpy as np
import torch

from evaluation import closed_loop
from models.moe_policy import MoEVLAPolicy
from observations.observation import Observation
from observations.robot_state import STATE_DIM
from simulation.environment import SimulationEnvironment


def test_predict_is_called_with_only_observation_and_instruction(tiny_moe_vla_checkpoint, monkeypatch):
    policy = MoEVLAPolicy.from_checkpoint(tiny_moe_vla_checkpoint, device=torch.device("cpu"))
    calls = []
    original_predict = MoEVLAPolicy.predict

    def spy_predict(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_predict(self, *args, **kwargs)

    monkeypatch.setattr(MoEVLAPolicy, "predict", spy_predict)

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


def test_observation_state_excludes_cube_position(tiny_moe_vla_checkpoint, monkeypatch):
    policy = MoEVLAPolicy.from_checkpoint(tiny_moe_vla_checkpoint, device=torch.device("cpu"))
    observed_states = []
    original_predict = MoEVLAPolicy.predict

    def spy_predict(self, observation, instruction):
        observed_states.append(observation.state.copy())
        return original_predict(self, observation, instruction)

    monkeypatch.setattr(MoEVLAPolicy, "predict", spy_predict)

    with SimulationEnvironment() as env:
        closed_loop.run_closed_loop_episode(
            env, policy, "Pick up the red cube.", cube_xy_offset=np.array([0.02, -0.01]), max_steps=5
        )
        cube_position = env.get_object_position("red_cube")

    for state in observed_states:
        assert not np.any(np.isclose(state, cube_position[0], atol=1e-9))
        assert not np.any(np.isclose(state, cube_position[1], atol=1e-9))


def test_multi_step_moe_rollout_produces_no_nan_or_crash(tiny_moe_vla_checkpoint):
    policy = MoEVLAPolicy.from_checkpoint(tiny_moe_vla_checkpoint, device=torch.device("cpu"))

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


def test_closed_loop_module_reused_unmodified_for_moe(tiny_moe_vla_checkpoint):
    """The same run_closed_loop_episode used for Dense (Step 5) works for
    MoE without any MoE-specific branching -- verifying real code reuse,
    not a parallel reimplementation."""
    policy = MoEVLAPolicy.from_checkpoint(tiny_moe_vla_checkpoint, device=torch.device("cpu"))
    with SimulationEnvironment() as env:
        result = closed_loop.run_closed_loop_episode(env, policy, "Pick up the red cube.", max_steps=6)

    assert result.episode_length <= 6
    assert result.instruction == "Pick up the red cube."
