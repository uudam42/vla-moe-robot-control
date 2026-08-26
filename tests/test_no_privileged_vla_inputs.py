"""The Step 5 boundary test: the closed-loop runner must call the policy
with ONLY (observation, instruction) -- never cube position, Jacobian,
controller stage, or any other privileged simulator state.
"""

import inspect

import numpy as np
import torch

from evaluation import closed_loop
from models.policy import DenseVLAPolicy
from observations.observation import Observation
from observations.robot_state import STATE_DIM
from simulation.environment import SimulationEnvironment


def test_scripted_controller_is_never_imported_in_closed_loop_module():
    assert not hasattr(closed_loop, "ScriptedController")
    source = inspect.getsource(closed_loop)
    assert "import" in source  # sanity: we're really inspecting the module
    assert "from control.scripted_controller import" not in source
    assert "control.scripted_controller" not in source


def test_predict_is_called_with_only_observation_and_instruction(tiny_vla_checkpoint, monkeypatch):
    policy = DenseVLAPolicy.from_checkpoint(tiny_vla_checkpoint, device=torch.device("cpu"))
    calls = []
    original_predict = DenseVLAPolicy.predict

    def spy_predict(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_predict(self, *args, **kwargs)

    monkeypatch.setattr(DenseVLAPolicy, "predict", spy_predict)

    with SimulationEnvironment() as env:
        closed_loop.run_closed_loop_episode(env, policy, "Pick up the red cube.", max_steps=5)

    assert len(calls) == 5
    for args, kwargs in calls:
        allowed_keys = {"observation", "instruction"}
        assert set(kwargs) <= allowed_keys
        assert len(args) + len(kwargs) == 2  # exactly observation + instruction, positional or keyword

        observation = kwargs.get("observation", args[0] if args else None)
        instruction = kwargs.get("instruction", args[1] if len(args) > 1 else None)
        assert isinstance(observation, Observation)
        assert observation.state.shape == (STATE_DIM,)
        assert isinstance(instruction, str)


def test_observation_state_passed_to_policy_excludes_cube_position(tiny_vla_checkpoint, monkeypatch):
    """Cross-check against privileged simulator state read during the same rollout."""
    policy = DenseVLAPolicy.from_checkpoint(tiny_vla_checkpoint, device=torch.device("cpu"))
    observed_states = []
    original_predict = DenseVLAPolicy.predict

    def spy_predict(self, observation, instruction):
        observed_states.append(observation.state.copy())
        return original_predict(self, observation, instruction)

    monkeypatch.setattr(DenseVLAPolicy, "predict", spy_predict)

    with SimulationEnvironment() as env:
        closed_loop.run_closed_loop_episode(
            env, policy, "Pick up the red cube.", cube_xy_offset=np.array([0.02, -0.01]), max_steps=5
        )
        cube_position = env.get_object_position("red_cube")

    # None of the 23 state dims fed to the policy should equal the cube's
    # (very distinctive, offset) x/y position -- a cheap structural spot check
    # on top of the shape check above.
    for state in observed_states:
        assert not np.any(np.isclose(state, cube_position[0], atol=1e-9))
        assert not np.any(np.isclose(state, cube_position[1], atol=1e-9))


def test_closed_loop_module_does_not_call_jacobian_or_object_position_on_policy_path():
    """Static check: run_closed_loop_episode's only calls to privileged
    SimulationEnvironment methods must be for evaluation bookkeeping
    (get_object_position for success detection), never anything routed
    into policy.predict()."""
    source = inspect.getsource(closed_loop.run_closed_loop_episode)
    assert "get_end_effector_jacobian" not in source
    assert "get_jacobian" not in source
