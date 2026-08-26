"""Step 8 runtime-purity tests (README "No runtime expert fallback" /
"Runtime"): the final Temporal+DAgger policy is the SAME
``TemporalDenseVLAPolicy`` class Step 7 used (README "DAgger policy
class") -- architecture is unchanged, only the checkpoint's weights
differ. These tests confirm that using a DAgger-fine-tuned checkpoint does
not, and structurally cannot, introduce any expert/privileged-simulator
dependency on the execution path.
"""

import inspect

import numpy as np
import torch

from evaluation import closed_loop
from models import temporal_policy as policy_module
from models.temporal_policy import TemporalDenseVLAPolicy
from observations.observation import Observation
from observations.robot_state import STATE_DIM
from simulation.environment import SimulationEnvironment


def test_policy_module_never_imports_scripted_or_corrective_expert():
    source = inspect.getsource(policy_module)
    assert "scripted_controller" not in source
    assert "corrective_expert" not in source
    assert "dagger" not in source
    assert not hasattr(policy_module, "mujoco")


def test_dagger_finetuned_checkpoint_loads_into_the_same_policy_class(tiny_temporal_vla_checkpoint):
    """A DAgger checkpoint has the identical TemporalDenseVLAConfig shape
    (architecture unchanged -- README "Architecture must stay the same"),
    so it must load into TemporalDenseVLAPolicy with no special-casing."""
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    assert isinstance(policy, TemporalDenseVLAPolicy)


def test_predict_called_with_only_observation_and_instruction(tiny_temporal_vla_checkpoint, monkeypatch):
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
        assert isinstance(observation, Observation)
        assert observation.state.shape == (STATE_DIM,)


def test_run_closed_loop_episode_never_imports_expert_modules():
    import evaluation.closed_loop as closed_loop_module

    assert "import control.scripted_controller" not in inspect.getsource(closed_loop_module)
    assert "from control.scripted_controller" not in inspect.getsource(closed_loop_module)
    assert "corrective_expert" not in inspect.getsource(closed_loop_module)
    assert not hasattr(closed_loop_module, "ScriptedController")
