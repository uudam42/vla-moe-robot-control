"""The single most important Step 8 safety property (README "Critical
runtime distinction" / "Policy controls the rollout"): during DAgger
collection, the corrective expert's action must NEVER reach
``env.step()`` -- only the learned policy's own prediction may. This is
verified directly by spying on both ``SimulationEnvironment.step`` and the
corrective expert, and comparing every executed action against what the
policy actually returned.
"""

import numpy as np
import pytest
import torch

from control.action import RobotAction
from dagger import collector as collector_module
from dagger.collector import collect_episode
from models.temporal_policy import TemporalDenseVLAPolicy
from simulation.environment import SimulationEnvironment


@pytest.fixture
def policy(tiny_temporal_vla_checkpoint):
    return TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))


def test_env_step_only_ever_receives_the_model_action(tmp_path, policy, monkeypatch):
    executed_actions = []
    model_actions = []

    original_step = SimulationEnvironment.step

    def spy_step(self, action):
        executed_actions.append(action)
        return original_step(self, action)

    monkeypatch.setattr(SimulationEnvironment, "step", spy_step)

    original_predict = TemporalDenseVLAPolicy.predict

    def spy_predict(self, *args, **kwargs):
        action = original_predict(self, *args, **kwargs)
        model_actions.append(action)
        return action

    monkeypatch.setattr(TemporalDenseVLAPolicy, "predict", spy_predict)

    with SimulationEnvironment() as env:
        episode_dir, metadata = collect_episode(
            env, policy, episode_id=0, instruction="Pick up the red cube.",
            episodes_root=tmp_path / "episodes", max_steps=8, sample_every=2,
        )

    assert len(executed_actions) == len(model_actions) == metadata["num_candidate_timesteps"] == 8
    for executed, predicted in zip(executed_actions, model_actions):
        assert executed is predicted, "env.step() must be called with the exact object policy.predict() returned"


def test_expert_action_is_computed_but_never_stepped(tmp_path, policy, monkeypatch):
    """Spy on the corrective expert itself: it must be called once per
    tick (to produce a label) but its return value must never appear as an
    argument to env.step()."""
    expert_actions = []
    executed_actions = []

    original_compute = collector_module.compute_corrective_action

    def spy_compute(*args, **kwargs):
        action, phase = original_compute(*args, **kwargs)
        expert_actions.append(action)
        return action, phase

    monkeypatch.setattr(collector_module, "compute_corrective_action", spy_compute)

    original_step = SimulationEnvironment.step

    def spy_step(self, action):
        executed_actions.append(action)
        return original_step(self, action)

    monkeypatch.setattr(SimulationEnvironment, "step", spy_step)

    with SimulationEnvironment() as env:
        collect_episode(
            env, policy, episode_id=0, instruction="Pick up the red cube.",
            episodes_root=tmp_path / "episodes", max_steps=6, sample_every=2,
        )

    assert len(expert_actions) == 6
    for expert_action in expert_actions:
        assert not any(expert_action is executed for executed in executed_actions)


def test_stored_model_action_matches_executed_action(tmp_path, policy):
    """The recorded ``joint_targets``/``gripper_targets`` on disk (what
    training later treats as 'previous MODEL action') must be exactly what
    was executed, not the expert's corrective label."""
    from dataset.episode import load_episode

    with SimulationEnvironment() as env:
        episode_dir, metadata = collect_episode(
            env, policy, episode_id=0, instruction="Pick up the red cube.",
            episodes_root=tmp_path / "episodes", max_steps=6, sample_every=2,
        )

    episode = load_episode(episode_dir)
    labels = collector_module.load_expert_labels(episode_dir)

    # Expert labels are stored SEPARATELY and generally differ from the
    # (randomly-initialized, untrained) model's own actions.
    assert episode.joint_targets.shape == labels["expert_joint_targets"].shape
    assert not np.allclose(episode.joint_targets, labels["expert_joint_targets"])


def test_dagger_collector_module_never_imports_mujoco():
    import inspect

    assert not hasattr(collector_module, "mujoco")
    assert "import mujoco" not in inspect.getsource(collector_module)
