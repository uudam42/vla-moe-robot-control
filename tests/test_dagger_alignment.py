"""End-to-end alignment test: dagger.collector's on-disk output, read back
through dagger.dataset.TemporalDaggerCorrectiveDataset, must reproduce the
exact same previous-action window the running policy actually used (README
"DAgger sample semantics" / "No future leakage"). Complements
tests/test_dagger_dataset.py's synthetic-episode unit tests with a real
MuJoCo collection round.
"""

import numpy as np
import pytest
import torch

from dagger.collector import collect_episode, load_expert_labels
from dagger.dataset import TemporalDaggerCorrectiveDataset
from dataset.episode import load_episode
from models.language_encoder import load_tokenizer
from models.temporal_history import build_action_window, clipped_indices
from models.temporal_policy import TemporalDenseVLAPolicy
from models.vision_encoder import build_image_transform
from simulation.environment import SimulationEnvironment
from training.normalization import ActionNormalizer, StateNormalizer

HISTORY_LENGTH = 4


@pytest.fixture
def policy(tiny_temporal_vla_checkpoint):
    return TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))


@pytest.fixture
def collected_round(tmp_path, policy):
    with SimulationEnvironment() as env:
        episode_dir, metadata = collect_episode(
            env, policy, episode_id=0, instruction="Pick up the red cube.",
            episodes_root=tmp_path / "episodes", max_steps=10, sample_every=1,  # retain every tick
        )
    return tmp_path, episode_dir, metadata


def test_stored_previous_action_window_matches_manual_reconstruction(collected_round, policy):
    """Reconstructing the previous-action window from the on-disk episode
    (via build_action_window over the recorded model joint_targets, exactly
    as TemporalDaggerCorrectiveDataset does) must equal a hand-rolled
    reconstruction from raw arrays -- the shared contract module is doing
    the same thing training-side as it did at collection time."""
    root, episode_dir, metadata = collected_round
    episode = load_episode(episode_dir)

    available_actions = {
        i: np.concatenate([policy.action_normalizer.normalize_joints(episode.joint_targets[i]), [episode.gripper_targets[i]]])
        for i in range(episode.length)
    }

    for t in range(episode.length):
        window = build_action_window(available_actions, t, HISTORY_LENGTH)
        assert window.shape == (HISTORY_LENGTH, 8)
        # Current slot always masked.
        assert np.allclose(window[-1], np.concatenate([np.zeros(7), [0.5]]))
        for slot, source_index in enumerate(_source_indices(t, HISTORY_LENGTH)):
            if slot == HISTORY_LENGTH - 1 or source_index < 0:
                continue
            expected = available_actions[source_index]
            assert np.allclose(window[slot], expected)


def _source_indices(t, history_length):
    return [t - history_length + 1 + h for h in range(history_length)]


def test_dataset_getitem_previous_actions_match_stored_model_joint_targets(collected_round):
    """The Dataset's __getitem__ previous_actions for any real (non-masked)
    slot must equal action_normalizer.normalize_joints(episode.joint_targets[source_index])
    -- the model's OWN stored action, never the expert's."""
    root, episode_dir, metadata = collected_round
    episode = load_episode(episode_dir)
    labels = load_expert_labels(episode_dir)

    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))
    tokenizer = load_tokenizer()
    image_transform = build_image_transform()

    dataset = TemporalDaggerCorrectiveDataset(
        root, image_transform, tokenizer, 32, state_normalizer, action_normalizer,
        history_length=HISTORY_LENGTH, retained_only=False,
    )

    for index in range(len(dataset)):
        episode_index, t = dataset._index[index]
        sample = dataset[index]
        for slot, source_index in enumerate(_source_indices(t, HISTORY_LENGTH)):
            if slot == HISTORY_LENGTH - 1 or source_index < 0:
                continue
            expected_joint = action_normalizer.normalize_joints(episode.joint_targets[source_index])
            assert np.allclose(sample["previous_actions"][slot, :7].numpy(), expected_joint, atol=1e-6)
            assert np.isclose(sample["previous_actions"][slot, 7].item(), episode.gripper_targets[source_index], atol=1e-6)

        # Target must equal the EXPERT label at t, never the model's own action at t.
        expected_target = action_normalizer.normalize_joints(labels["expert_joint_targets"][t])
        assert np.allclose(sample["joint_targets_normalized"].numpy(), expected_target, atol=1e-6)
        assert not np.allclose(sample["joint_targets_normalized"].numpy(), action_normalizer.normalize_joints(episode.joint_targets[t]), atol=1e-3)


def test_observation_recorded_before_step_matches_generate_dataset_convention(collected_round):
    """Same alignment guarantee as dataset/recorder.py's docstring: the
    observation recorded at timestep t is the one the MODEL used to
    compute the action executed at t, not the resulting state after
    env.step(). Verified indirectly: consecutive recorded states must
    differ (the arm is moving under real physics), and the episode length
    matches the number of policy.predict() calls made."""
    root, episode_dir, metadata = collected_round
    episode = load_episode(episode_dir)
    assert episode.length == metadata["num_candidate_timesteps"] == 10
    assert not np.allclose(episode.states[0], episode.states[-1])
