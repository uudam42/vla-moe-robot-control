"""Tests for dagger/dataset.py::TemporalDaggerCorrectiveDataset -- shapes,
retention filtering, and (most importantly) that the previous-action window
uses the MODEL's own issued actions while the training target is the
EXPERT's corrective label, with no future/target/cross-episode leakage.

MuJoCo-free: episodes are synthesized directly via the same
``dataset.recorder.EpisodeRecorder`` DAgger collection uses, with a
hand-built ``expert_labels.npz`` sidecar (mirroring
``tests/test_temporal_dataset.py``'s style for the non-DAgger dataset).
"""

import numpy as np
import pytest

from control.action import RobotAction
from dagger.dataset import TemporalDaggerCorrectiveDataset
from dataset.recorder import EpisodeRecorder
from models.language_encoder import load_tokenizer
from models.temporal_history import NO_ACTION_VECTOR
from models.vision_encoder import build_image_transform
from observations.observation import Observation
from training.normalization import ActionNormalizer, StateNormalizer

HISTORY_LENGTH = 4
MAX_LENGTH = 32
MODEL_OFFSET = 0.0
EXPERT_OFFSET = 5000.0  # far away from MODEL_OFFSET -- leakage/mixup is unmistakable


def make_observation(t: int, episode_offset: float = 0.0) -> Observation:
    rgb = np.full((8, 8, 3), t % 256, dtype=np.uint8)
    state = np.full(23, episode_offset + t, dtype=np.float64)
    return Observation(rgb=rgb, state=state, timestamp=float(t) * 0.02)


def write_dagger_episode(
    episodes_root, episode_id, length, episode_offset=0.0, retained_mod=1, instruction="Pick up the red cube."
):
    """Model-issued action at t == MODEL_OFFSET + episode_offset + t;
    expert corrective action at t == EXPERT_OFFSET + episode_offset + t --
    a sample's origin (model vs expert) is unambiguous from its value."""
    recorder = EpisodeRecorder(episodes_root, episode_id)
    for t in range(length):
        obs = make_observation(t, episode_offset)
        model_action = RobotAction(
            joint_targets=np.full(7, MODEL_OFFSET + episode_offset + t), gripper_target=1.0 if t % 2 == 0 else 0.0
        )
        recorder.record(obs, model_action, "TEST")
    episode_dir = recorder.finalize(episodes_root, instruction, True, {})

    expert_joint_targets = np.stack([np.full(7, EXPERT_OFFSET + episode_offset + t) for t in range(length)])
    expert_gripper_targets = np.array([0.0 if t % 2 == 0 else 1.0 for t in range(length)])  # deliberately inverted vs model
    retained = np.array([(t % retained_mod) == 0 for t in range(length)])
    np.savez(
        episode_dir / "expert_labels.npz",
        expert_joint_targets=expert_joint_targets,
        expert_gripper_targets=expert_gripper_targets,
        joint_l2_disagreement=np.zeros(length),
        joint_mae_disagreement=np.zeros(length),
        gripper_disagreement=np.zeros(length, dtype=bool),
        retained=retained,
        cube_position=np.zeros((length, 3)),
    )
    return episode_dir.name


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def image_transform():
    return build_image_transform()


@pytest.fixture
def normalizers():
    return StateNormalizer(mean=np.zeros(23), std=np.ones(23)), ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))


def _build(root, tokenizer, image_transform, normalizers, **kwargs):
    state_normalizer, action_normalizer = normalizers
    return TemporalDaggerCorrectiveDataset(
        root, image_transform, tokenizer, MAX_LENGTH, state_normalizer, action_normalizer,
        history_length=HISTORY_LENGTH, **kwargs,
    )


def test_retained_only_filters_to_retained_timesteps(tmp_path, tokenizer, image_transform, normalizers):
    episodes_root = tmp_path / "episodes"
    write_dagger_episode(episodes_root, 0, length=10, retained_mod=3)  # retained at t=0,3,6,9 -> 4 samples

    dataset = _build(tmp_path, tokenizer, image_transform, normalizers, retained_only=True)
    assert len(dataset) == 4

    dataset_all = _build(tmp_path, tokenizer, image_transform, normalizers, retained_only=False)
    assert len(dataset_all) == 10


def test_sample_shapes(tmp_path, tokenizer, image_transform, normalizers):
    episodes_root = tmp_path / "episodes"
    write_dagger_episode(episodes_root, 0, length=8, retained_mod=1)
    dataset = _build(tmp_path, tokenizer, image_transform, normalizers)
    sample = dataset[0]
    assert sample["pixel_values"].shape == (HISTORY_LENGTH, 3, 224, 224)
    assert sample["states"].shape == (HISTORY_LENGTH, 23)
    assert sample["previous_actions"].shape == (HISTORY_LENGTH, 8)
    assert sample["joint_targets_normalized"].shape == (7,)
    assert sample["gripper_target"].shape == ()


def test_target_is_expert_action_not_model_action(tmp_path, tokenizer, image_transform, normalizers):
    episodes_root = tmp_path / "episodes"
    write_dagger_episode(episodes_root, 0, length=8, retained_mod=1)
    dataset = _build(tmp_path, tokenizer, image_transform, normalizers)

    sample = dataset[5]  # t=5
    target_joint = sample["joint_targets_normalized"].numpy()
    assert np.allclose(target_joint, EXPERT_OFFSET + 5.0)
    assert not np.allclose(target_joint, MODEL_OFFSET + 5.0)
    # gripper target: expert's inverted convention (1.0 at odd t)
    assert sample["gripper_target"].item() == pytest.approx(1.0)


def test_previous_actions_use_model_issued_history_not_expert(tmp_path, tokenizer, image_transform, normalizers):
    episodes_root = tmp_path / "episodes"
    write_dagger_episode(episodes_root, 0, length=8, retained_mod=1)
    dataset = _build(tmp_path, tokenizer, image_transform, normalizers)

    sample = dataset[5]  # t=5: history covers t=2,3,4 (real, model-issued), t=5 masked (current)
    assert np.allclose(sample["previous_actions"][0, :7].numpy(), MODEL_OFFSET + 2.0)
    assert np.allclose(sample["previous_actions"][1, :7].numpy(), MODEL_OFFSET + 3.0)
    assert np.allclose(sample["previous_actions"][2, :7].numpy(), MODEL_OFFSET + 4.0)
    # None of the real history slots may equal the EXPERT's action value.
    for slot in range(3):
        assert not np.allclose(sample["previous_actions"][slot, :7].numpy(), EXPERT_OFFSET + 2.0 + slot)


def test_first_timestep_has_all_masked_previous_actions(tmp_path, tokenizer, image_transform, normalizers):
    episodes_root = tmp_path / "episodes"
    write_dagger_episode(episodes_root, 0, length=8, retained_mod=1)
    dataset = _build(tmp_path, tokenizer, image_transform, normalizers)
    sample = dataset[0]
    for slot in range(HISTORY_LENGTH):
        assert np.allclose(sample["previous_actions"][slot].numpy(), NO_ACTION_VECTOR)


def test_no_future_or_target_leakage_exhaustive(tmp_path, tokenizer, image_transform, normalizers):
    episodes_root = tmp_path / "episodes"
    write_dagger_episode(episodes_root, 0, length=10, retained_mod=1)
    dataset = _build(tmp_path, tokenizer, image_transform, normalizers)

    for index in range(len(dataset)):
        sample = dataset[index]
        # Current/target slot must always be masked (never the model's own
        # action at t, and never the expert's -- that IS the target).
        assert np.allclose(sample["previous_actions"][-1, :7].numpy(), 0.0)
        # No history slot's action value may exceed t (i.e. reference a
        # future model action) -- history slot h corresponds to model
        # action MODEL_OFFSET + (t - H + 1 + h); by construction this is
        # always <= MODEL_OFFSET + t - 1 for h < H-1, or the sentinel.
        target = sample["joint_targets_normalized"][0].item()  # == EXPERT_OFFSET + t
        implied_t = target - EXPERT_OFFSET
        for slot in range(HISTORY_LENGTH - 1):
            value = sample["previous_actions"][slot, 0].item()
            is_masked = np.isclose(value, 0.0)
            if not is_masked:
                assert value < MODEL_OFFSET + implied_t


def test_no_cross_episode_leakage(tmp_path, tokenizer, image_transform, normalizers):
    episodes_root = tmp_path / "episodes"
    write_dagger_episode(episodes_root, 0, length=8, episode_offset=0.0, retained_mod=1)
    write_dagger_episode(episodes_root, 1, length=6, episode_offset=1_000_000.0, retained_mod=1)
    dataset = _build(tmp_path, tokenizer, image_transform, normalizers)

    assert len(dataset) == 14
    global_index_of_ep1_t0 = 8
    sample = dataset[global_index_of_ep1_t0]
    for slot in range(HISTORY_LENGTH):
        assert np.allclose(sample["previous_actions"][slot].numpy(), NO_ACTION_VECTOR)
    assert np.all(sample["states"].numpy() >= 999_999.0)


def test_episode_subset_selects_only_named_episodes(tmp_path, tokenizer, image_transform, normalizers):
    episodes_root = tmp_path / "episodes"
    name0 = write_dagger_episode(episodes_root, 0, length=5, retained_mod=1)
    write_dagger_episode(episodes_root, 1, length=5, retained_mod=1)

    dataset = _build(tmp_path, tokenizer, image_transform, normalizers, episode_names=[name0])
    assert dataset.num_episodes == 1
    assert len(dataset) == 5


def test_instruction_is_returned(tmp_path, tokenizer, image_transform, normalizers):
    episodes_root = tmp_path / "episodes"
    write_dagger_episode(episodes_root, 0, length=4, retained_mod=1, instruction="Grasp the red cube.")
    dataset = _build(tmp_path, tokenizer, image_transform, normalizers)
    assert dataset[0]["instruction"] == "Grasp the red cube."
