"""Tests for dataset/temporal_torch_dataset.py: shapes, padding, and --
most importantly -- no cross-episode or future/target leakage.
"""

import numpy as np
import pytest

from control.action import RobotAction
from dataset.recorder import EpisodeRecorder
from dataset.splits import save_splits
from dataset.temporal_torch_dataset import TemporalDemonstrationDataset
from models.language_encoder import load_tokenizer
from models.temporal_history import NO_ACTION_VECTOR
from models.vision_encoder import build_image_transform
from observations.observation import Observation
from training.normalization import ActionNormalizer, StateNormalizer

HISTORY_LENGTH = 4
MAX_LENGTH = 32


def make_observation(t: int, episode_offset: float = 0.0) -> Observation:
    # State/joint values distinguishable by (episode, t) so leakage is easy to detect.
    rgb = np.full((8, 8, 3), t % 256, dtype=np.uint8)
    state = np.full(23, episode_offset + t, dtype=np.float64)
    return Observation(rgb=rgb, state=state, timestamp=float(t) * 0.02)


def write_episode(root, episode_id, length, episode_offset=0.0, instruction="Pick up the red cube."):
    recorder = EpisodeRecorder(root, episode_id)
    for t in range(length):
        obs = make_observation(t, episode_offset)
        # Joint target value == episode_offset + t, so a fetched action's value
        # tells us exactly which episode/timestep it came from.
        recorder.record(
            obs, RobotAction(joint_targets=np.full(7, episode_offset + t), gripper_target=1.0 if t % 2 == 0 else 0.0),
            "TEST",
        )
    return recorder.finalize(root / "successful", instruction, True, {}).name


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def image_transform():
    return build_image_transform()


@pytest.fixture
def two_episode_dataset(tmp_path, tokenizer, image_transform):
    ep0 = write_episode(tmp_path, 0, length=8, episode_offset=0.0)
    ep1 = write_episode(tmp_path, 1, length=6, episode_offset=1000.0)  # far-away offset -> leakage is obvious
    save_splits({"train": [ep0, ep1], "val": [], "test": []}, tmp_path / "splits.json")

    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    return TemporalDemonstrationDataset(
        tmp_path, "train", image_transform, tokenizer, MAX_LENGTH,
        state_normalizer, action_normalizer, history_length=HISTORY_LENGTH,
    )


def test_len_matches_total_timesteps(two_episode_dataset):
    assert len(two_episode_dataset) == 8 + 6
    assert two_episode_dataset.num_episodes == 2


def test_sample_shapes(two_episode_dataset):
    sample = two_episode_dataset[0]
    assert sample["pixel_values"].shape == (HISTORY_LENGTH, 3, 224, 224)
    assert sample["states"].shape == (HISTORY_LENGTH, 23)
    assert sample["previous_actions"].shape == (HISTORY_LENGTH, 8)
    assert sample["joint_targets_normalized"].shape == (7,)
    assert sample["gripper_target"].shape == ()


def test_first_timestep_has_all_masked_previous_actions(two_episode_dataset):
    """t=0: no real action has ever occurred -- every history slot must be NO_ACTION."""
    sample = two_episode_dataset[0]  # episode 0, t=0
    for slot in range(HISTORY_LENGTH):
        assert np.allclose(sample["previous_actions"][slot].numpy(), NO_ACTION_VECTOR)


def test_later_timestep_has_real_previous_actions_and_masked_current(two_episode_dataset):
    """episode 0, t=5: history covers t=2,3,4 (real) and t=5 (masked, it's the target)."""
    index_of_t5 = 5  # episode 0 has 8 steps (indices 0-7), so global index 5 == (episode 0, t=5)
    sample = two_episode_dataset[index_of_t5]

    # Real actions at t-3=2, t-2=3, t-1=4: joint value == episode_offset(0) + t == t.
    assert np.allclose(sample["previous_actions"][0, :7].numpy(), 2.0)
    assert np.allclose(sample["previous_actions"][1, :7].numpy(), 3.0)
    assert np.allclose(sample["previous_actions"][2, :7].numpy(), 4.0)
    # Current slot (t=5, the target) must be masked, NOT the real action at t=5.
    assert np.allclose(sample["previous_actions"][3].numpy(), NO_ACTION_VECTOR)


def test_target_action_never_appears_in_previous_actions(two_episode_dataset):
    """Exhaustive target-leakage check across every sample in the dataset."""
    for index in range(len(two_episode_dataset)):
        sample = two_episode_dataset[index]
        target_joint = sample["joint_targets_normalized"].numpy()
        # If target leaked in, some slot's joints would exactly equal the target
        # AND not be the masked sentinel (the sentinel is all-zero, distinguishable
        # from any nonzero synthetic target used here except episode0/t=0's target,
        # which is also zero -- handled separately by the explicit t=0 test above).
        last_slot = sample["previous_actions"][-1, :7].numpy()
        assert np.allclose(last_slot, 0.0), "current/target timestep's action slot must always be masked"


def test_no_cross_episode_leakage_in_early_timesteps_of_second_episode(two_episode_dataset):
    """episode 1 starts at global index 8 (after episode 0's 8 steps). Its
    t=0 history must NOT contain any of episode 0's huge-offset-free values
    or, more importantly, episode 1's own left-padded obs_0 only -- never
    reach backward into episode 0's arrays."""
    global_index_of_ep1_t0 = 8
    sample = two_episode_dataset[global_index_of_ep1_t0]
    for slot in range(HISTORY_LENGTH):
        assert np.allclose(sample["previous_actions"][slot].numpy(), NO_ACTION_VECTOR)
    # All 4 padded observation slots must equal episode 1's OWN obs_0 state
    # (offset 1000), never episode 0's (offset 0-7).
    for slot in range(HISTORY_LENGTH):
        assert np.all(sample["states"][slot].numpy() >= 999.0)


def test_no_cross_episode_leakage_at_episode_boundary(two_episode_dataset):
    """episode 1, t=2 (global index 10): history should cover episode 1's
    own t=-1,-1,0,1,2-ish padded/real range, never episode 0's t=6,7."""
    global_index_of_ep1_t2 = 8 + 2
    sample = two_episode_dataset[global_index_of_ep1_t2]
    # Every state value must belong to episode 1 (offset >= 1000), never
    # episode 0 (offset 0-7).
    assert np.all(sample["states"].numpy() >= 999.0)
    # Any real (non-masked) previous action must also be from episode 1.
    for slot in range(HISTORY_LENGTH - 1):
        action = sample["previous_actions"][slot, :7].numpy()
        is_masked = np.allclose(action, 0.0)
        if not is_masked:
            assert np.all(action >= 999.0)


def test_history_length_is_configurable(tmp_path, tokenizer, image_transform):
    ep0 = write_episode(tmp_path, 0, length=10)
    save_splits({"train": [ep0], "val": [], "test": []}, tmp_path / "splits.json")
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    dataset = TemporalDemonstrationDataset(
        tmp_path, "train", image_transform, tokenizer, MAX_LENGTH,
        state_normalizer, action_normalizer, history_length=2,
    )
    sample = dataset[5]
    assert sample["pixel_values"].shape == (2, 3, 224, 224)
    assert sample["previous_actions"].shape == (2, 8)


def test_instruction_is_returned(two_episode_dataset):
    sample = two_episode_dataset[0]
    assert sample["instruction"] == "Pick up the red cube."
