"""Tests for dataset/loader.py (DemonstrationDataset, load_episode)."""

import numpy as np
import pytest

from control.action import RobotAction
from dataset.loader import DemonstrationDataset, load_episode
from dataset.recorder import EpisodeRecorder
from dataset.splits import save_splits
from observations.observation import Observation


def make_observation(t: int) -> Observation:
    rgb = np.full((6, 6, 3), t % 256, dtype=np.uint8)
    state = np.arange(23, dtype=np.float64) + t
    return Observation(rgb=rgb, state=state, timestamp=float(t) * 0.02)


def write_episode(root, episode_id, length, instruction="Pick up the red cube."):
    recorder = EpisodeRecorder(root, episode_id)
    for t in range(length):
        recorder.record(
            make_observation(t),
            RobotAction(joint_targets=np.full(7, float(episode_id) + 0.01 * t), gripper_target=0.25),
            "TEST",
        )
    return recorder.finalize(root / "successful", instruction, True, {})


@pytest.fixture
def small_dataset(tmp_path):
    write_episode(tmp_path, episode_id=0, length=3, instruction="Pick up the red cube.")
    write_episode(tmp_path, episode_id=1, length=5, instruction="Grasp the red cube.")
    return tmp_path


def test_len_matches_total_timesteps(small_dataset):
    dataset = DemonstrationDataset(small_dataset)
    assert len(dataset) == 3 + 5
    assert dataset.num_episodes == 2


def test_getitem_returns_expected_keys_shapes_types(small_dataset):
    dataset = DemonstrationDataset(small_dataset)
    sample = dataset[0]

    assert set(sample.keys()) == {"rgb", "state", "instruction", "action"}
    assert sample["rgb"].shape == (6, 6, 3)
    assert sample["rgb"].dtype == np.uint8
    assert sample["state"].shape == (23,)
    assert isinstance(sample["instruction"], str)
    assert sample["action"].shape == (8,)


def test_getitem_first_and_last_sample_come_from_correct_episode(small_dataset):
    dataset = DemonstrationDataset(small_dataset)

    first = dataset[0]
    assert first["instruction"] == "Pick up the red cube."
    assert np.isclose(first["state"][0], 0.0)  # episode 0, t=0

    last = dataset[len(dataset) - 1]
    assert last["instruction"] == "Grasp the red cube."
    assert np.isclose(last["state"][0], 4.0)  # episode 1, t=4 (length 5)


def test_getitem_out_of_range_raises(small_dataset):
    dataset = DemonstrationDataset(small_dataset)
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(IndexError):
        dataset[-len(dataset) - 1]


def test_negative_index_wraps(small_dataset):
    dataset = DemonstrationDataset(small_dataset)
    assert np.allclose(dataset[-1]["state"], dataset[len(dataset) - 1]["state"])


def test_load_episode_returns_expected_fields(small_dataset):
    episode_dir = small_dataset / "successful" / "episode_000001"
    episode = load_episode(episode_dir)
    assert episode.length == 5
    assert episode.instruction == "Grasp the red cube."
    assert len(episode.rgb_frame_paths) == 5


def test_split_loading_restricts_episodes(small_dataset):
    save_splits(
        {"train": ["episode_000000"], "val": ["episode_000001"], "test": []},
        small_dataset / "splits.json",
    )
    train_dataset = DemonstrationDataset(small_dataset, split="train")
    val_dataset = DemonstrationDataset(small_dataset, split="val")

    assert len(train_dataset) == 3
    assert len(val_dataset) == 5
    assert train_dataset[0]["instruction"] == "Pick up the red cube."
    assert val_dataset[0]["instruction"] == "Grasp the red cube."


def test_missing_successful_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DemonstrationDataset(tmp_path)
