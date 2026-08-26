"""Tests for dataset/torch_dataset.py::DemonstrationTorchDataset."""

import numpy as np
import pytest
import torch

from control.action import RobotAction
from dataset.loader import DemonstrationDataset
from dataset.recorder import EpisodeRecorder
from dataset.torch_dataset import DemonstrationTorchDataset
from models.language_encoder import load_tokenizer
from models.vision_encoder import build_image_transform
from observations.observation import Observation
from training.normalization import ActionNormalizer, StateNormalizer

MAX_LENGTH = 32


def make_observation(t: int) -> Observation:
    rgb = np.full((16, 16, 3), t % 256, dtype=np.uint8)
    state = np.linspace(0, 1, 23) + t * 0.01
    return Observation(rgb=rgb, state=state, timestamp=float(t) * 0.02)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def image_transform():
    return build_image_transform()


@pytest.fixture
def wrapped_dataset(tmp_path, tokenizer, image_transform):
    recorder = EpisodeRecorder(tmp_path, episode_id=0)
    for t in range(6):
        recorder.record(
            make_observation(t),
            RobotAction(joint_targets=np.full(7, 0.1 * t), gripper_target=1.0 if t % 2 == 0 else 0.0),
            "TEST",
        )
    recorder.finalize(tmp_path / "successful", "Pick up the red cube.", True, {})

    base_dataset = DemonstrationDataset(tmp_path)
    state_normalizer = StateNormalizer.fit(np.stack([make_observation(t).state for t in range(6)]))
    action_normalizer = ActionNormalizer.fit(np.stack([np.full(7, 0.1 * t) for t in range(6)]))

    return DemonstrationTorchDataset(
        base_dataset, image_transform, tokenizer, MAX_LENGTH, state_normalizer, action_normalizer
    )


def test_len_matches_base_dataset(wrapped_dataset):
    assert len(wrapped_dataset) == 6


def test_sample_shapes_and_dtypes(wrapped_dataset):
    sample = wrapped_dataset[0]

    assert sample["pixel_values"].shape == (3, 224, 224)
    assert sample["pixel_values"].dtype == torch.float32
    assert sample["state"].shape == (23,)
    assert sample["input_ids"].shape == (MAX_LENGTH,)
    assert sample["attention_mask"].shape == (MAX_LENGTH,)
    assert sample["joint_targets_normalized"].shape == (7,)
    assert sample["gripper_target"].shape == ()
    assert isinstance(sample["instruction"], str)


def test_state_is_normalized(wrapped_dataset):
    sample = wrapped_dataset[3]
    # Normalized state should not equal the raw state (unless std happens to be 1 and mean 0).
    assert not torch.allclose(sample["state"], torch.zeros(23))
    assert torch.all(torch.isfinite(sample["state"]))


def test_gripper_target_matches_recorded_value(wrapped_dataset):
    assert wrapped_dataset[0]["gripper_target"].item() == 1.0
    assert wrapped_dataset[1]["gripper_target"].item() == 0.0


def test_batching_via_dataloader_produces_consistent_shapes(wrapped_dataset):
    loader = torch.utils.data.DataLoader(wrapped_dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["pixel_values"].shape == (4, 3, 224, 224)
    assert batch["state"].shape == (4, 23)
    assert batch["input_ids"].shape == (4, MAX_LENGTH)
    assert batch["joint_targets_normalized"].shape == (4, 7)
    assert batch["gripper_target"].shape == (4,)
    assert len(batch["instruction"]) == 4
