"""Tests for the DAgger training pipeline: aggregated-dataset sample
accounting (dagger/aggregation.py) and a tiny corrective-overfit sanity
check (README "DAgger tiny sanity test") proving corrective samples
actually reach the model and reduce loss.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from control.action import RobotAction
from dagger.aggregation import build_aggregated_dataloader
from dagger.dataset import TemporalDaggerCorrectiveDataset
from dataset.recorder import EpisodeRecorder
from models.language_encoder import load_tokenizer
from models.temporal_vla import TemporalDenseVLA, TemporalDenseVLAConfig
from models.vision_encoder import build_image_transform
from observations.observation import Observation
from training.losses import compute_loss
from training.normalization import ActionNormalizer, StateNormalizer

HISTORY_LENGTH = 4


class _TinyDataset(Dataset):
    """Minimal stand-in for an expert/dagger dataset -- constant-valued
    samples, just enough shape to exercise the sampler/loader plumbing."""

    def __init__(self, n: int, tag: float) -> None:
        self.n = n
        self.tag = tag

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict:
        return {"tag": torch.tensor(self.tag)}


def test_aggregated_dataloader_reports_exact_sample_accounting():
    expert = _TinyDataset(100, tag=0.0)
    dagger = _TinyDataset(20, tag=1.0)

    loader, stats = build_aggregated_dataloader(expert, dagger, batch_size=8, expert_weight=0.5, dagger_weight=0.5)

    assert stats.num_expert_samples == 100
    assert stats.num_dagger_samples == 20
    stats_dict = stats.to_dict()
    assert stats_dict["dagger_to_expert_ratio"] == pytest.approx(0.2)
    assert stats_dict["num_expert_samples"] == 100
    assert stats_dict["num_dagger_samples"] == 20


def test_aggregated_dataloader_mixes_roughly_at_requested_ratio():
    """With very different raw sizes (100 vs 20), a 50/50 weighted sampler
    should still draw close to half its batches from each source over many
    samples -- not the raw 100:20 ratio."""
    torch.manual_seed(0)
    expert = _TinyDataset(400, tag=0.0)
    dagger = _TinyDataset(20, tag=1.0)

    loader, _ = build_aggregated_dataloader(
        expert, dagger, batch_size=32, expert_weight=0.5, dagger_weight=0.5, num_samples_per_epoch=3200,
    )
    tags = torch.cat([batch["tag"] for batch in loader])
    dagger_fraction = tags.mean().item()  # fraction with tag==1.0 (dagger)
    assert 0.35 < dagger_fraction < 0.65


def test_build_aggregated_dataloader_rejects_empty_dagger_dataset():
    expert = _TinyDataset(10, tag=0.0)
    dagger = _TinyDataset(0, tag=1.0)
    try:
        build_aggregated_dataloader(expert, dagger, batch_size=4)
        assert False, "expected ValueError for empty dagger dataset"
    except ValueError:
        pass


# --- DAgger tiny corrective-overfit sanity check --------------------------


def _make_observation(t: int) -> Observation:
    rgb = np.random.default_rng(t).integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    state = np.random.default_rng(t + 100).normal(size=23)
    return Observation(rgb=rgb, state=state, timestamp=float(t) * 0.02)


def _write_synthetic_dagger_episode(episodes_root, episode_id, length, rng):
    """A DAgger episode with genuinely distinguishable, non-trivial
    model-vs-expert actions (random but fixed), so overfitting the
    corrective target is a real learning problem, not a constant fit."""
    recorder = EpisodeRecorder(episodes_root, episode_id)
    model_joint_targets, model_gripper_targets = [], []
    for t in range(length):
        obs = _make_observation(episode_id * 1000 + t)
        joint_targets = rng.normal(scale=0.1, size=7)
        gripper_target = float(rng.integers(0, 2))
        model_joint_targets.append(joint_targets)
        model_gripper_targets.append(gripper_target)
        recorder.record(obs, RobotAction(joint_targets=joint_targets, gripper_target=gripper_target), "TEST")
    episode_dir = recorder.finalize(episodes_root, "Pick up the red cube.", True, {})

    expert_joint_targets = np.stack([rng.normal(scale=0.1, size=7) + 0.5 for _ in range(length)])  # offset from model
    expert_gripper_targets = np.array([float(1 - g) for g in model_gripper_targets])  # deliberately different
    np.savez(
        episode_dir / "expert_labels.npz",
        expert_joint_targets=expert_joint_targets,
        expert_gripper_targets=expert_gripper_targets,
        joint_l2_disagreement=np.ones(length),
        joint_mae_disagreement=np.ones(length),
        gripper_disagreement=np.ones(length, dtype=bool),
        retained=np.ones(length, dtype=bool),
        cube_position=np.zeros((length, 3)),
    )
    return episode_dir.name


def test_tiny_corrective_overfit_reduces_loss_substantially(tmp_path):
    rng = np.random.default_rng(0)
    episodes_root = tmp_path / "episodes"
    for episode_id in range(3):
        _write_synthetic_dagger_episode(episodes_root, episode_id, length=8, rng=rng)

    config = TemporalDenseVLAConfig(hidden_dim=32, num_layers=2, num_heads=4, ffn_dim=64, history_length=HISTORY_LENGTH)
    model = TemporalDenseVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)
    image_transform = build_image_transform()
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    dataset = TemporalDaggerCorrectiveDataset(
        tmp_path, image_transform, tokenizer, config.max_instruction_length,
        state_normalizer, action_normalizer, history_length=HISTORY_LENGTH, retained_only=True,
    )
    assert len(dataset) == 24  # 3 episodes x 8 retained ticks each

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    def _epoch_loss():
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                prediction = model(
                    batch["pixel_values"], batch["states"], batch["previous_actions"],
                    batch["input_ids"], batch["attention_mask"],
                )
                _, metrics = compute_loss(prediction, batch)
                total += metrics["loss"] * batch["states"].shape[0]
                n += batch["states"].shape[0]
        return total / n

    initial_loss = _epoch_loss()

    model.train()
    for _ in range(150):
        for batch in loader:
            optimizer.zero_grad()
            prediction = model(
                batch["pixel_values"], batch["states"], batch["previous_actions"],
                batch["input_ids"], batch["attention_mask"],
            )
            loss, _ = compute_loss(prediction, batch)
            loss.backward()
            optimizer.step()

    final_loss = _epoch_loss()
    assert final_loss < initial_loss * 0.5, f"expected substantial loss reduction, got {initial_loss} -> {final_loss}"
