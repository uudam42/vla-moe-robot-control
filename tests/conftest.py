"""Shared fixtures for Step 5 closed-loop tests.

A tiny-config DenseVLA checkpoint is expensive to build (loading the fixed
pretrained ResNet18 + DistilBERT backbones dominates construction time,
independent of hidden_dim/num_layers) -- session-scoped so every Step 5
test file reuses the same on-disk checkpoint instead of rebuilding one.
"""

import numpy as np
import pytest
import torch

from models.dense_vla import DenseVLA, DenseVLAConfig
from models.moe_vla import MoEVLA, MoEVLAConfig
from models.temporal_vla import TemporalDenseVLA, TemporalDenseVLAConfig
from training.checkpoint import save_checkpoint
from training.normalization import ActionNormalizer, StateNormalizer


@pytest.fixture(scope="session")
def tiny_vla_checkpoint(tmp_path_factory):
    """A small, untrained (random-weight) DenseVLA checkpoint -- fast to
    build, sufficient to exercise the closed-loop runtime plumbing. Not
    expected to succeed at the task; only used where the tests care about
    "does the loop run correctly", not "is the policy good"."""
    config = DenseVLAConfig(hidden_dim=64, num_layers=2, num_heads=4, ffn_dim=128, dropout=0.0)
    model = DenseVLA(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    path = tmp_path_factory.mktemp("checkpoints") / "tiny_vla.pt"
    save_checkpoint(
        path, model, optimizer, epoch=0, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer, val_metrics={},
    )
    return path


@pytest.fixture(scope="session")
def tiny_moe_vla_checkpoint(tmp_path_factory):
    """Small, untrained (random-weight) MoEVLA checkpoint -- same role as
    ``tiny_vla_checkpoint`` but for Step 6 MoE runtime tests."""
    config = MoEVLAConfig(
        hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64, dropout=0.0,
        num_experts=4, top_k=1, moe_layers=(1, 3),
    )
    model = MoEVLA(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    path = tmp_path_factory.mktemp("checkpoints") / "tiny_moe_vla.pt"
    save_checkpoint(
        path, model, optimizer, epoch=0, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer, val_metrics={},
    )
    return path


@pytest.fixture(scope="session")
def tiny_temporal_vla_checkpoint(tmp_path_factory):
    """Small, untrained (random-weight) TemporalDenseVLA checkpoint -- same
    role as ``tiny_vla_checkpoint``/``tiny_moe_vla_checkpoint`` but for
    Step 7 temporal runtime tests. history_length=4."""
    config = TemporalDenseVLAConfig(
        hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64, dropout=0.0, history_length=4,
    )
    model = TemporalDenseVLA(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    path = tmp_path_factory.mktemp("checkpoints") / "tiny_temporal_vla.pt"
    save_checkpoint(
        path, model, optimizer, epoch=0, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer, val_metrics={},
    )
    return path
