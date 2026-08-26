"""Temporal checkpoint save/reload must reproduce predictions and preserve history_length."""

import numpy as np
import torch

from models.language_encoder import load_tokenizer
from models.temporal_vla import TemporalDenseVLA, TemporalDenseVLAConfig
from training.checkpoint import load_checkpoint, save_checkpoint
from training.normalization import ActionNormalizer, StateNormalizer

HISTORY_LENGTH = 4


def make_batch(config, tokenizer, batch_size=2):
    tokenized = tokenizer(
        ["Pick up the red cube."] * batch_size, return_tensors="pt", padding="max_length",
        truncation=True, max_length=config.max_instruction_length,
    )
    return {
        "pixel_values": torch.randn(batch_size, config.history_length, 3, 224, 224),
        "states": torch.randn(batch_size, config.history_length, config.state_dim),
        "previous_actions": torch.randn(batch_size, config.history_length, 8),
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def test_save_and_reload_reproduces_predictions(tmp_path):
    config = TemporalDenseVLAConfig(hidden_dim=32, num_layers=2, num_heads=4, ffn_dim=64, history_length=HISTORY_LENGTH)
    model = TemporalDenseVLA(config)
    model.eval()
    tokenizer = load_tokenizer(config.language_backbone)
    batch = make_batch(config, tokenizer)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    with torch.no_grad():
        original_output = model(
            batch["pixel_values"], batch["states"], batch["previous_actions"],
            batch["input_ids"], batch["attention_mask"],
        )

    checkpoint_path = tmp_path / "temporal_test.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, epoch=4, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer, val_metrics={"joint_mae": 0.01},
    )

    checkpoint = load_checkpoint(checkpoint_path)
    restored_config = TemporalDenseVLAConfig.from_dict(checkpoint["config"])
    restored_model = TemporalDenseVLA(restored_config)
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_model.eval()

    with torch.no_grad():
        restored_output = restored_model(
            batch["pixel_values"], batch["states"], batch["previous_actions"],
            batch["input_ids"], batch["attention_mask"],
        )

    assert torch.allclose(
        original_output["joint_targets_normalized"], restored_output["joint_targets_normalized"], atol=1e-6
    )
    assert torch.allclose(original_output["gripper_logit"], restored_output["gripper_logit"], atol=1e-6)
    assert restored_config.history_length == HISTORY_LENGTH
    assert checkpoint["epoch"] == 4


def test_checkpoint_preserves_history_length(tmp_path):
    config = TemporalDenseVLAConfig(hidden_dim=32, num_layers=2, num_heads=4, ffn_dim=64, history_length=6)
    model = TemporalDenseVLA(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    checkpoint_path = tmp_path / "temporal_hl_test.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, epoch=0, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer, val_metrics={},
    )

    checkpoint = load_checkpoint(checkpoint_path)
    restored_config = TemporalDenseVLAConfig.from_dict(checkpoint["config"])
    assert restored_config.history_length == 6
