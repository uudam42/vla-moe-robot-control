"""Tests for training/checkpoint.py: save/reload must reproduce predictions."""

import numpy as np
import torch

from models.dense_vla import DenseVLA, DenseVLAConfig
from models.language_encoder import load_tokenizer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.normalization import ActionNormalizer, StateNormalizer


def make_model_and_batch(config, tokenizer, batch_size=2):
    model = DenseVLA(config)
    tokenized = tokenizer(
        ["Pick up the red cube."] * batch_size,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.max_instruction_length,
    )
    batch = {
        "pixel_values": torch.randn(batch_size, 3, 224, 224),
        "state": torch.randn(batch_size, config.state_dim),
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }
    return model, batch


def test_save_and_reload_reproduces_predictions(tmp_path):
    config = DenseVLAConfig(hidden_dim=64, num_layers=2, num_heads=4, ffn_dim=128)
    tokenizer = load_tokenizer(config.language_backbone)
    model, batch = make_model_and_batch(config, tokenizer)
    model.eval()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    with torch.no_grad():
        original_output = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])

    checkpoint_path = tmp_path / "test.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, epoch=3, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer,
        val_metrics={"joint_mae": 0.1},
    )

    checkpoint = load_checkpoint(checkpoint_path)
    restored_config = DenseVLAConfig.from_dict(checkpoint["config"])
    restored_model = DenseVLA(restored_config)
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_model.eval()

    with torch.no_grad():
        restored_output = restored_model(
            batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"]
        )

    assert torch.allclose(
        original_output["joint_targets_normalized"], restored_output["joint_targets_normalized"], atol=1e-6
    )
    assert torch.allclose(original_output["gripper_logit"], restored_output["gripper_logit"], atol=1e-6)

    assert checkpoint["epoch"] == 3
    assert checkpoint["val_metrics"]["joint_mae"] == 0.1
    restored_state_normalizer = StateNormalizer.from_dict(checkpoint["state_normalizer"])
    restored_action_normalizer = ActionNormalizer.from_dict(checkpoint["action_normalizer"])
    assert np.allclose(restored_state_normalizer.mean, state_normalizer.mean)
    assert np.allclose(restored_action_normalizer.joint_mean, action_normalizer.joint_mean)


def test_optimizer_state_reloads_and_resumes(tmp_path):
    config = DenseVLAConfig(hidden_dim=64, num_layers=2, num_heads=4, ffn_dim=128)
    tokenizer = load_tokenizer(config.language_backbone)
    model, batch = make_model_and_batch(config, tokenizer)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    # Take one optimizer step so it has real internal state (Adam moment buffers).
    output = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
    loss = output["joint_targets_normalized"].pow(2).mean()
    loss.backward()
    optimizer.step()

    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))
    checkpoint_path = tmp_path / "resume_test.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, epoch=7, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer, val_metrics={},
    )

    checkpoint = load_checkpoint(checkpoint_path)
    new_model = DenseVLA(config)
    new_model.load_state_dict(checkpoint["model_state_dict"])
    new_optimizer = torch.optim.AdamW([p for p in new_model.parameters() if p.requires_grad], lr=1e-3)
    new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    assert checkpoint["epoch"] == 7
    assert len(new_optimizer.state) > 0
