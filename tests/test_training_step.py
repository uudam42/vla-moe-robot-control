"""Forward/backward smoke test: one real optimizer step must actually work."""

import pytest
import torch

from models.dense_vla import DenseVLA, DenseVLAConfig
from models.language_encoder import load_tokenizer
from training.losses import compute_loss


def test_one_training_step_produces_finite_loss_and_updates_weights():
    config = DenseVLAConfig(hidden_dim=64, num_layers=2, num_heads=4, ffn_dim=128)
    model = DenseVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)

    batch_size = 4
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
        "joint_targets_normalized": torch.randn(batch_size, 7),
        "gripper_target": torch.randint(0, 2, (batch_size,)).float(),
    }

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    params_before = [p.detach().clone() for p in trainable_params]

    prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
    loss, metrics = compute_loss(prediction, batch)

    assert torch.isfinite(loss)
    assert metrics["loss"] == pytest.approx(metrics["joint_loss"] + metrics["gripper_loss"], abs=1e-6)

    optimizer.zero_grad()
    loss.backward()

    assert any(p.grad is not None and torch.any(p.grad != 0) for p in trainable_params)

    optimizer.step()

    assert any(
        not torch.equal(before, after) for before, after in zip(params_before, trainable_params)
    )


def test_frozen_encoder_parameters_receive_no_gradient():
    config = DenseVLAConfig(hidden_dim=64, num_layers=2, num_heads=4, ffn_dim=128)
    model = DenseVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)

    tokenized = tokenizer(
        ["Pick up the red cube."], return_tensors="pt", padding="max_length", truncation=True, max_length=32
    )
    batch = {
        "pixel_values": torch.randn(1, 3, 224, 224),
        "state": torch.randn(1, config.state_dim),
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "joint_targets_normalized": torch.randn(1, 7),
        "gripper_target": torch.tensor([1.0]),
    }

    prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
    loss, _ = compute_loss(prediction, batch)
    loss.backward()

    for param in model.vision_encoder.parameters():
        assert param.grad is None
    for param in model.language_encoder.parameters():
        assert param.grad is None
