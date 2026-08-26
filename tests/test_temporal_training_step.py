"""Forward/backward smoke test for TemporalDenseVLA."""

import torch

from models.language_encoder import load_tokenizer
from models.temporal_vla import TemporalDenseVLA, TemporalDenseVLAConfig
from training.losses import compute_loss

HISTORY_LENGTH = 4


def make_batch(config, tokenizer, batch_size=6):
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
        "joint_targets_normalized": torch.randn(batch_size, 7),
        "gripper_target": torch.randint(0, 2, (batch_size,)).float(),
    }


def test_one_training_step_produces_finite_loss_and_updates_weights():
    config = TemporalDenseVLAConfig(hidden_dim=32, num_layers=2, num_heads=4, ffn_dim=64, history_length=HISTORY_LENGTH)
    model = TemporalDenseVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)
    batch = make_batch(config, tokenizer)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    params_before = [p.detach().clone() for p in trainable_params]

    prediction = model(
        batch["pixel_values"], batch["states"], batch["previous_actions"],
        batch["input_ids"], batch["attention_mask"],
    )
    loss, metrics = compute_loss(prediction, batch)

    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert any(not torch.equal(before, after) for before, after in zip(params_before, trainable_params))


def test_action_history_encoder_receives_gradients():
    config = TemporalDenseVLAConfig(hidden_dim=32, num_layers=2, num_heads=4, ffn_dim=64, history_length=HISTORY_LENGTH)
    model = TemporalDenseVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)
    batch = make_batch(config, tokenizer)

    prediction = model(
        batch["pixel_values"], batch["states"], batch["previous_actions"],
        batch["input_ids"], batch["attention_mask"],
    )
    loss, _ = compute_loss(prediction, batch)
    loss.backward()

    for param in model.action_history_encoder.parameters():
        assert param.grad is not None
        assert torch.any(param.grad != 0)
    assert model.temporal_position_embedding.grad is not None


def test_frozen_encoders_receive_no_gradient():
    config = TemporalDenseVLAConfig(hidden_dim=32, num_layers=2, num_heads=4, ffn_dim=64, history_length=HISTORY_LENGTH)
    model = TemporalDenseVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)
    batch = make_batch(config, tokenizer, batch_size=2)

    prediction = model(
        batch["pixel_values"], batch["states"], batch["previous_actions"],
        batch["input_ids"], batch["attention_mask"],
    )
    loss, _ = compute_loss(prediction, batch)
    loss.backward()

    for param in model.vision_encoder.parameters():
        assert param.grad is None
    for param in model.language_encoder.parameters():
        assert param.grad is None
