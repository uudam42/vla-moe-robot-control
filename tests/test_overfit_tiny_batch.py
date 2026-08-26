"""The Step 4 hard gate: can the architecture + loss actually learn?

Trains on one small, fixed, synthetic batch repeatedly and requires a large
loss reduction -- if this fails, the bug is in normalization/alignment/
freezing/loss weighting, not model capacity (see README "Tiny overfit is a
hard gate"). Uses a small Transformer config (not the full-size default)
purely to keep this fast in the regular test suite; the real overfit check
against actual demonstration data is run manually via
``python -m training.train --overfit-samples 64``.
"""

import torch

from models.dense_vla import DenseVLA, DenseVLAConfig
from models.language_encoder import load_tokenizer
from training.losses import compute_loss

NUM_STEPS = 150
BATCH_SIZE = 6


def _make_fixed_batch(config, tokenizer, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokenized = tokenizer(
        ["Pick up the red cube."] * BATCH_SIZE,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.max_instruction_length,
    )
    return {
        "pixel_values": torch.randn(BATCH_SIZE, 3, 224, 224, generator=generator),
        "state": torch.randn(BATCH_SIZE, config.state_dim, generator=generator),
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "joint_targets_normalized": torch.randn(BATCH_SIZE, 7, generator=generator),
        "gripper_target": (torch.rand(BATCH_SIZE, generator=generator) > 0.5).float(),
    }


def test_model_overfits_a_tiny_fixed_batch():
    torch.manual_seed(0)
    config = DenseVLAConfig(hidden_dim=64, num_layers=2, num_heads=4, ffn_dim=128, dropout=0.0)
    model = DenseVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)
    batch = _make_fixed_batch(config, tokenizer)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-3)

    losses = []
    model.train()
    for _ in range(NUM_STEPS):
        prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
        loss, metrics = compute_loss(prediction, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(metrics["joint_loss"])

    initial_joint_loss = losses[0]
    final_joint_loss = min(losses[-5:])  # last few steps, robust to step-to-step noise
    reduction = 1.0 - (final_joint_loss / initial_joint_loss)

    assert reduction > 0.90, (
        f"joint loss only reduced by {reduction:.1%} (initial={initial_joint_loss:.4f}, "
        f"final={final_joint_loss:.4f}) -- expected >90%"
    )

    model.eval()
    with torch.no_grad():
        prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
        gripper_pred = (torch.sigmoid(prediction["gripper_logit"].squeeze(-1)) >= 0.5).float()
    gripper_accuracy = (gripper_pred == batch["gripper_target"]).float().mean().item()
    assert gripper_accuracy >= 0.9, f"gripper accuracy only {gripper_accuracy:.2f} after overfitting"
