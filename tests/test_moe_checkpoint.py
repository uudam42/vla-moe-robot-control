"""MoE checkpoint save/reload must reproduce predictions and preserve MoE config fields."""

import numpy as np
import torch

from models.language_encoder import load_tokenizer
from models.moe_vla import MoEVLA, MoEVLAConfig
from training.checkpoint import load_checkpoint, save_checkpoint
from training.normalization import ActionNormalizer, StateNormalizer


def make_batch(config, tokenizer, batch_size=2):
    tokenized = tokenizer(
        ["Pick up the red cube."] * batch_size, return_tensors="pt", padding="max_length",
        truncation=True, max_length=config.max_instruction_length,
    )
    return {
        "pixel_values": torch.randn(batch_size, 3, 224, 224),
        "state": torch.randn(batch_size, config.state_dim),
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def test_save_and_reload_reproduces_predictions(tmp_path):
    config = MoEVLAConfig(hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64, num_experts=4, top_k=1, moe_layers=(1, 3))
    model = MoEVLA(config)
    model.eval()
    tokenizer = load_tokenizer(config.language_backbone)
    batch = make_batch(config, tokenizer)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    with torch.no_grad():
        original_output = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])

    checkpoint_path = tmp_path / "moe_test.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, epoch=5, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer, val_metrics={"joint_mae": 0.02},
    )

    checkpoint = load_checkpoint(checkpoint_path)
    restored_config = MoEVLAConfig.from_dict(checkpoint["config"])
    restored_model = MoEVLA(restored_config)
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


def test_checkpoint_preserves_moe_specific_config_fields(tmp_path):
    config = MoEVLAConfig(
        hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64,
        num_experts=6, top_k=2, moe_layers=(0, 2, 3), router_aux_loss_weight=0.05,
    )
    model = MoEVLA(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    checkpoint_path = tmp_path / "moe_config_test.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, epoch=0, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer, val_metrics={},
    )

    checkpoint = load_checkpoint(checkpoint_path)
    restored_config = MoEVLAConfig.from_dict(checkpoint["config"])

    assert restored_config.num_experts == 6
    assert restored_config.top_k == 2
    assert restored_config.moe_layers == (0, 2, 3)
    assert restored_config.router_aux_loss_weight == 0.05


def test_reloaded_model_reproduces_diagnostics(tmp_path):
    config = MoEVLAConfig(hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64, num_experts=4, top_k=1, moe_layers=(1, 3))
    model = MoEVLA(config)
    model.eval()
    tokenizer = load_tokenizer(config.language_backbone)
    batch = make_batch(config, tokenizer)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23))
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7))

    checkpoint_path = tmp_path / "moe_diag_test.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, epoch=0, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer, val_metrics={},
    )

    with torch.no_grad():
        original = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"], collect_diagnostics=True)

    checkpoint = load_checkpoint(checkpoint_path)
    restored_model = MoEVLA(MoEVLAConfig.from_dict(checkpoint["config"]))
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_model.eval()
    with torch.no_grad():
        restored = restored_model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"], collect_diagnostics=True)

    for layer_index in original["layer_diagnostics"]:
        assert torch.equal(
            original["layer_diagnostics"][layer_index]["routing_indices"],
            restored["layer_diagnostics"][layer_index]["routing_indices"],
        )
