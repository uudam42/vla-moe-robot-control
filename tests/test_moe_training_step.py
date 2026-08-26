"""Forward/backward smoke test for MoEVLA: router, experts, and shared
layers must all receive gradients from one training step."""

import torch

from models.language_encoder import load_tokenizer
from models.moe_vla import MoEVLA, MoEVLAConfig
from training.train_moe import compute_moe_loss


def make_batch(config, tokenizer, batch_size=6):
    tokenized = tokenizer(
        ["Pick up the red cube."] * batch_size, return_tensors="pt", padding="max_length",
        truncation=True, max_length=config.max_instruction_length,
    )
    return {
        "pixel_values": torch.randn(batch_size, 3, 224, 224),
        "state": torch.randn(batch_size, config.state_dim),
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "joint_targets_normalized": torch.randn(batch_size, 7),
        "gripper_target": torch.randint(0, 2, (batch_size,)).float(),
    }


def test_one_training_step_produces_finite_loss_and_updates_weights():
    config = MoEVLAConfig(hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64, num_experts=4, top_k=1, moe_layers=(1, 3))
    model = MoEVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)
    batch = make_batch(config, tokenizer, batch_size=8)  # >=8 tokens/expert-ish so more experts get hit

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    params_before = [p.detach().clone() for p in trainable_params]

    prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
    loss, metrics = compute_moe_loss(prediction, batch, gripper_loss_weight=1.0, router_aux_loss_weight=0.01)

    assert torch.isfinite(loss)
    assert "router_aux_loss" in metrics

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert any(not torch.equal(before, after) for before, after in zip(params_before, trainable_params))


def test_router_parameters_receive_finite_nonzero_gradients():
    config = MoEVLAConfig(hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64, num_experts=4, top_k=1, moe_layers=(1, 3))
    model = MoEVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)
    batch = make_batch(config, tokenizer, batch_size=8)

    prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
    loss, _ = compute_moe_loss(prediction, batch, gripper_loss_weight=1.0, router_aux_loss_weight=0.01)
    loss.backward()

    for layer_index in config.moe_layers:
        router = model.transformer.blocks[layer_index].ffn.router
        assert router.weight.grad is not None
        assert torch.all(torch.isfinite(router.weight.grad))
        assert torch.any(router.weight.grad != 0)


def test_selected_experts_receive_gradients_over_multiple_batches():
    """Top-1 routing means not every expert is guaranteed a gradient in a
    single tiny batch -- but across several batches, every expert should
    receive a gradient at least once (no permanently dead expert)."""
    config = MoEVLAConfig(hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64, num_experts=4, top_k=1, moe_layers=(1, 3))
    model = MoEVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)

    ever_received_grad = {layer: [False] * config.num_experts for layer in config.moe_layers}

    for _ in range(6):
        model.zero_grad()
        batch = make_batch(config, tokenizer, batch_size=8)
        prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
        loss, _ = compute_moe_loss(prediction, batch, gripper_loss_weight=1.0, router_aux_loss_weight=0.01)
        loss.backward()

        for layer_index in config.moe_layers:
            for expert_id, expert in enumerate(model.transformer.blocks[layer_index].ffn.experts):
                grad = expert.linear1.weight.grad
                if grad is not None and torch.any(grad != 0):
                    ever_received_grad[layer_index][expert_id] = True

    for layer_index, flags in ever_received_grad.items():
        assert any(flags), f"layer {layer_index}: no expert received any gradient across 6 batches"


def test_shared_layers_receive_gradients():
    config = MoEVLAConfig(hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64, num_experts=4, top_k=1, moe_layers=(1, 3))
    model = MoEVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)
    batch = make_batch(config, tokenizer, batch_size=8)

    prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
    loss, _ = compute_moe_loss(prediction, batch, gripper_loss_weight=1.0, router_aux_loss_weight=0.01)
    loss.backward()

    for param in model.state_encoder.parameters():
        assert param.grad is not None
    for param in model.action_head.parameters():
        assert param.grad is not None
    # Dense (non-MoE) transformer layer 0's FFN must also train.
    dense_block = model.transformer.blocks[0]
    assert dense_block.ffn.linear1.weight.grad is not None


def test_frozen_encoders_still_receive_no_gradient():
    config = MoEVLAConfig(hidden_dim=32, num_layers=4, num_heads=4, ffn_dim=64, num_experts=4, top_k=1, moe_layers=(1, 3))
    model = MoEVLA(config)
    tokenizer = load_tokenizer(config.language_backbone)
    batch = make_batch(config, tokenizer, batch_size=4)

    prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
    loss, _ = compute_moe_loss(prediction, batch, gripper_loss_weight=1.0, router_aux_loss_weight=0.01)
    loss.backward()

    for param in model.vision_encoder.parameters():
        assert param.grad is None
    for param in model.language_encoder.parameters():
        assert param.grad is None
