"""Tests for the router inside models/moe.py::MoEFFN -- shapes and top-k selection."""

import torch

from models.moe import MoEFFN, load_balance_loss, router_entropy


def test_router_output_shape():
    layer = MoEFFN(hidden_dim=16, ffn_dim=32, num_experts=4, top_k=1)
    x = torch.randn(2, 4, 16)
    output, aux_loss, diagnostics = layer(x)

    assert output.shape == (2, 4, 16)
    assert diagnostics["probs"].shape == (2, 4, 4)  # (B, T, num_experts)
    assert diagnostics["routing_indices"].shape == (2, 4, 1)  # top_k=1


def test_exactly_top_k_experts_selected_per_token_top1():
    layer = MoEFFN(hidden_dim=16, ffn_dim=32, num_experts=4, top_k=1)
    x = torch.randn(3, 4, 16)
    _, _, diagnostics = layer(x)

    indices = diagnostics["routing_indices"]
    assert indices.shape[-1] == 1
    assert torch.all(indices >= 0) and torch.all(indices < 4)


def test_exactly_top_k_experts_selected_per_token_topk2():
    layer = MoEFFN(hidden_dim=16, ffn_dim=32, num_experts=4, top_k=2)
    x = torch.randn(3, 4, 16)
    output, _, diagnostics = layer(x)

    indices = diagnostics["routing_indices"]
    assert indices.shape[-1] == 2
    # The two selected experts per token must be distinct (softmax topk never repeats).
    assert torch.all(indices[..., 0] != indices[..., 1])
    assert output.shape == (3, 4, 16)


def test_routing_weights_sum_to_one_for_topk():
    layer = MoEFFN(hidden_dim=16, ffn_dim=32, num_experts=4, top_k=2)
    x = torch.randn(2, 4, 16)
    _, _, diagnostics = layer(x)
    weight_sums = diagnostics["routing_weights"].sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


def test_router_rejects_invalid_top_k():
    import pytest

    with pytest.raises(ValueError):
        MoEFFN(hidden_dim=16, ffn_dim=32, num_experts=4, top_k=5)
    with pytest.raises(ValueError):
        MoEFFN(hidden_dim=16, ffn_dim=32, num_experts=4, top_k=0)


def test_router_entropy_uniform_vs_confident():
    uniform_probs = torch.full((10, 4), 0.25)
    confident_probs = torch.zeros(10, 4)
    confident_probs[:, 0] = 1.0

    uniform_entropy = router_entropy(uniform_probs)
    confident_entropy = router_entropy(confident_probs)

    assert uniform_entropy > confident_entropy
    assert confident_entropy < 1e-3


def test_load_balance_loss_finite_and_differentiable():
    logits = torch.randn(20, 4, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    top1_index = probs.argmax(dim=-1)

    loss = load_balance_loss(probs, top1_index, num_experts=4)
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_load_balance_loss_minimized_when_uniform():
    """Perfectly uniform routing (every expert gets exactly 1/E of tokens
    and 1/E mean probability) should give the theoretical minimum: 1.0."""
    num_experts = 4
    probs = torch.full((8, num_experts), 1.0 / num_experts)
    top1_index = torch.arange(8) % num_experts

    loss = load_balance_loss(probs, top1_index, num_experts)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)
