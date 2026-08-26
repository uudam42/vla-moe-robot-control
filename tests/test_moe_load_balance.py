"""Tests for load-balancing behavior at the MoEFFN/HybridTransformer level
(complementing the pure-function tests in tests/test_moe_router.py)."""

import torch
import torch.nn as nn

from models.moe import MoEFFN
from models.moe_transformer import HybridTransformer


def test_moe_ffn_forward_returns_finite_aux_loss():
    layer = MoEFFN(hidden_dim=16, ffn_dim=32, num_experts=4, top_k=1)
    x = torch.randn(4, 4, 16)
    _, aux_loss, _ = layer(x)
    assert torch.isfinite(aux_loss)
    assert aux_loss.item() > 0  # random init is never perfectly uniform


def test_hybrid_transformer_aggregates_aux_loss_across_moe_layers():
    transformer = HybridTransformer(
        hidden_dim=16, num_layers=4, num_heads=2, ffn_dim=32, dropout=0.0,
        moe_layers=(1, 3), num_experts=4, top_k=1,
    )
    x = torch.randn(2, 4, 16)
    _, total_aux_loss, diagnostics = transformer(x, collect_diagnostics=True)

    assert torch.isfinite(total_aux_loss)
    assert set(diagnostics.keys()) == {1, 3}


def test_hybrid_transformer_zero_aux_loss_with_no_moe_layers():
    transformer = HybridTransformer(
        hidden_dim=16, num_layers=2, num_heads=2, ffn_dim=32, dropout=0.0,
        moe_layers=(), num_experts=4, top_k=1,
    )
    x = torch.randn(2, 4, 16)
    _, total_aux_loss, diagnostics = transformer(x, collect_diagnostics=True)

    assert total_aux_loss.item() == 0.0
    assert diagnostics == {}


def test_detects_collapsed_routing_from_utilization_counts():
    """If a router collapses onto one expert, utilization counts must clearly show it
    (this is the diagnostic the training loop's expert_utilization.json relies on)."""
    layer = MoEFFN(hidden_dim=8, ffn_dim=16, num_experts=4, top_k=1)
    with torch.no_grad():
        # Force collapse: router always strongly prefers expert 0.
        layer.router.weight.zero_()
        layer.router.bias.copy_(torch.tensor([10.0, -10.0, -10.0, -10.0]))

    x = torch.randn(5, 4, 8)
    _, _, diagnostics = layer(x)
    indices = diagnostics["routing_indices"][:, :, 0]
    fraction_expert_0 = (indices == 0).float().mean().item()

    assert fraction_expert_0 > 0.95  # collapse is measurable, not hidden
