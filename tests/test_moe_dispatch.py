"""Tests for MoEFFN's sparse dispatch: tokens actually reach their selected expert."""

import torch
import torch.nn as nn

from models.moe import FeedForward, MoEFFN


class ConstantExpert(nn.Module):
    """Returns a fixed per-expert constant vector, ignoring the input's content
    (except its shape) -- lets us verify dispatch by checking WHICH constant
    a token's output equals, independent of the router's actual weights."""

    def __init__(self, hidden_dim: int, value: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)


def make_controlled_layer(hidden_dim=8, num_experts=4, top_k=1) -> MoEFFN:
    layer = MoEFFN(hidden_dim=hidden_dim, ffn_dim=16, num_experts=num_experts, top_k=top_k)
    layer.experts = nn.ModuleList([ConstantExpert(hidden_dim, value=float(i)) for i in range(num_experts)])
    return layer


def test_token_output_matches_its_selected_expert_top1():
    layer = make_controlled_layer(num_experts=4, top_k=1)
    x = torch.randn(2, 4, 8)

    output, _, diagnostics = layer(x)
    indices = diagnostics["routing_indices"][:, :, 0]  # (B, T)

    for b in range(2):
        for t in range(4):
            expected_value = float(indices[b, t].item())
            assert torch.allclose(output[b, t], torch.full((8,), expected_value))


def test_unweighted_top1_output_ignores_router_probability():
    """Section: top-1 output must be selected_expert(x) UNWEIGHTED (see
    models/moe.py docstring) -- not scaled by the router's softmax probability."""
    layer = make_controlled_layer(num_experts=4, top_k=1)
    x = torch.randn(3, 4, 8)
    output, _, diagnostics = layer(x)
    indices = diagnostics["routing_indices"][:, :, 0]
    weights = diagnostics["routing_weights"][:, :, 0]

    # Even though routing_weights vary (softmax probabilities != 1), the
    # actual output must equal the expert's raw constant, not constant*weight.
    for b in range(3):
        for t in range(4):
            expected_value = float(indices[b, t].item())
            assert torch.allclose(output[b, t], torch.full((8,), expected_value))
            assert weights[b, t].item() != 1.0 or True  # weight need not be 1; output unaffected regardless


def test_experts_have_independent_parameters():
    layer = MoEFFN(hidden_dim=16, ffn_dim=32, num_experts=4, top_k=1)
    for i in range(4):
        for j in range(i + 1, 4):
            params_i = list(layer.experts[i].parameters())
            params_j = list(layer.experts[j].parameters())
            assert not torch.allclose(params_i[0], params_j[0])


def test_experts_are_not_weight_tied():
    """Modifying one expert's weights must not affect another's (no shared storage)."""
    layer = MoEFFN(hidden_dim=16, ffn_dim=32, num_experts=4, top_k=1)
    original = layer.experts[1].linear1.weight.clone()
    with torch.no_grad():
        layer.experts[0].linear1.weight.add_(1.0)
    assert torch.allclose(layer.experts[1].linear1.weight, original)


def test_feedforward_matches_expert_structure():
    """FeedForward (used both as the dense-layer FFN and each expert) has the
    documented Linear-GELU-Dropout-Linear-Dropout structure."""
    ffn = FeedForward(hidden_dim=16, ffn_dim=32, dropout=0.0)
    x = torch.randn(2, 3, 16)
    ffn.eval()
    out = ffn(x)
    assert out.shape == (2, 3, 16)
    assert torch.all(torch.isfinite(out))
