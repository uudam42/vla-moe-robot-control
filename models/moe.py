"""Core sparse Mixture-of-Experts building blocks: shared FFN shape,
top-k router with real conditional dispatch, and the Switch-Transformer-style
load-balancing auxiliary loss.

Nothing here knows about VLA/robotics -- see ``models/moe_transformer.py``
for where this gets wired into a Transformer block, and
``models/moe_vla.py`` for the full policy and the Dense->MoE conversion
that initializes every expert as an exact copy of the trained Dense FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """``Linear -> GELU -> Dropout -> Linear -> Dropout``.

    Deliberately matches ``nn.TransformerEncoderLayer``'s internal FFN
    sublayer exactly (same submodule names ``linear1``/``linear2``, same
    dropout placement) so that Dense FFN weights can be copied into this
    class parameter-for-parameter with a plain ``load_state_dict`` -- used
    both as the non-MoE layers' FFN and as each individual expert's FFN.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))


def load_balance_loss(router_probs: torch.Tensor, top1_index: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch-Transformer-style load-balancing auxiliary loss.

    ``L = num_experts * sum_e( f_e * P_e )`` where ``f_e`` is the fraction
    of tokens whose top-1 choice is expert ``e`` (a constant w.r.t.
    gradients -- computed from a hard index) and ``P_e`` is the mean
    router softmax probability assigned to expert ``e`` (differentiable).
    This is the only path gradient reaches the router in this
    implementation: top-1 hard selection itself is non-differentiable, and
    (per README "Router weighting detail") the selected expert's output is
    NOT scaled by its router probability, so no gradient reaches the
    router through the task loss either -- only through this auxiliary
    term. Minimized when tokens are spread uniformly across experts.

    Args:
        router_probs: Softmax router probabilities, shape ``(N, num_experts)``.
        top1_index: Hard top-1 expert assignment per token, shape ``(N,)``.
        num_experts: Number of experts.
    """
    one_hot = F.one_hot(top1_index, num_classes=num_experts).to(router_probs.dtype)
    fraction_per_expert = one_hot.mean(dim=0)
    mean_prob_per_expert = router_probs.mean(dim=0)
    return num_experts * torch.sum(fraction_per_expert * mean_prob_per_expert)


def router_entropy(router_probs: torch.Tensor) -> torch.Tensor:
    """Mean per-token entropy of the router's softmax distribution, in nats.

    Diagnostic only (README "Router entropy") -- not part of any loss.
    Near ``log(num_experts)`` means near-uniform/undecided routing; near 0
    means confident (possibly collapsed) routing.
    """
    eps = 1e-9
    return -(router_probs * torch.log(router_probs + eps)).sum(dim=-1).mean()


class MoEFFN(nn.Module):
    """Top-k sparse MoE feed-forward layer with genuine conditional dispatch.

    Each expert only runs on the tokens actually routed to it (grouped by
    boolean mask), not "run every expert on every token then mask" -- see
    README "Sparse dispatch" for why that distinction matters and what it
    does/doesn't buy on MPS.

    For ``top_k == 1``, the selected expert's output is used UNWEIGHTED
    (not scaled by its router probability) -- see README "Router weighting
    detail": this is what makes a freshly Dense->MoE-converted model
    (every expert initialized as an exact copy of the Dense FFN)
    numerically reproduce the Dense model's output. For ``top_k > 1``,
    outputs are combined with router weights renormalized to sum to 1
    across the selected experts (standard soft top-k blending).
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int = 4, top_k: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        if not (1 <= top_k <= num_experts):
            raise ValueError(f"top_k must be in [1, num_experts={num_experts}], got {top_k}")
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([FeedForward(hidden_dim, ffn_dim, dropout) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> tuple:
        """``x``: ``(B, T, D)``. Returns ``(output, aux_loss, diagnostics)``.

        ``diagnostics`` (all detached, evaluation/logging use only):
        ``probs (B,T,E)``, ``routing_indices (B,T,top_k)``, ``routing_weights (B,T,top_k)``.
        """
        batch_size, num_tokens, hidden_dim = x.shape
        flat = x.reshape(batch_size * num_tokens, hidden_dim)

        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        top1_index = probs.argmax(dim=-1)

        output = torch.zeros_like(flat)

        if self.top_k == 1:
            for expert_id, expert in enumerate(self.experts):
                mask = top1_index == expert_id
                if mask.any():
                    output[mask] = expert(flat[mask])
            routing_indices = top1_index.reshape(batch_size, num_tokens, 1)
            routing_weights = probs.gather(-1, top1_index.unsqueeze(-1)).reshape(batch_size, num_tokens, 1)
        else:
            topk_probs, topk_index = probs.topk(self.top_k, dim=-1)
            topk_weight = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            for slot in range(self.top_k):
                slot_index = topk_index[:, slot]
                slot_weight = topk_weight[:, slot]
                for expert_id, expert in enumerate(self.experts):
                    mask = slot_index == expert_id
                    if mask.any():
                        output[mask] = output[mask] + expert(flat[mask]) * slot_weight[mask].unsqueeze(-1)
            routing_indices = topk_index.reshape(batch_size, num_tokens, self.top_k)
            routing_weights = topk_weight.reshape(batch_size, num_tokens, self.top_k)

        aux_loss = load_balance_loss(probs, top1_index, self.num_experts)

        diagnostics = {
            "probs": probs.reshape(batch_size, num_tokens, self.num_experts).detach(),
            "routing_indices": routing_indices.detach(),
            "routing_weights": routing_weights.detach(),
            "entropy": router_entropy(probs).detach(),
        }
        return output.reshape(batch_size, num_tokens, hidden_dim), aux_loss, diagnostics
