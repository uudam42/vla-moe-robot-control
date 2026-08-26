"""Hybrid Dense/MoE Transformer: dense self-attention everywhere, FFN
sublayer swappable per layer between the standard dense ``FeedForward``
and sparse ``MoEFFN``.

Reimplements ``nn.TransformerEncoderLayer(norm_first=True)``'s pre-norm
math by hand (rather than subclassing it) specifically so the FFN
sublayer can be swapped -- the built-in layer bakes attention and FFN
together into one non-extensible ``forward()``. Every other piece
(``self_attn``, ``norm1``, ``norm2``, dropout placement) is built to be
parameter-compatible with ``nn.TransformerEncoderLayer``, so
``models/moe_vla.py``'s Dense->MoE conversion can copy weights directly.
"""

import torch
import torch.nn as nn

from models.moe import FeedForward, MoEFFN


class TransformerBlock(nn.Module):
    """One pre-norm block: ``x + Attn(norm1(x))``, then ``x + FFN(norm2(x))``.

    Args:
        ffn: Either a ``models.moe.FeedForward`` (dense) or
            ``models.moe.MoEFFN`` (sparse) instance.
        is_moe: Must match ``ffn``'s type -- determines whether ``forward``
            expects a ``(output, aux_loss, diagnostics)`` tuple back from
            ``ffn`` (MoE) or a plain tensor (dense).
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ffn: nn.Module, is_moe: bool) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = ffn
        self.is_moe = is_moe

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns ``(x, aux_loss_or_None, diagnostics_or_None)``."""
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        x = x + self.dropout1(attn_out)

        ffn_input = self.norm2(x)
        if self.is_moe:
            ffn_out, aux_loss, diagnostics = self.ffn(ffn_input)
            return x + ffn_out, aux_loss, diagnostics
        return x + self.ffn(ffn_input), None, None


class HybridTransformer(nn.Module):
    """Stack of ``TransformerBlock``s; layers in ``moe_layers`` are sparse MoE, rest dense."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        moe_layers: tuple,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        moe_layer_set = set(moe_layers)
        invalid = moe_layer_set - set(range(num_layers))
        if invalid:
            raise ValueError(f"moe_layers {sorted(invalid)} out of range for num_layers={num_layers}")

        blocks = []
        for layer_index in range(num_layers):
            if layer_index in moe_layer_set:
                ffn = MoEFFN(hidden_dim, ffn_dim, num_experts=num_experts, top_k=top_k, dropout=dropout)
                blocks.append(TransformerBlock(hidden_dim, num_heads, dropout, ffn, is_moe=True))
            else:
                ffn = FeedForward(hidden_dim, ffn_dim, dropout)
                blocks.append(TransformerBlock(hidden_dim, num_heads, dropout, ffn, is_moe=False))
        self.blocks = nn.ModuleList(blocks)
        self.moe_layers = tuple(sorted(moe_layer_set))

    def forward(self, x: torch.Tensor, collect_diagnostics: bool = False) -> tuple:
        """Returns ``(output, total_router_aux_loss, layer_diagnostics)``.

        ``total_router_aux_loss`` is a 0-valued tensor (not ``None``) when
        there are no MoE layers, so callers can always add it into the
        total loss unconditionally. ``layer_diagnostics`` is
        ``{layer_index: diagnostics_dict}`` for MoE layers only, populated
        only when ``collect_diagnostics=True`` (skipped by default to keep
        the normal training/inference path cheap).
        """
        total_aux_loss = x.new_zeros(())
        layer_diagnostics = {}
        for layer_index, block in enumerate(self.blocks):
            x, aux_loss, diagnostics = block(x)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
                if collect_diagnostics:
                    layer_diagnostics[layer_index] = diagnostics
        return x, total_aux_loss, layer_diagnostics
