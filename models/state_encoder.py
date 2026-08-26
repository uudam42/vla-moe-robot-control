"""Small MLP encoding the 23D robot proprioceptive state."""

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """``state_dim -> 128 -> hidden_dim``, GELU + LayerNorm, no pretraining."""

    def __init__(self, state_dim: int = 23, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """``state``: ``(B, state_dim)`` -> ``(B, hidden_dim)``."""
        return self.net(state)
