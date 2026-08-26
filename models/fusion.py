"""Projects each modality embedding to a common width and builds the
``[VISION, LANGUAGE, STATE, ACTION_QUERY]`` token sequence the dense
Transformer consumes.

Every FFN downstream of this (in ``dense_vla.py``'s Transformer) is a
standard dense FFN -- this token layout is deliberately explicit so a
future MoE milestone can replace those FFNs with expert layers without
touching fusion or the action head.
"""

import torch
import torch.nn as nn

NUM_TOKENS = 4  # VISION, LANGUAGE, STATE, ACTION_QUERY
ACTION_QUERY_INDEX = -1
TOKEN_NAMES = ("VISION", "LANGUAGE", "STATE", "ACTION_QUERY")  # fixed order below, position == index


class MultimodalFusion(nn.Module):
    def __init__(self, vision_dim: int, language_dim: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        # Learnable per-slot embedding so the Transformer can tell the
        # (otherwise position-less) modality tokens apart.
        self.token_type_embedding = nn.Parameter(torch.randn(1, NUM_TOKENS, hidden_dim) * 0.02)

    def forward(
        self, vision_embedding: torch.Tensor, language_embedding: torch.Tensor, state_embedding: torch.Tensor
    ) -> torch.Tensor:
        batch_size = vision_embedding.shape[0]
        vision_token = self.vision_proj(vision_embedding).unsqueeze(1)
        language_token = self.language_proj(language_embedding).unsqueeze(1)
        state_token = self.state_proj(state_embedding).unsqueeze(1)
        action_query = self.action_query.expand(batch_size, -1, -1)

        tokens = torch.cat([vision_token, language_token, state_token, action_query], dim=1)
        return tokens + self.token_type_embedding
