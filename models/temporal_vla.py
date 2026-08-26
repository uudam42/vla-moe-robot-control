"""Temporal Dense VLA: the Step 7 model. Same modality encoders and dense
Transformer backbone as Step 4's ``DenseVLA``, but conditioned on a short
window of past observations and actions instead of only the current one.

```
For each of H=4 window positions (t-3, t-2, t-1, t):
  RGB   -> VisionEncoder (shared, frozen)   --\
  State -> StateEncoder (shared)              +--> sum --> + temporal position embedding --> token_h
  PrevAction -> ActionHistoryEncoder (shared)--/
  (position t's PrevAction slot is always the NO_ACTION sentinel -- masked, never the target)

[token_{t-3}, token_{t-2}, token_{t-1}, token_t, LANGUAGE, ACTION_QUERY]
              -> Dense Transformer (HybridTransformer, moe_layers=()) -> ACTION_QUERY output -> ActionHead -> 8D action
```

Uses the "fused temporal token per timestep" layout (README "Alternative
simpler token layout") rather than one token per (modality, timestep)
pair -- simpler to reason about and test, and keeps the sequence short
(6 tokens for H=4) rather than growing with H*3+2.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.dense_vla import DenseVLA
from models.language_encoder import DEFAULT_LANGUAGE_BACKBONE, LanguageEncoder
from models.moe_transformer import HybridTransformer
from models.state_encoder import StateEncoder
from models.temporal_history import ACTION_DIM as PREV_ACTION_DIM
from models.vision_encoder import VisionEncoder

NUM_JOINTS = 7
ACTION_DIM = NUM_JOINTS + 1  # model output: 7 joint targets + 1 gripper logit


@dataclass
class TemporalDenseVLAConfig:
    """Same fields as ``DenseVLAConfig`` plus ``history_length``. Persisted
    verbatim into every checkpoint."""

    state_dim: int = 23
    action_dim: int = ACTION_DIM
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_instruction_length: int = 32
    vision_backbone: str = "resnet18"
    language_backbone: str = DEFAULT_LANGUAGE_BACKBONE
    train_vision_encoder: bool = False
    train_language_encoder: bool = False
    history_length: int = 4

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "TemporalDenseVLAConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ActionHistoryEncoder(nn.Module):
    """``8 -> 128 -> hidden_dim``, GELU + LayerNorm -- same shape/style as
    ``models.state_encoder.StateEncoder``, encoding the previous-action
    representation (README "Previous action representation")."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PREV_ACTION_DIM, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalDenseVLA(nn.Module):
    def __init__(self, config: TemporalDenseVLAConfig) -> None:
        super().__init__()
        self.config = config
        H = config.history_length

        self.vision_encoder = VisionEncoder(trainable=config.train_vision_encoder)
        self.language_encoder = LanguageEncoder(
            model_name=config.language_backbone, trainable=config.train_language_encoder
        )
        self.state_encoder = StateEncoder(state_dim=config.state_dim, hidden_dim=config.hidden_dim)
        self.action_history_encoder = ActionHistoryEncoder(hidden_dim=config.hidden_dim)

        self.vision_proj = nn.Linear(self.vision_encoder.output_dim, config.hidden_dim)
        self.state_proj = nn.Linear(self.state_encoder.output_dim, config.hidden_dim)
        self.action_proj = nn.Linear(self.action_history_encoder.output_dim, config.hidden_dim)
        self.language_proj = nn.Linear(self.language_encoder.output_dim, config.hidden_dim)

        # Learned per-slot embeddings: one per temporal position (README
        # "Temporal position embeddings"), plus one each for the LANGUAGE
        # and ACTION_QUERY slots that follow the temporal window.
        self.temporal_position_embedding = nn.Parameter(torch.randn(1, H, config.hidden_dim) * 0.02)
        self.extra_token_embedding = nn.Parameter(torch.randn(1, 2, config.hidden_dim) * 0.02)  # [LANGUAGE, ACTION_QUERY]
        self.action_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)

        # Dense Transformer -- HybridTransformer with no MoE layers is a
        # plain dense pre-norm Transformer (see models/moe_transformer.py);
        # reused here rather than reimplemented (README "No MoE this milestone").
        self.transformer = HybridTransformer(
            hidden_dim=config.hidden_dim, num_layers=config.num_layers, num_heads=config.num_heads,
            ffn_dim=config.ffn_dim, dropout=config.dropout, moe_layers=(), num_experts=1, top_k=1,
        )

        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        states: torch.Tensor,
        previous_actions: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Args:
            pixel_values: ``(B, H, 3, 224, 224)``.
            states: ``(B, H, state_dim)``, normalized.
            previous_actions: ``(B, H, 8)`` -- see ``models/temporal_history.py``
                for the padding/masking contract; slot ``H-1`` (current
                timestep) is always the NO_ACTION sentinel.
            input_ids, attention_mask: ``(B, L)``.

        Returns dict with ``joint_targets_normalized (B,7)`` and ``gripper_logit (B,1)``
        -- identical output contract to ``DenseVLA.forward()``.
        """
        batch_size, history_length = pixel_values.shape[:2]

        flat_pixels = pixel_values.reshape(batch_size * history_length, *pixel_values.shape[2:])
        vision_embedding = self.vision_encoder(flat_pixels).reshape(batch_size, history_length, -1)

        flat_states = states.reshape(batch_size * history_length, -1)
        state_embedding = self.state_encoder(flat_states).reshape(batch_size, history_length, -1)

        flat_actions = previous_actions.reshape(batch_size * history_length, -1)
        action_embedding = self.action_history_encoder(flat_actions).reshape(batch_size, history_length, -1)

        temporal_tokens = (
            self.vision_proj(vision_embedding) + self.state_proj(state_embedding) + self.action_proj(action_embedding)
        )
        temporal_tokens = temporal_tokens + self.temporal_position_embedding

        language_embedding = self.language_encoder(input_ids, attention_mask)
        language_token = self.language_proj(language_embedding).unsqueeze(1) + self.extra_token_embedding[:, 0:1, :]
        action_query = self.action_query.expand(batch_size, -1, -1) + self.extra_token_embedding[:, 1:2, :]

        tokens = torch.cat([temporal_tokens, language_token, action_query], dim=1)  # (B, H+2, D)
        transformed, _, _ = self.transformer(tokens)
        action_query_output = transformed[:, -1, :]

        raw_action = self.action_head(action_query_output)
        return {
            "joint_targets_normalized": raw_action[:, :NUM_JOINTS],
            "gripper_logit": raw_action[:, NUM_JOINTS:],
        }


def convert_dense_to_temporal(dense_model: DenseVLA, config: TemporalDenseVLAConfig) -> TemporalDenseVLA:
    """Dense->Temporal initialization (README "Dense initialization").

    Copied verbatim from the trained Dense checkpoint (identical shapes,
    same role): ``vision_encoder``, ``language_encoder``, ``state_encoder``,
    ``action_head``, ``language_proj``, ``vision_proj``, ``state_proj``
    (Dense's ``fusion.*_proj`` layers -- Temporal applies the same
    per-modality projections, just per-timestep with shared weights),
    ``action_query``, and every Transformer layer's self-attention +
    LayerNorms + dense FFN (Temporal's ``HybridTransformer`` with no MoE
    layers has the same per-layer structure as Dense's
    ``nn.TransformerEncoder``).

    Randomly initialized (no Dense equivalent): ``action_history_encoder``,
    ``action_proj``, ``temporal_position_embedding``, ``extra_token_embedding``.

    Unlike Step 6's Dense->MoE conversion, this does NOT reproduce Dense's
    output numerically -- the token sequence itself is structurally
    different (temporal window + language + action-query vs. Dense's
    single-timestep 4 tokens), so exact functional equivalence isn't
    expected or tested here.
    """
    if dense_model.config.hidden_dim != config.hidden_dim or dense_model.config.num_layers != config.num_layers:
        raise ValueError("Dense and Temporal configs must share hidden_dim/num_layers for conversion")

    temporal_model = TemporalDenseVLA(config)

    temporal_model.vision_encoder.load_state_dict(dense_model.vision_encoder.state_dict())
    temporal_model.language_encoder.load_state_dict(dense_model.language_encoder.state_dict())
    temporal_model.state_encoder.load_state_dict(dense_model.state_encoder.state_dict())
    temporal_model.action_head.load_state_dict(dense_model.action_head.state_dict())

    temporal_model.vision_proj.load_state_dict(dense_model.fusion.vision_proj.state_dict())
    temporal_model.state_proj.load_state_dict(dense_model.fusion.state_proj.state_dict())
    temporal_model.language_proj.load_state_dict(dense_model.fusion.language_proj.state_dict())
    with torch.no_grad():
        temporal_model.action_query.copy_(dense_model.fusion.action_query)

    for layer_index, dense_layer in enumerate(dense_model.transformer.layers):
        temporal_block = temporal_model.transformer.blocks[layer_index]
        temporal_block.self_attn.load_state_dict(dense_layer.self_attn.state_dict())
        temporal_block.norm1.load_state_dict(dense_layer.norm1.state_dict())
        temporal_block.norm2.load_state_dict(dense_layer.norm2.state_dict())
        temporal_block.ffn.linear1.load_state_dict(dense_layer.linear1.state_dict())
        temporal_block.ffn.linear2.load_state_dict(dense_layer.linear2.state_dict())

    return temporal_model
