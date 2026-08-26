"""Dense Vision-Language-Action policy: the Step 4 behavior-cloning model.

```
RGB -> VisionEncoder ----------\
Language -> LanguageEncoder -----> MultimodalFusion -> [4 tokens] -> Transformer -> action_query token -> ActionHead -> 8D action
State -> StateEncoder ----------/
```

Every FFN inside the Transformer is a standard dense ``nn.TransformerEncoderLayer``
FFN -- no expert routing. This is intentional: Step 4 is the dense baseline
a future Mixture-of-Experts milestone will be compared against.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.fusion import MultimodalFusion
from models.language_encoder import DEFAULT_LANGUAGE_BACKBONE, LanguageEncoder
from models.state_encoder import StateEncoder
from models.vision_encoder import VisionEncoder

NUM_JOINTS = 7


@dataclass
class DenseVLAConfig:
    """Architecture + freezing configuration. Persisted verbatim into every checkpoint."""

    state_dim: int = 23
    action_dim: int = NUM_JOINTS + 1  # 7 joint targets + 1 gripper logit
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

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "DenseVLAConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DenseVLA(nn.Module):
    def __init__(self, config: DenseVLAConfig) -> None:
        super().__init__()
        self.config = config

        self.vision_encoder = VisionEncoder(trainable=config.train_vision_encoder)
        self.language_encoder = LanguageEncoder(
            model_name=config.language_backbone, trainable=config.train_language_encoder
        )
        self.state_encoder = StateEncoder(state_dim=config.state_dim, hidden_dim=config.hidden_dim)
        self.fusion = MultimodalFusion(
            vision_dim=self.vision_encoder.output_dim,
            language_dim=self.language_encoder.output_dim,
            state_dim=self.state_encoder.output_dim,
            hidden_dim=config.hidden_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers, enable_nested_tensor=False
        )

        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> dict:
        """Returns a dict with ``joint_targets_normalized`` (B,7) and ``gripper_logit`` (B,1).

        The gripper head outputs a raw logit (not a probability) -- see
        ``training/losses.py`` for why (BCEWithLogitsLoss expects logits);
        callers apply ``sigmoid`` themselves (``models/policy.py`` does this
        at inference).
        """
        vision_embedding = self.vision_encoder(pixel_values)
        language_embedding = self.language_encoder(input_ids, attention_mask)
        state_embedding = self.state_encoder(state)

        tokens = self.fusion(vision_embedding, language_embedding, state_embedding)
        transformed = self.transformer(tokens)
        action_query_output = transformed[:, -1, :]  # ACTION_QUERY is always the last token

        raw_action = self.action_head(action_query_output)
        return {
            "joint_targets_normalized": raw_action[:, :NUM_JOINTS],
            "gripper_logit": raw_action[:, NUM_JOINTS:],
        }


def count_parameters(model: nn.Module) -> tuple:
    """Returns ``(total_parameters, trainable_parameters)``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
