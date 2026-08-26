"""Sparse Mixture-of-Experts Vision-Language-Action policy: the Step 6 model.

Identical input/output contract and identical shared components (vision/
language/state encoders, fusion, action head) as ``models.dense_vla.DenseVLA``
-- the only architectural difference is that selected Transformer layers'
FFN sublayers are sparse ``MoEFFN`` instead of dense ``FeedForward`` (see
``models/moe_transformer.py``). This isolates the Dense-vs-MoE comparison
to exactly that one change, per README "Experimental fairness".
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from models.dense_vla import NUM_JOINTS, DenseVLA
from models.fusion import MultimodalFusion
from models.language_encoder import DEFAULT_LANGUAGE_BACKBONE, LanguageEncoder
from models.moe_transformer import HybridTransformer
from models.state_encoder import StateEncoder
from models.vision_encoder import VisionEncoder

DEFAULT_MOE_LAYERS = (1, 3)


@dataclass
class MoEVLAConfig:
    """Same fields as ``DenseVLAConfig`` plus MoE-specific ones. Persisted
    verbatim into every checkpoint."""

    state_dim: int = 23
    action_dim: int = NUM_JOINTS + 1
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
    num_experts: int = 4
    top_k: int = 1
    moe_layers: tuple = DEFAULT_MOE_LAYERS
    router_aux_loss_weight: float = 0.01

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        data["moe_layers"] = list(data["moe_layers"])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MoEVLAConfig":
        data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "moe_layers" in data:
            data["moe_layers"] = tuple(data["moe_layers"])
        return cls(**data)


class MoEVLA(nn.Module):
    def __init__(self, config: MoEVLAConfig) -> None:
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
        self.transformer = HybridTransformer(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            moe_layers=config.moe_layers,
            num_experts=config.num_experts,
            top_k=config.top_k,
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
        collect_diagnostics: bool = False,
    ) -> dict:
        """Same output contract as ``DenseVLA.forward()`` plus ``router_aux_loss``
        (a 0-valued tensor if there are no MoE layers) and, only when
        ``collect_diagnostics=True``, ``layer_diagnostics``.
        """
        vision_embedding = self.vision_encoder(pixel_values)
        language_embedding = self.language_encoder(input_ids, attention_mask)
        state_embedding = self.state_encoder(state)

        tokens = self.fusion(vision_embedding, language_embedding, state_embedding)
        transformed, router_aux_loss, layer_diagnostics = self.transformer(
            tokens, collect_diagnostics=collect_diagnostics
        )
        action_query_output = transformed[:, -1, :]

        raw_action = self.action_head(action_query_output)
        result = {
            "joint_targets_normalized": raw_action[:, :NUM_JOINTS],
            "gripper_logit": raw_action[:, NUM_JOINTS:],
            "router_aux_loss": router_aux_loss,
        }
        if collect_diagnostics:
            result["layer_diagnostics"] = layer_diagnostics
        return result


def convert_dense_to_moe(dense_model: DenseVLA, config: MoEVLAConfig) -> MoEVLA:
    """Controlled Dense->MoE conversion (README "Recommended initialization strategy").

    Copies every shared component (vision/language/state encoders, fusion,
    action head) verbatim from the trained Dense model. For each
    Transformer layer, self-attention and both LayerNorms are copied
    verbatim; for layers in ``config.moe_layers``, the Dense FFN's weights
    are copied into EVERY expert (so all experts start as exact copies of
    each other and of the Dense FFN); the router is left at its random
    initialization. Layers not in ``config.moe_layers`` keep a plain dense
    FFN, also copied verbatim.

    With ``top_k=1`` and unweighted expert output (see ``models/moe.py``),
    this makes the converted model's forward pass numerically match the
    Dense model's almost exactly regardless of which (identical) expert
    the randomly-initialized router happens to pick -- verified by
    ``tests/test_moe_shapes.py`` and reported in the Step 6 final report.
    """
    if dense_model.config.hidden_dim != config.hidden_dim or dense_model.config.num_layers != config.num_layers:
        raise ValueError("Dense and MoE configs must share hidden_dim/num_layers for conversion")

    moe_model = MoEVLA(config)

    moe_model.vision_encoder.load_state_dict(dense_model.vision_encoder.state_dict())
    moe_model.language_encoder.load_state_dict(dense_model.language_encoder.state_dict())
    moe_model.state_encoder.load_state_dict(dense_model.state_encoder.state_dict())
    moe_model.fusion.load_state_dict(dense_model.fusion.state_dict())
    moe_model.action_head.load_state_dict(dense_model.action_head.state_dict())

    for layer_index, dense_layer in enumerate(dense_model.transformer.layers):
        moe_block = moe_model.transformer.blocks[layer_index]
        moe_block.self_attn.load_state_dict(dense_layer.self_attn.state_dict())
        moe_block.norm1.load_state_dict(dense_layer.norm1.state_dict())
        moe_block.norm2.load_state_dict(dense_layer.norm2.state_dict())

        if moe_block.is_moe:
            for expert in moe_block.ffn.experts:
                expert.linear1.load_state_dict(dense_layer.linear1.state_dict())
                expert.linear2.load_state_dict(dense_layer.linear2.state_dict())
            # Router is intentionally left at its random init.
        else:
            moe_block.ffn.linear1.load_state_dict(dense_layer.linear1.state_dict())
            moe_block.ffn.linear2.load_state_dict(dense_layer.linear2.state_dict())

    return moe_model


def parameter_accounting(model: MoEVLA) -> dict:
    """Total / trainable / active-per-token parameter counts (README "Active parameter count").

    "Active parameters per token" = every parameter that participates in
    computing that token's output: all non-MoE parameters, plus (for each
    MoE layer) only the ``top_k`` experts actually selected for that
    token -- not the full ``num_experts`` that merely occupy memory.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_expert_parameters = 0
    inactive_parameters_per_token = 0
    num_moe_layers = 0
    for block in model.transformer.blocks:
        if block.is_moe:
            num_moe_layers += 1
            one_expert_params = sum(p.numel() for p in block.ffn.experts[0].parameters())
            total_expert_parameters += one_expert_params * model.config.num_experts
            inactive_parameters_per_token += one_expert_params * (model.config.num_experts - model.config.top_k)

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "active_parameters_per_token": total - inactive_parameters_per_token,
        "total_expert_parameters": total_expert_parameters,
        "num_moe_layers": num_moe_layers,
    }
