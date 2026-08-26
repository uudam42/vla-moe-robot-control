"""Checkpoint save/load. A checkpoint is self-contained: it must be enough
to reconstruct the model architecture, restore preprocessing (normalization
stats), and resume optimization -- not just raw weights (see README
"Checkpointing").
"""

from pathlib import Path

import torch

from models.dense_vla import DenseVLAConfig
from training.normalization import ActionNormalizer, StateNormalizer


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch: int,
    config: DenseVLAConfig,
    state_normalizer: StateNormalizer,
    action_normalizer: ActionNormalizer,
    val_metrics: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config.to_dict(),
            "state_normalizer": state_normalizer.to_dict(),
            "action_normalizer": action_normalizer.to_dict(),
            "val_metrics": val_metrics,
        },
        path,
    )


def load_checkpoint(path, map_location: str = "cpu") -> dict:
    """Returns the raw checkpoint dict. Callers reconstruct model/optimizer/
    normalizers from it -- see ``training/train.py``'s ``--resume`` handling
    and ``models/policy.py::DenseVLAPolicy.from_checkpoint``.
    """
    return torch.load(path, map_location=map_location, weights_only=False)
