"""Training-run configuration (hyperparameters + CLI defaults), device
selection, and seeding. Architecture configuration lives separately in
``models.dense_vla.DenseVLAConfig`` -- this dataclass is everything about
*how* to train, not the model shape.
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch

DEFAULT_DATA_ROOT = "data/demonstrations"
DEFAULT_OUTPUT_ROOT = "outputs/training"


@dataclass
class TrainingConfig:
    data: str = DEFAULT_DATA_ROOT
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    seed: int = 42
    device: str = None  # None -> auto-select (cuda > mps > cpu)
    num_workers: int = 2
    output: str = DEFAULT_OUTPUT_ROOT
    overfit_samples: int = 0  # 0 disables overfit mode
    resume: str = None
    gripper_loss_weight: float = 1.0
    train_vision_encoder: bool = False
    train_language_encoder: bool = False
    run_name: str = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def resolve_device(requested: str = None) -> torch.device:
    """``requested`` overrides auto-selection when given. Otherwise: CUDA > MPS > CPU."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
