"""Pretrained vision backbone producing one pooled visual embedding per frame.

Frozen by default (see ``DenseVLAConfig.train_vision_encoder``): the Step 3
dataset has ~100 episodes, far too little to safely fine-tune an ImageNet
backbone from scratch without overfitting, so only the projection/fusion/
transformer layers built on top of it are trained initially.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform() -> T.Compose:
    """Deterministic preprocessing: PIL Image -> normalized (3, 224, 224) tensor.

    No augmentation -- see README "Image preprocessing" for why augmentation
    is deferred until after the baseline trains. Applied in the dataset
    loader (``dataset/torch_dataset.py``) and at inference
    (``models/policy.py``); the on-disk Step 3 PNGs are never modified.
    """
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class VisionEncoder(nn.Module):
    """Pretrained ResNet18 with its classification head removed.

    Args:
        trainable: If False (default), backbone parameters are frozen and
            kept in eval mode (so BatchNorm uses running statistics, not
            batch statistics) regardless of the outer model's train/eval
            mode.
    """

    def __init__(self, trainable: bool = False) -> None:
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.output_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.trainable = trainable
        if not trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``pixel_values``: ``(B, 3, 224, 224)`` -> pooled embedding ``(B, 512)``."""
        return self.backbone(pixel_values)

    def train(self, mode: bool = True) -> "VisionEncoder":
        super().train(mode)
        if not self.trainable:
            self.backbone.eval()
        return self
