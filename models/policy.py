"""Inference adapter: DenseVLA model + preprocessing -> RobotAction.

This is the only place Step 4 code produces a ``RobotAction`` -- it never
calls MuJoCo (see README "No closed-loop claims yet"). Closed-loop
deployment (``env.step(policy.predict(...))``) is Step 5.
"""

import numpy as np
import torch
from PIL import Image

from control.action import RobotAction
from models.dense_vla import DenseVLA, DenseVLAConfig
from models.language_encoder import load_tokenizer
from models.vision_encoder import build_image_transform
from observations.observation import Observation
from training.checkpoint import load_checkpoint
from training.normalization import ActionNormalizer, StateNormalizer


class DenseVLAPolicy:
    """Wraps a trained ``DenseVLA`` with the exact preprocessing it was trained with."""

    def __init__(
        self,
        model: DenseVLA,
        state_normalizer: StateNormalizer,
        action_normalizer: ActionNormalizer,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.device = device
        self.tokenizer = load_tokenizer(model.config.language_backbone)
        self.image_transform = build_image_transform()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device: torch.device = None) -> "DenseVLAPolicy":
        device = device or torch.device("cpu")
        checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))

        config = DenseVLAConfig.from_dict(checkpoint["config"])
        model = DenseVLA(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        state_normalizer = StateNormalizer.from_dict(checkpoint["state_normalizer"])
        action_normalizer = ActionNormalizer.from_dict(checkpoint["action_normalizer"])

        return cls(model, state_normalizer, action_normalizer, device)

    @torch.inference_mode()
    def predict(self, observation: Observation, instruction: str) -> RobotAction:
        """Preprocess -> forward -> denormalize -> ``RobotAction``. No MuJoCo calls.

        Runtime-safe for closed-loop use (Step 5): always runs in eval mode
        under ``torch.inference_mode()``, so dropout/backbone BatchNorm
        stay fixed and no autograd graph is built per control tick.
        """
        self.model.eval()
        pixel_values = self.image_transform(Image.fromarray(observation.rgb)).unsqueeze(0).to(self.device)

        state_normalized = self.state_normalizer.normalize(observation.state)
        state_tensor = torch.from_numpy(state_normalized).float().unsqueeze(0).to(self.device)

        tokenized = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.model.config.max_instruction_length,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        output = self.model(pixel_values, input_ids, attention_mask, state_tensor)

        joint_targets_normalized = output["joint_targets_normalized"][0].cpu().numpy()
        joint_targets = self.action_normalizer.denormalize_joints(joint_targets_normalized)

        gripper_target = torch.sigmoid(output["gripper_logit"][0, 0]).cpu().item()
        gripper_target = float(np.clip(gripper_target, 0.0, 1.0))

        return RobotAction(
            joint_targets=joint_targets.astype(np.float64),
            gripper_target=gripper_target,
        )
