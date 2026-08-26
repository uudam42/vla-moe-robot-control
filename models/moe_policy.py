"""Inference adapter for the MoE model: MoEVLA + preprocessing -> RobotAction.

Mirrors ``models/policy.py::DenseVLAPolicy`` exactly in every way except
the underlying model class -- same preprocessing, same checkpoint format
shape, same never-touches-MuJoCo guarantee. ``predict()`` is the only
method used for closed-loop control and matches the input/output contract
required by ``env.step()``; ``predict_with_routing()`` is an
evaluation-only addition that also returns per-layer routing diagnostics
for post-hoc analysis (README "Router analysis during closed loop") --
never used to influence the action itself.
"""

import numpy as np
import torch
from PIL import Image

from control.action import RobotAction
from models.language_encoder import load_tokenizer
from models.moe_vla import MoEVLA, MoEVLAConfig
from models.vision_encoder import build_image_transform
from observations.observation import Observation
from training.checkpoint import load_checkpoint
from training.normalization import ActionNormalizer, StateNormalizer


class MoEVLAPolicy:
    """Wraps a trained ``MoEVLA`` with the exact preprocessing it was trained with."""

    def __init__(
        self,
        model: MoEVLA,
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
    def from_checkpoint(cls, checkpoint_path, device: torch.device = None) -> "MoEVLAPolicy":
        device = device or torch.device("cpu")
        checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))

        config = MoEVLAConfig.from_dict(checkpoint["config"])
        model = MoEVLA(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        state_normalizer = StateNormalizer.from_dict(checkpoint["state_normalizer"])
        action_normalizer = ActionNormalizer.from_dict(checkpoint["action_normalizer"])

        return cls(model, state_normalizer, action_normalizer, device)

    def _prepare_inputs(self, observation: Observation, instruction: str) -> tuple:
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
        return pixel_values, input_ids, attention_mask, state_tensor

    def _to_robot_action(self, output: dict) -> RobotAction:
        joint_targets_normalized = output["joint_targets_normalized"][0].cpu().numpy()
        joint_targets = self.action_normalizer.denormalize_joints(joint_targets_normalized)
        gripper_target = torch.sigmoid(output["gripper_logit"][0, 0]).cpu().item()
        gripper_target = float(np.clip(gripper_target, 0.0, 1.0))
        return RobotAction(joint_targets=joint_targets.astype(np.float64), gripper_target=gripper_target)

    @torch.inference_mode()
    def predict(self, observation: Observation, instruction: str) -> RobotAction:
        """Preprocess -> forward -> denormalize -> ``RobotAction``. No MuJoCo calls.

        Input contract identical to ``DenseVLAPolicy.predict()``: only
        ``observation`` (RGB + 23D state) and ``instruction``. No routing
        diagnostics are computed on this path (``collect_diagnostics=False``)
        to keep closed-loop inference at the same cost as Dense.
        """
        self.model.eval()
        pixel_values, input_ids, attention_mask, state_tensor = self._prepare_inputs(observation, instruction)
        output = self.model(pixel_values, input_ids, attention_mask, state_tensor, collect_diagnostics=False)
        return self._to_robot_action(output)

    @torch.inference_mode()
    def predict_with_routing(self, observation: Observation, instruction: str) -> tuple:
        """Evaluation-only variant: also returns per-MoE-layer routing diagnostics.

        Returns ``(RobotAction, layer_diagnostics)``. Never used in the
        core closed-loop control path -- only by
        ``evaluation/moe_diagnostics.py`` for post-hoc router analysis.
        """
        self.model.eval()
        pixel_values, input_ids, attention_mask, state_tensor = self._prepare_inputs(observation, instruction)
        output = self.model(pixel_values, input_ids, attention_mask, state_tensor, collect_diagnostics=True)
        return self._to_robot_action(output), output["layer_diagnostics"]
