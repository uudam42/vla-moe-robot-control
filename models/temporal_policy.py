"""Inference adapter for the Temporal Dense VLA: maintains an internal
observation/action history buffer so the runtime API stays as simple as
Dense/MoE's (``predict(observation, instruction) -> RobotAction``) despite
the model needing a window of past inputs.

Critical distinction from training (README "Teacher-forcing distinction"):
during training, the previous-action window comes from the EXPERT's
recorded demonstration actions (teacher forcing). At runtime, the
previous-action window is built from THIS POLICY'S OWN previously issued
actions -- never the expert's. This is a real, acknowledged source of
train/inference distribution shift, not hidden.
"""

import numpy as np
import torch
from PIL import Image

from control.action import RobotAction
from models.language_encoder import load_tokenizer
from models.temporal_history import build_action_window
from models.temporal_vla import TemporalDenseVLA, TemporalDenseVLAConfig
from models.vision_encoder import build_image_transform
from observations.observation import Observation
from training.checkpoint import load_checkpoint
from training.normalization import ActionNormalizer, StateNormalizer


class TemporalDenseVLAPolicy:
    """Wraps a trained ``TemporalDenseVLA`` with a rolling history buffer.

    Call :meth:`reset` at the start of every episode (README "Episode
    reset") -- history must never leak across episodes.
    """

    def __init__(
        self,
        model: TemporalDenseVLA,
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
        self.history_length = model.config.history_length
        self.reset()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device: torch.device = None) -> "TemporalDenseVLAPolicy":
        device = device or torch.device("cpu")
        checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
        config = TemporalDenseVLAConfig.from_dict(checkpoint["config"])
        model = TemporalDenseVLA(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        state_normalizer = StateNormalizer.from_dict(checkpoint["state_normalizer"])
        action_normalizer = ActionNormalizer.from_dict(checkpoint["action_normalizer"])
        return cls(model, state_normalizer, action_normalizer, device)

    def reset(self) -> None:
        """Clear all history. Must be called once per episode, right after
        ``env.reset()`` -- see README "Episode reset" /
        ``tests/test_temporal_no_privileged_inputs.py``."""
        self._rgb_history = []  # raw uint8 arrays, oldest first, len <= history_length
        self._state_history = []  # raw (unnormalized) 23D arrays, aligned with _rgb_history
        # This policy's OWN previously issued action vectors (normalized-joint
        # + raw gripper, 8D), keyed by the timestep index they were issued at.
        self._issued_actions_by_index: dict = {}
        self._tick = 0

    @torch.inference_mode()
    def predict(self, observation: Observation, instruction: str) -> RobotAction:
        """Preprocess (with rolling history) -> forward -> denormalize -> ``RobotAction``.

        No MuJoCo calls, no privileged simulator state -- only
        ``observation`` (current RGB + 23D state), ``instruction``, and
        this policy's own stored past observations/actions.
        """
        self.model.eval()
        t = self._tick

        self._rgb_history.append(observation.rgb)
        self._state_history.append(observation.state.copy())
        if len(self._rgb_history) > self.history_length:
            self._rgb_history.pop(0)
            self._state_history.pop(0)

        H = self.history_length
        pad_count = H - len(self._rgb_history)
        rgb_window = [self._rgb_history[0]] * pad_count + list(self._rgb_history)
        state_window = [self._state_history[0]] * pad_count + list(self._state_history)

        pixel_values = torch.stack(
            [self.image_transform(Image.fromarray(rgb)) for rgb in rgb_window]
        ).unsqueeze(0).to(self.device)  # (1, H, 3, 224, 224)

        state_normalized = np.stack([self.state_normalizer.normalize(s) for s in state_window])
        states = torch.from_numpy(state_normalized).float().unsqueeze(0).to(self.device)  # (1, H, 23)

        action_window = build_action_window(self._issued_actions_by_index, t, H)
        previous_actions = torch.from_numpy(action_window).float().unsqueeze(0).to(self.device)  # (1, H, 8)

        tokenized = self.tokenizer(
            instruction, return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.model.config.max_instruction_length,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        output = self.model(pixel_values, states, previous_actions, input_ids, attention_mask)

        joint_targets_normalized = output["joint_targets_normalized"][0].cpu().numpy()
        joint_targets = self.action_normalizer.denormalize_joints(joint_targets_normalized)
        gripper_target = torch.sigmoid(output["gripper_logit"][0, 0]).cpu().item()
        gripper_target = float(np.clip(gripper_target, 0.0, 1.0))

        action = RobotAction(joint_targets=joint_targets.astype(np.float64), gripper_target=gripper_target)

        # Store THIS POLICY'S OWN issued action (not any expert action) for
        # future ticks' history -- see module docstring "Teacher-forcing distinction".
        self._issued_actions_by_index[t] = np.concatenate([joint_targets_normalized, [gripper_target]])
        self._tick += 1

        return action
