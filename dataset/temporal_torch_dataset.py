"""PyTorch dataset for the Step 7 Temporal Dense VLA: builds fixed-length
observation/action history windows from the existing Step 3 episodes.

No new demonstrations are generated -- this only re-slices the same
per-timestep arrays ``dataset.loader.DemonstrationDataset`` already
serves flat, into ``(history ending at t) -> action_t`` windows, using
the padding/masking contract in ``models/temporal_history.py`` so
training and runtime never diverge. History windows never cross episode
boundaries (each episode's own arrays are indexed independently) and
never include timestep ``t``'s own action (see
``models.temporal_history`` docstring for why).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset.episode import load_episode
from dataset.splits import load_splits
from models.temporal_history import build_action_window, clipped_indices
from training.normalization import ActionNormalizer, StateNormalizer

DEFAULT_HISTORY_LENGTH = 4


class TemporalDemonstrationDataset(Dataset):
    """Indexes every ``(episode, timestep)`` pair as one temporal-window sample.

    Args:
        root: Dataset root (containing ``successful/``, ``splits.json``).
        split: ``"train"``, ``"val"``, or ``"test"`` -- restricted to that
            split's episodes only (README "Data split": exactly the
            existing 80/10/10 episode-level split, no re-shuffling).
        image_transform: PIL Image -> ``Tensor[3,H,W]``.
        tokenizer: HuggingFace tokenizer.
        max_length: Tokenizer padding/truncation length.
        state_normalizer, action_normalizer: Fit on the TRAIN split only
            (same objects Dense/MoE use -- see README "Normalization").
        history_length: Window length ``H`` (default 4).
    """

    def __init__(
        self,
        root: Path,
        split: str,
        image_transform,
        tokenizer,
        max_length: int,
        state_normalizer: StateNormalizer,
        action_normalizer: ActionNormalizer,
        history_length: int = DEFAULT_HISTORY_LENGTH,
    ) -> None:
        self.root = Path(root)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.history_length = history_length

        splits = load_splits(self.root / "splits.json")
        episode_names = splits[split]

        self._episodes = [load_episode(self.root / "successful" / name) for name in episode_names]

        self._index = []  # (episode_index, t)
        for episode_index, episode in enumerate(self._episodes):
            for t in range(episode.length):
                self._index.append((episode_index, t))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> dict:
        episode_index, t = self._index[index]
        episode = self._episodes[episode_index]
        H = self.history_length

        obs_indices = clipped_indices(t, H)  # left-padded: repeats index 0 for indices < 0

        pixel_values = []
        states = []
        for obs_index in obs_indices:
            rgb = np.array(Image.open(episode.rgb_frame_paths[obs_index]).convert("RGB"))
            pixel_values.append(self.image_transform(Image.fromarray(rgb)))
            states.append(self.state_normalizer.normalize(episode.states[obs_index]))

        # Real actions available at THIS episode's own indices only -- keyed
        # by their true (unclipped) timestep index so build_action_window
        # can correctly identify and mask out-of-episode / current-slot entries.
        available_actions = {
            i: np.concatenate(
                [self.action_normalizer.normalize_joints(episode.joint_targets[i]), [episode.gripper_targets[i]]]
            )
            for i in range(episode.length)
        }
        action_window = build_action_window(available_actions, t, H)

        tokenized = self.tokenizer(
            episode.instruction, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length,
        )

        target_joint_normalized = self.action_normalizer.normalize_joints(episode.joint_targets[t])
        target_gripper = episode.gripper_targets[t]

        return {
            "pixel_values": torch.stack(pixel_values),  # (H, 3, 224, 224)
            "states": torch.from_numpy(np.stack(states)).float(),  # (H, 23)
            "previous_actions": torch.from_numpy(action_window).float(),  # (H, 8)
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "joint_targets_normalized": torch.from_numpy(target_joint_normalized).float(),
            "gripper_target": torch.tensor(target_gripper, dtype=torch.float32),
            "instruction": episode.instruction,
        }
