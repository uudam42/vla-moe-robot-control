"""PyTorch dataset for DAgger corrective samples: same fixed-length
history-window construction as ``dataset.temporal_torch_dataset`` (and the
SAME shared padding/masking contract, ``models.temporal_history``), except:

* the previous-action window is built from the MODEL's own issued actions
  (a DAgger episode's ``joint_targets``/``gripper_targets``, recorded by
  ``dagger.collector`` -- see README "DAgger sample semantics": do NOT
  replace previous model actions with expert actions), and
* the training target is the corrective EXPERT's action at ``t`` (from the
  episode's ``expert_labels.npz`` sidecar), not the model's own action at
  ``t``.

Only RETAINED timesteps (``dagger.disagreement.should_retain``) are indexed
by default -- see README "State sampling frequency".
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dagger.collector import load_expert_labels
from dataset.episode import load_episode
from models.temporal_history import build_action_window, clipped_indices
from training.normalization import ActionNormalizer, StateNormalizer

DEFAULT_HISTORY_LENGTH = 4


class TemporalDaggerCorrectiveDataset(Dataset):
    """Indexes retained ``(episode, timestep)`` pairs from one DAgger round.

    Args:
        root: A DAgger round directory (e.g. ``data/dagger/round_001``),
            containing an ``episodes/`` subdirectory of Step-3-format
            episode directories, each with a sibling ``expert_labels.npz``.
        episode_names: Optional explicit subset of episode directory names
            to include (e.g. for a train/corrective-val split of the same
            round -- see ``training/train_dagger.py``). Defaults to every
            episode directory under ``root/episodes``.
        retained_only: If True (default), only timesteps marked
            ``retained`` by the collector's sampling policy are indexed.
    """

    def __init__(
        self,
        root: Path,
        image_transform,
        tokenizer,
        max_length: int,
        state_normalizer: StateNormalizer,
        action_normalizer: ActionNormalizer,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        episode_names: list = None,
        retained_only: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.history_length = history_length

        episodes_dir = self.root / "episodes"
        if episode_names is None:
            episode_dirs = sorted(episodes_dir.glob("episode_*"))
        else:
            episode_dirs = [episodes_dir / name for name in sorted(episode_names)]

        self._episodes = []
        self._labels = []
        for episode_dir in episode_dirs:
            episode = load_episode(episode_dir)
            labels = load_expert_labels(episode_dir)
            self._episodes.append(episode)
            self._labels.append(labels)

        self._index = []  # (episode_index, t)
        for episode_index, (episode, labels) in enumerate(zip(self._episodes, self._labels)):
            retained = labels["retained"]
            for t in range(episode.length):
                if (not retained_only) or bool(retained[t]):
                    self._index.append((episode_index, t))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> dict:
        episode_index, t = self._index[index]
        episode = self._episodes[episode_index]
        labels = self._labels[episode_index]
        H = self.history_length

        obs_indices = clipped_indices(t, H)  # left-padded within THIS episode only

        pixel_values = []
        states = []
        for obs_index in obs_indices:
            rgb = np.array(Image.open(episode.rgb_frame_paths[obs_index]).convert("RGB"))
            pixel_values.append(self.image_transform(Image.fromarray(rgb)))
            states.append(self.state_normalizer.normalize(episode.states[obs_index]))

        # MODEL-issued action history -- episode.joint_targets/gripper_targets
        # ARE the model's own executed actions (see dagger/collector.py),
        # never the expert's. Indices >= t are never included (no future
        # leakage) and slot H-1 (current, t) is always masked by
        # build_action_window regardless of what's in this dict.
        available_actions = {
            i: np.concatenate(
                [self.action_normalizer.normalize_joints(episode.joint_targets[i]), [episode.gripper_targets[i]]]
            )
            for i in range(t + 1)
        }
        action_window = build_action_window(available_actions, t, H)

        tokenized = self.tokenizer(
            episode.instruction, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length,
        )

        # Target is exactly the corrective expert's label AT t -- never the
        # model's own action at t (that would just be identity imitation).
        target_joint_normalized = self.action_normalizer.normalize_joints(labels["expert_joint_targets"][t])
        target_gripper = float(labels["expert_gripper_targets"][t])

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
