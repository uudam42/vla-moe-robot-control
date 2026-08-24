"""Timestep-level and episode-level dataset loading.

No PyTorch dependency here -- Step 4 wraps ``DemonstrationDataset`` in a
``torch.utils.data.Dataset``. This module only does NumPy/PIL/plain Python.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .episode import METADATA_FILENAME, TRAJECTORY_FILENAME, frame_path, load_episode

__all__ = ["DemonstrationDataset", "load_episode"]


class DemonstrationDataset:
    """Indexes every (episode, timestep) pair as one flat, lazily-loaded sample.

    Args:
        root: Dataset root (containing ``successful/``, ``manifest.json``,
            optionally ``splits.json``).
        split: One of ``"train"``, ``"val"``, ``"test"`` to restrict to a
            split recorded in ``root/splits.json``; ``None`` uses every
            episode under ``split_subdir``.
        split_subdir: Subdirectory holding the episodes to index (only
            ``"successful"`` episodes form the official dataset -- see
            README "Success filtering").
    """

    def __init__(self, root: Path, split: str = None, split_subdir: str = "successful") -> None:
        self.root = Path(root)
        episodes_root = self.root / split_subdir
        if not episodes_root.is_dir():
            raise FileNotFoundError(f"No '{split_subdir}' directory under {self.root}")

        if split is not None:
            splits_path = self.root / "splits.json"
            if not splits_path.exists():
                raise FileNotFoundError(f"No splits.json under {self.root}")
            with open(splits_path, "r", encoding="utf-8") as f:
                splits = json.load(f)
            if split not in splits:
                raise KeyError(f"Unknown split '{split}', have {sorted(splits)}")
            episode_names = splits[split]
        else:
            episode_names = sorted(p.name for p in episodes_root.glob("episode_*") if p.is_dir())

        self._episode_dirs = [episodes_root / name for name in episode_names]
        if not self._episode_dirs:
            raise ValueError(f"No episodes found for split={split!r} under {episodes_root}")

        self._metadata_by_dir = {}
        lengths = []
        for episode_dir in self._episode_dirs:
            with open(episode_dir / METADATA_FILENAME, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self._metadata_by_dir[episode_dir] = metadata
            lengths.append(int(metadata["episode_length"]))

        self._cumulative = np.concatenate([[0], np.cumsum(lengths)])
        self._trajectory_cache = {}

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    @property
    def num_episodes(self) -> int:
        return len(self._episode_dirs)

    def locate(self, index: int) -> tuple:
        """Resolve a flat sample index to ``(episode_dir, timestep)``."""
        if index < 0:
            index += len(self)
        if not (0 <= index < len(self)):
            raise IndexError(f"index {index} out of range for dataset of size {len(self)}")
        episode_idx = int(np.searchsorted(self._cumulative, index, side="right") - 1)
        timestep = index - int(self._cumulative[episode_idx])
        return self._episode_dirs[episode_idx], timestep

    def __getitem__(self, index: int) -> dict:
        episode_dir, t = self.locate(index)
        arrays = self._trajectory_arrays(episode_dir)
        metadata = self._metadata_by_dir[episode_dir]

        rgb = np.array(Image.open(frame_path(episode_dir, t)).convert("RGB"))
        action = np.concatenate(
            [arrays["joint_targets"][t], [arrays["gripper_targets"][t]]]
        ).astype(np.float64)

        return {
            "rgb": rgb,
            "state": arrays["states"][t],
            "instruction": metadata["instruction"],
            "action": action,
        }

    def _trajectory_arrays(self, episode_dir: Path) -> dict:
        cached = self._trajectory_cache.get(episode_dir)
        if cached is None:
            with np.load(episode_dir / TRAJECTORY_FILENAME) as npz:
                cached = {
                    "states": npz["states"],
                    "joint_targets": npz["joint_targets"],
                    "gripper_targets": npz["gripper_targets"],
                }
            self._trajectory_cache[episode_dir] = cached
        return cached
