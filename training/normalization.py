"""Train-split-only normalization for the 23D state and 7D joint targets.

Both normalizers must be fit from the training split alone (see
``fit_normalizers_from_split``) and persisted with every checkpoint --
inference must use exactly the statistics the model was trained with, not
recomputed ones (see ``models/policy.py``).

The gripper target is deliberately NOT normalized: it's already in
``[0, 1]`` and treated as a binary open/closed target via
``BCEWithLogitsLoss`` (see ``training/losses.py``), so mean/std scaling
would just distort a value that's already meaningful on its own scale.
"""

from dataclasses import dataclass

import numpy as np

from dataset.episode import load_episode
from dataset.splits import load_splits

EPS = 1e-6


@dataclass
class StateNormalizer:
    mean: np.ndarray  # (23,)
    std: np.ndarray  # (23,)

    def normalize(self, state: np.ndarray) -> np.ndarray:
        return (state - self.mean) / self.std

    def denormalize(self, state_normalized: np.ndarray) -> np.ndarray:
        return state_normalized * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "StateNormalizer":
        return cls(mean=np.array(data["mean"], dtype=np.float64), std=np.array(data["std"], dtype=np.float64))

    @classmethod
    def fit(cls, states: np.ndarray) -> "StateNormalizer":
        mean = states.mean(axis=0)
        std = states.std(axis=0)
        std = np.where(std < EPS, 1.0, std)  # near-constant dims: don't divide by ~0
        return cls(mean=mean, std=std)


@dataclass
class ActionNormalizer:
    joint_mean: np.ndarray  # (7,)
    joint_std: np.ndarray  # (7,)

    def normalize_joints(self, joint_targets: np.ndarray) -> np.ndarray:
        return (joint_targets - self.joint_mean) / self.joint_std

    def denormalize_joints(self, joint_targets_normalized: np.ndarray) -> np.ndarray:
        return joint_targets_normalized * self.joint_std + self.joint_mean

    def to_dict(self) -> dict:
        return {"joint_mean": self.joint_mean.tolist(), "joint_std": self.joint_std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "ActionNormalizer":
        return cls(
            joint_mean=np.array(data["joint_mean"], dtype=np.float64),
            joint_std=np.array(data["joint_std"], dtype=np.float64),
        )

    @classmethod
    def fit(cls, joint_targets: np.ndarray) -> "ActionNormalizer":
        mean = joint_targets.mean(axis=0)
        std = joint_targets.std(axis=0)
        std = np.where(std < EPS, 1.0, std)
        return cls(joint_mean=mean, joint_std=std)


def fit_normalizers_from_split(root, split: str = "train") -> tuple:
    """Load every episode in ``split`` (states + joint targets only, no RGB
    decoding) and fit both normalizers. Train-split-only by construction --
    callers must not pass "val"/"test" here.
    """
    root = str(root)
    splits = load_splits(f"{root}/splits.json")
    episode_names = splits[split]

    all_states = []
    all_joint_targets = []
    for name in episode_names:
        episode = load_episode(f"{root}/successful/{name}")
        all_states.append(episode.states)
        all_joint_targets.append(episode.joint_targets)

    states = np.concatenate(all_states, axis=0)
    joint_targets = np.concatenate(all_joint_targets, axis=0)

    return StateNormalizer.fit(states), ActionNormalizer.fit(joint_targets)
