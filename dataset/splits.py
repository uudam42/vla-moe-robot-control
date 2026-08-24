"""Deterministic episode-level train/val/test split.

Splitting must happen per-episode, never per-timestep: adjacent timesteps
are nearly identical, so a random per-frame split would leak near-duplicate
frames between train and validation and make validation artificially easy.
"""

import json
from pathlib import Path

import numpy as np

SPLITS_FILENAME = "splits.json"


def make_splits(
    episode_names: list,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict:
    """Shuffle episode names deterministically and cut into train/val/test.

    Args:
        episode_names: Episode directory names (e.g. ``"episode_000003"``),
            already restricted to whichever episodes are eligible (i.e.
            successful ones).
        seed: Seeds the shuffle; same input + seed -> same split.
        train_frac: Fraction of episodes assigned to "train".
        val_frac: Fraction assigned to "val"; the remainder goes to "test".

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}``, each a sorted
        list of episode names. Every input episode appears in exactly one.
    """
    if not (0.0 < train_frac < 1.0) or not (0.0 <= val_frac < 1.0) or train_frac + val_frac > 1.0:
        raise ValueError(f"invalid train_frac={train_frac}, val_frac={val_frac}")

    names = sorted(episode_names)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(names).tolist()

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def save_splits(splits: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
