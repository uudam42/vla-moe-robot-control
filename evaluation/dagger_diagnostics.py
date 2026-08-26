"""Lightweight, POST-HOC exposure-bias diagnostics (README "Exposure-bias
diagnostics" / "Recovery analysis"): given trajectories already saved by a
closed-loop evaluation run (``--save-trajectories``), how far did the
policy's states drift from the expert training distribution, and did any
large model/expert disagreement episodes recover?

Deliberately small (README "Do not build a large retrieval system" / "Do
not overengineer"): a simple normalized-state nearest-neighbor distance
against a subsample of TRAIN-split expert states, and a straightforward
spike-then-fall recovery-event detector. Qualitative/exploratory only --
never used to alter any action or metric that feeds the headline success
numbers.
"""

import numpy as np

from dataset.episode import load_episode
from dataset.splits import load_splits
from training.normalization import StateNormalizer

DEFAULT_MAX_REFERENCE_STATES = 2000


def build_reference_expert_states(
    data_root, state_normalizer: StateNormalizer, split: str = "train",
    max_samples: int = DEFAULT_MAX_REFERENCE_STATES, seed: int = 0,
) -> np.ndarray:
    """A (subsampled) matrix of NORMALIZED expert states from one split,
    used as the reference distribution for nearest-neighbor distance."""
    root = str(data_root)
    splits = load_splits(f"{root}/splits.json")
    episode_names = splits[split]

    all_states = []
    for name in episode_names:
        episode = load_episode(f"{root}/successful/{name}")
        all_states.append(episode.states)
    states = np.concatenate(all_states, axis=0)
    normalized = np.stack([state_normalizer.normalize(s) for s in states])

    if normalized.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(normalized.shape[0], size=max_samples, replace=False)
        normalized = normalized[indices]
    return normalized


def nearest_expert_state_distance(state_normalized: np.ndarray, reference_states: np.ndarray) -> float:
    """Euclidean distance from one normalized state to its nearest neighbor
    in ``reference_states``."""
    diffs = reference_states - state_normalized[None, :]
    return float(np.min(np.linalg.norm(diffs, axis=1)))


def trajectory_exposure_bias(states_normalized: np.ndarray, reference_states: np.ndarray) -> list:
    """Nearest-neighbor distance to the expert reference distribution at
    every tick of one trajectory. ``states_normalized``: ``(T, state_dim)``."""
    diffs = reference_states[None, :, :] - states_normalized[:, None, :]  # (T, R, D)
    distances = np.linalg.norm(diffs, axis=2)  # (T, R)
    return np.min(distances, axis=1).tolist()


def summarize_exposure_bias(distance_sequences: list) -> dict:
    """Aggregate ``trajectory_exposure_bias`` output across many episodes."""
    if not distance_sequences:
        return {"mean_distance": None, "max_distance": None, "num_trajectories": 0}
    all_distances = np.concatenate([np.asarray(d) for d in distance_sequences if len(d) > 0])
    if all_distances.size == 0:
        return {"mean_distance": None, "max_distance": None, "num_trajectories": len(distance_sequences)}
    return {
        "mean_distance": float(all_distances.mean()),
        "median_distance": float(np.median(all_distances)),
        "max_distance": float(all_distances.max()),
        "num_trajectories": len(distance_sequences),
    }


def detect_recovery_events(
    joint_l2_disagreement: list, spike_threshold: float, recovered_threshold: float, min_gap_ticks: int = 3,
) -> list:
    """Find simple "spike then fall" recovery events in a per-tick
    model/expert disagreement sequence: disagreement exceeds
    ``spike_threshold`` at tick ``i``, then falls back below
    ``recovered_threshold`` within a later window.

    Returns a list of ``{"spike_tick": i, "recovered_tick": j}`` dicts.
    Purely descriptive/qualitative (README "Do not overengineer") -- no
    claim of causality, just where a large disagreement was followed by a
    smaller one.
    """
    values = np.asarray(joint_l2_disagreement, dtype=np.float64)
    events = []
    i = 0
    while i < len(values):
        if values[i] > spike_threshold:
            spike_tick = i
            recovered_tick = None
            for j in range(i + min_gap_ticks, len(values)):
                if values[j] < recovered_threshold:
                    recovered_tick = j
                    break
            events.append({"spike_tick": spike_tick, "recovered_tick": recovered_tick})
            i = recovered_tick if recovered_tick is not None else i + 1
        i += 1
    return events
