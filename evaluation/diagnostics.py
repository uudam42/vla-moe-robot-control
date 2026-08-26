"""Optional failure-analysis diagnostics -- not run by default during
evaluation (see README "Distribution-shift diagnosis").

``nearest_training_state_distance`` is a simple nearest-neighbor Euclidean
distance in normalized state space against a reference set of training
states -- not a sophisticated retrieval system, just enough to see
whether a rollout gradually drifts off the training distribution
(classic behavior-cloning compounding error) before it fails.
"""

import numpy as np

from dataset.episode import load_episode
from dataset.splits import load_splits
from training.normalization import StateNormalizer


def load_reference_states(data_root, split: str = "train", max_episodes: int = None) -> np.ndarray:
    """Raw (un-normalized) states from every recorded timestep in ``split``."""
    splits = load_splits(f"{data_root}/splits.json")
    episode_names = splits[split]
    if max_episodes is not None:
        episode_names = episode_names[:max_episodes]

    all_states = [load_episode(f"{data_root}/successful/{name}").states for name in episode_names]
    return np.concatenate(all_states, axis=0)


def nearest_training_state_distance(
    state: np.ndarray, reference_states: np.ndarray, state_normalizer: StateNormalizer
) -> float:
    """Distance (in normalized state space) from ``state`` to its nearest reference state."""
    normalized_query = state_normalizer.normalize(state)
    normalized_reference = state_normalizer.normalize(reference_states)
    distances = np.linalg.norm(normalized_reference - normalized_query, axis=1)
    return float(distances.min())


def trajectory_drift(
    states: np.ndarray, reference_states: np.ndarray, state_normalizer: StateNormalizer
) -> list:
    """Per-timestep nearest-training-state distance for a whole rollout.

    A rising trend over the episode is the signature of compounding
    behavior-cloning error: the policy starts near the training
    distribution and drifts away as its own imperfect actions accumulate.
    """
    normalized_reference = state_normalizer.normalize(reference_states)
    normalized_states = state_normalizer.normalize(states)
    distances = []
    for state in normalized_states:
        distances.append(float(np.linalg.norm(normalized_reference - state, axis=1).min()))
    return distances
