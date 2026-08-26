"""The temporal-history padding/masking contract, shared by training
(``dataset/temporal_torch_dataset.py``) and runtime
(``models/temporal_policy.py``) so the two can never silently diverge --
see README "Padding at runtime": training/runtime padding must match
exactly, and this module is the single place that convention lives.

For a sample/window ending at the current timestep ``t`` with
``history_length = H``, the window covers observation indices
``t-H+1 .. t``. Each slot's PREVIOUS-ACTION representation is:

* the real recorded/issued action at that index, if the index is
  ``>= 0`` (inside the episode) AND it is not the last (current, ``t``)
  slot;
* ``NO_ACTION_VECTOR`` otherwise -- i.e. for (a) indices before episode
  start (left-padding) and (b) the current slot itself, since the action
  at ``t`` is exactly what's being predicted and including it would be
  target leakage (README "Target-leakage tests").

Observation slots (RGB/state) are always left-padded by repeating the
episode's first observation -- there is no masking equivalent for
observations, only for actions.
"""

import numpy as np

NUM_JOINTS = 7
ACTION_DIM = NUM_JOINTS + 1  # 7 joint targets (normalized) + 1 gripper (raw [0,1])

# "No action information available" sentinel: zero in normalized joint
# space (= the train-split mean joint target) and gripper=0.5 (halfway
# between closed/open -- "unknown"), chosen to be clearly distinguishable
# from any real recorded/issued action rather than silently reusing a
# plausible-looking real value.
NO_ACTION_VECTOR = np.concatenate([np.zeros(NUM_JOINTS, dtype=np.float64), [0.5]])


def source_indices(t: int, history_length: int) -> list:
    """Unclipped observation indices for the window ending at ``t``: ``[t-H+1, ..., t]``."""
    return [t - history_length + 1 + h for h in range(history_length)]


def clipped_indices(t: int, history_length: int) -> list:
    """``source_indices`` clipped to ``>= 0`` -- which real array index to read
    each observation slot from (left-padding repeats index 0)."""
    return [max(index, 0) for index in source_indices(t, history_length)]


def is_maskable_action_slot(source_index: int, slot: int, history_length: int) -> bool:
    """True if this window slot's previous-action must be ``NO_ACTION_VECTOR``:
    either it's before the episode start (``source_index < 0``) or it's the
    current/last slot (whatever the action there is IS the prediction target)."""
    return source_index < 0 or slot == history_length - 1


def build_action_window(
    available_actions_by_index: dict, t: int, history_length: int
) -> np.ndarray:
    """Build the ``(history_length, ACTION_DIM)`` previous-action window for
    the sample/window ending at ``t``.

    Args:
        available_actions_by_index: ``{index: action_vector (ACTION_DIM,)}``
            for real recorded/issued actions, keyed by their own timestep index.
        t: Current (last) timestep index of the window.
        history_length: Window length ``H``.
    """
    window = np.zeros((history_length, ACTION_DIM), dtype=np.float64)
    for slot, source_index in enumerate(source_indices(t, history_length)):
        if is_maskable_action_slot(source_index, slot, history_length):
            window[slot] = NO_ACTION_VECTOR
        else:
            window[slot] = available_actions_by_index[source_index]
    return window
