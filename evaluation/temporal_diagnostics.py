"""Gripper-oscillation diagnostics -- the central Step 7 measurement.

Step 5/6 diagnosed gripper open/close oscillation near the grasp point as
the dominant closed-loop failure mode for both Dense and MoE. These
functions give that diagnosis a concrete number so Dense vs Temporal can
be compared directly, not just eyeballed from example trajectories.
"""

import numpy as np

GRIPPER_OPEN_THRESHOLD = 0.5


def gripper_switch_count(gripper_probabilities: list, threshold: float = GRIPPER_OPEN_THRESHOLD) -> int:
    """Number of open<->closed transitions in a thresholded gripper-probability sequence."""
    if len(gripper_probabilities) < 2:
        return 0
    is_open = np.asarray(gripper_probabilities) >= threshold
    return int(np.sum(is_open[1:] != is_open[:-1]))


def gripper_switch_rate(gripper_probabilities: list, threshold: float = GRIPPER_OPEN_THRESHOLD) -> float:
    """``gripper_switch_count`` normalized by the number of consecutive-tick
    transitions possible (``len - 1``); 0.0 for episodes too short to have any."""
    if len(gripper_probabilities) < 2:
        return 0.0
    return gripper_switch_count(gripper_probabilities, threshold) / (len(gripper_probabilities) - 1)


def windowed_gripper_switch_count(
    gripper_probabilities: list, center_index: int, window: int = 20, threshold: float = GRIPPER_OPEN_THRESHOLD
) -> int:
    """Switch count within ``+/-window`` ticks of ``center_index`` (e.g. the
    tick of minimum eef-cube distance, as a proxy for "near grasp") --
    lets oscillation specifically AROUND the grasp attempt be distinguished
    from oscillation spread evenly across the whole episode."""
    start = max(0, center_index - window)
    end = min(len(gripper_probabilities), center_index + window + 1)
    return gripper_switch_count(gripper_probabilities[start:end], threshold)


def summarize_gripper_stability(results: list) -> dict:
    """Aggregate gripper-switch statistics over a batch of
    ``evaluation.closed_loop.RolloutResult`` (uses ``gripper_probabilities``,
    already recorded by every closed-loop rollout regardless of policy type)."""
    counts = [gripper_switch_count(r.gripper_probabilities) for r in results]
    rates = [gripper_switch_rate(r.gripper_probabilities) for r in results]
    return {
        "mean_switch_count": float(np.mean(counts)) if counts else None,
        "median_switch_count": float(np.median(counts)) if counts else None,
        "max_switch_count": int(np.max(counts)) if counts else None,
        "mean_switch_rate": float(np.mean(rates)) if rates else None,
    }
