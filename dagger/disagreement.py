"""Model/expert action-disagreement diagnostics and DAgger retention policy.

Pure functions over two ``RobotAction``s -- no simulator, no model, no I/O
-- so they're trivial to unit test and reuse between collection
(``dagger/collector.py``) and any diagnostic evaluation tooling
(``evaluation/dagger_diagnostics.py``).
"""

import numpy as np

from control.action import RobotAction

GRIPPER_DECISION_THRESHOLD = 0.5


def compute_disagreement(model_action: RobotAction, expert_action: RobotAction) -> dict:
    """Joint-space and gripper-decision disagreement between one model
    action and the corrective expert's label for the same state."""
    diff = model_action.joint_targets - expert_action.joint_targets
    joint_l2 = float(np.linalg.norm(diff))
    joint_mae = float(np.mean(np.abs(diff)))
    model_open = model_action.gripper_target >= GRIPPER_DECISION_THRESHOLD
    expert_open = expert_action.gripper_target >= GRIPPER_DECISION_THRESHOLD
    gripper_disagreement = bool(model_open != expert_open)

    return {
        "joint_l2": joint_l2,
        "joint_mae": joint_mae,
        "gripper_disagreement": gripper_disagreement,
    }


def should_retain(
    tick: int,
    sample_every: int,
    gripper_disagreement: bool,
    joint_l2: float,
    joint_l2_threshold: float,
) -> bool:
    """DAgger sample-retention rule (README "State sampling frequency" /
    "Disagreement-based sampling"): keep a periodic backbone (every
    ``sample_every`` ticks) PLUS any tick where the model/expert
    meaningfully disagree, so corrective states aren't diluted by
    thousands of near-identical easy states."""
    if sample_every <= 0:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")
    if tick % sample_every == 0:
        return True
    if gripper_disagreement:
        return True
    if joint_l2 > joint_l2_threshold:
        return True
    return False
