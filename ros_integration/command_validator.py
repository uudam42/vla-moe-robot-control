"""Minimal command-validation layer (README "Action validation" / "Minimal
safety boundary") -- operates on RAW deserialized action fields (not yet a
``RobotAction``), so a malformed message is rejected with a clear reason
instead of raising an uncaught exception from ``RobotAction.__post_init__``.

Deliberately small: shape/finiteness/range/extreme-delta checks only. The
existing environment-side joint-limit clamping
(``SimulationEnvironment.step``) is reused as-is and NOT duplicated here
(README "Do not duplicate conflicting joint-limit definitions"). A larger
industrial safety supervisor is explicitly out of scope for Step 9.
"""

from dataclasses import dataclass

import numpy as np

from control.action import RobotAction

DEFAULT_NUM_JOINTS = 7


@dataclass
class ValidationResult:
    valid: bool
    reason: str = None


class CommandRejected(Exception):
    """Raised by ``CommandValidator.build_action`` for an invalid command."""


class CommandValidator:
    """Validates raw ``(joint_targets, gripper_target)`` fields before they
    become a ``RobotAction`` and reach a ``RobotBackend``.

    Args:
        num_joints: Expected joint count (7 for the Panda arm).
        max_joint_delta: If given, the maximum allowed per-joint change
            (radians) from ``previous_joint_positions``, when supplied to
            ``validate``/``build_action``. ``None`` disables this check
            (e.g. when no previous command is known yet).
    """

    def __init__(self, num_joints: int = DEFAULT_NUM_JOINTS, max_joint_delta: float = None) -> None:
        self.num_joints = num_joints
        self.max_joint_delta = max_joint_delta

    def validate(
        self, joint_targets, gripper_target, previous_joint_positions: np.ndarray = None
    ) -> ValidationResult:
        try:
            joint_targets = np.asarray(joint_targets, dtype=np.float64)
        except (TypeError, ValueError):
            return ValidationResult(False, "joint_targets could not be converted to a float array")

        if joint_targets.shape != (self.num_joints,):
            return ValidationResult(False, f"invalid joint count: expected ({self.num_joints},), got {joint_targets.shape}")
        if not np.all(np.isfinite(joint_targets)):
            return ValidationResult(False, "non-finite joint_targets (NaN/Inf)")

        try:
            gripper_target = float(gripper_target)
        except (TypeError, ValueError):
            return ValidationResult(False, "gripper_target could not be converted to float")
        if not np.isfinite(gripper_target):
            return ValidationResult(False, "non-finite gripper_target (NaN/Inf)")
        if not (0.0 <= gripper_target <= 1.0):
            return ValidationResult(False, f"gripper_target {gripper_target} outside [0, 1]")

        if self.max_joint_delta is not None and previous_joint_positions is not None:
            previous_joint_positions = np.asarray(previous_joint_positions, dtype=np.float64)
            if previous_joint_positions.shape == joint_targets.shape:
                delta = float(np.max(np.abs(joint_targets - previous_joint_positions)))
                if delta > self.max_joint_delta:
                    return ValidationResult(False, f"extreme joint delta {delta:.4f} rad > max {self.max_joint_delta} rad")

        return ValidationResult(True, None)

    def build_action(
        self, joint_targets, gripper_target, previous_joint_positions: np.ndarray = None
    ) -> RobotAction:
        """Validate and construct a ``RobotAction``, or raise ``CommandRejected``."""
        result = self.validate(joint_targets, gripper_target, previous_joint_positions)
        if not result.valid:
            raise CommandRejected(result.reason)
        return RobotAction(
            joint_targets=np.asarray(joint_targets, dtype=np.float64), gripper_target=float(gripper_target)
        )
