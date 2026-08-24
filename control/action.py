"""Simulator-agnostic robot action abstraction.

:class:`RobotAction` is the output type of any controller -- the scripted
expert today, a learned VLA later -- and the input type of
``SimulationEnvironment.step()``. Both a scripted controller and a future
VLA model must produce this same type, so that swapping one for the other
requires no change to the simulation/control boundary.

Nothing in this file may import or reference MuJoCo types.
"""

from dataclasses import dataclass

import numpy as np

NUM_ARM_JOINTS = 7
ACTION_DIM = NUM_ARM_JOINTS + 1  # 7 joint targets + 1 gripper target


@dataclass
class RobotAction:
    """A single commanded action.

    Attributes:
        joint_targets: Desired arm joint positions, shape ``(7,)``, radians.
            Joint-space (not Cartesian) because direct joint-position
            control is simpler and more deterministic than full Cartesian
            control.
        gripper_target: Normalized gripper command in ``[0, 1]``, where
            ``0.0`` means fully closed and ``1.0`` means fully open. The
            simulation layer maps this to actuator-specific units --
            callers never see MuJoCo actuator ranges.
    """

    joint_targets: np.ndarray
    gripper_target: float

    def __post_init__(self) -> None:
        if not isinstance(self.joint_targets, np.ndarray):
            raise TypeError(
                f"joint_targets must be a numpy array, got {type(self.joint_targets)}"
            )
        if self.joint_targets.shape != (NUM_ARM_JOINTS,):
            raise ValueError(
                f"joint_targets must have shape ({NUM_ARM_JOINTS},), "
                f"got {self.joint_targets.shape}"
            )
        if not np.all(np.isfinite(self.joint_targets)):
            raise ValueError("joint_targets must be finite (no NaN/Inf)")

        if not isinstance(self.gripper_target, (int, float)):
            raise TypeError(
                f"gripper_target must be numeric, got {type(self.gripper_target)}"
            )
        if not np.isfinite(self.gripper_target):
            raise ValueError("gripper_target must be finite (no NaN/Inf)")
        if not (0.0 <= self.gripper_target <= 1.0):
            raise ValueError(
                f"gripper_target must be in [0, 1], got {self.gripper_target}"
            )

    def as_vector(self) -> np.ndarray:
        """Flatten into the single vector used as a dataset training label.

        Layout: ``[joint1_target, ..., joint7_target, gripper_target]`` --
        shape ``(8,)``.
        """
        return np.concatenate([self.joint_targets, [self.gripper_target]]).astype(np.float64)
