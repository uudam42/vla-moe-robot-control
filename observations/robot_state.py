"""Structured robot proprioception, extracted by the simulation layer.

:class:`RobotState` is the simulator-agnostic shape of "what the robot
currently knows about itself" -- joint configuration, end-effector pose, and
gripper opening. ``SimulationEnvironment.get_robot_state()`` produces one of
these from MuJoCo, and ``RobotState.as_vector()`` flattens it into the
``state`` field of an :class:`~observations.observation.Observation`.

Nothing in this file may import or reference MuJoCo types.
"""

from dataclasses import dataclass

import numpy as np

STATE_DIM = 23

# Index layout of RobotState.as_vector() / Observation.state, so callers
# that need to slice the flat vector (e.g. the dataset recorder pulling
# diagnostic eef pose out of a recorded Observation.state) don't have to
# hardcode magic offsets.
JOINT_POSITIONS_SLICE = slice(0, 7)
JOINT_VELOCITIES_SLICE = slice(7, 14)
EEF_POSITION_SLICE = slice(14, 17)
EEF_ORIENTATION_SLICE = slice(17, 21)
GRIPPER_POSITION_SLICE = slice(21, 23)


@dataclass
class RobotState:
    """A single timestep of robot proprioception.

    Attributes:
        joint_positions: Arm joint angles, shape ``(7,)``, radians.
        joint_velocities: Arm joint velocities, shape ``(7,)``, radians/s.
        end_effector_position: Cartesian TCP position, shape ``(3,)``, meters.
        end_effector_orientation: TCP orientation as a quaternion ``(w, x,
            y, z)``, shape ``(4,)``.
        gripper_position: Raw finger joint positions, shape ``(2,)``, meters
            (0 = closed, ~0.04 = fully open per finger).
    """

    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    end_effector_position: np.ndarray
    end_effector_orientation: np.ndarray
    gripper_position: np.ndarray

    def __post_init__(self) -> None:
        _require_shape(self.joint_positions, (7,), "joint_positions")
        _require_shape(self.joint_velocities, (7,), "joint_velocities")
        _require_shape(self.end_effector_position, (3,), "end_effector_position")
        _require_shape(self.end_effector_orientation, (4,), "end_effector_orientation")
        _require_shape(self.gripper_position, (2,), "gripper_position")

    def as_vector(self) -> np.ndarray:
        """Flatten into the single vector used as ``Observation.state``.

        Layout: ``[joint_positions(7), joint_velocities(7),
        end_effector_position(3), end_effector_orientation(4),
        gripper_position(2)]`` -- 23 elements total.
        """
        return np.concatenate(
            [
                self.joint_positions,
                self.joint_velocities,
                self.end_effector_position,
                self.end_effector_orientation,
                self.gripper_position,
            ]
        ).astype(np.float64)


def _require_shape(array: np.ndarray, shape: tuple, name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array)}")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
