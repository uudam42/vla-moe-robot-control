"""Simulator-agnostic observation data structure.

This module defines :class:`Observation`, the boundary object between
whatever produces sensor data (MuJoCo today; a real robot, a webcam, or a
recorded dataset later) and any downstream consumer (a human inspecting
images now; a VLA model later). Nothing in this file may import or
reference MuJoCo types.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    """A single timestep of environment observation.

    Attributes:
        rgb: RGB image array of shape ``(H, W, 3)``, dtype ``uint8``.
            Raw and unprocessed -- no resizing or normalization is applied
            here. That is the responsibility of a future ML preprocessing
            stage, not this data structure.
        state: Flat numeric vector describing the environment's physical
            state (for Milestone 1: concatenated qpos/qvel of the scene).
            Later milestones may replace or extend this with robot joint
            positions/velocities, end-effector pose, and gripper state
            without changing this field's type or the surrounding API.
        timestamp: Simulation/wall-clock time in seconds at which this
            observation was captured.
    """

    rgb: np.ndarray
    state: np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.rgb, np.ndarray):
            raise TypeError(f"rgb must be a numpy array, got {type(self.rgb)}")
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError(
                f"rgb must have shape (H, W, 3), got {self.rgb.shape}"
            )
        if not isinstance(self.state, np.ndarray):
            raise TypeError(f"state must be a numpy array, got {type(self.state)}")
