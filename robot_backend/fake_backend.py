"""``FakeRobotBackend``: a MuJoCo-free ``RobotBackend`` test double.

Proves policy/control code depends only on the ``RobotBackend`` interface,
never on MuJoCo -- this is the whole point of the abstraction (README
"Backend abstraction test") and the shape any future real-hardware backend
will take.
"""

import numpy as np

from control.action import RobotAction
from observations.observation import Observation
from observations.robot_state import STATE_DIM
from robot_backend.base import RobotBackend

DEFAULT_IMAGE_SIZE = 64


class FakeRobotBackend(RobotBackend):
    """Deterministic, in-memory backend: no rendering, no physics.

    Each ``get_observation()`` returns a fixed-seed-derived RGB frame and a
    23D state vector that increments once per ``execute_action()`` call, so
    tests can assert timestamps/state actually advance without needing
    MuJoCo at all.

    Args:
        image_size: Height/width of the synthetic RGB frame.
        seed: Seeds the fixed RGB frame's pixel values.
    """

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE, seed: int = 0) -> None:
        self._rgb = np.random.default_rng(seed).integers(
            0, 255, size=(image_size, image_size, 3), dtype=np.uint8
        )
        self._tick = 0
        self._closed = False
        self.executed_actions: list = []  # test introspection: every action passed to execute_action

    def get_observation(self) -> Observation:
        state = np.full(STATE_DIM, float(self._tick), dtype=np.float64)
        return Observation(rgb=self._rgb.copy(), state=state, timestamp=float(self._tick) * 0.02)

    def execute_action(self, action: RobotAction) -> None:
        if not isinstance(action, RobotAction):
            raise TypeError(f"execute_action requires a RobotAction, got {type(action)}")
        self.executed_actions.append(action)
        self._tick += 1

    def reset(self) -> None:
        self._tick = 0
        self.executed_actions = []

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed
