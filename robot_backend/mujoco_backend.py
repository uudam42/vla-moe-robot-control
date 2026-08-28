"""``MuJoCoBackend``: the only ``RobotBackend`` implementation this
milestone actually executes against. Wraps the existing, unmodified
``SimulationEnvironment`` -- no physics logic is duplicated here, this is
a thin adapter.

Callers (policy loop, ROS2 bridge node) only ever see
``get_observation()``/``execute_action()``/``reset()``/``close()``; they
never touch ``self.env`` or know MuJoCo exists. Evaluation/diagnostic code
that legitimately needs privileged simulator access (ground-truth cube
position, Jacobians -- see ``control/scripted_controller.py``,
``evaluation/metrics.py``) may still use the passthrough properties/methods
below, which are deliberately NOT part of the ``RobotBackend`` ABC.
"""

from pathlib import Path

import numpy as np

from control.action import RobotAction
from observations.observation import Observation
from robot_backend.base import RobotBackend
from simulation.environment import DEFAULT_CONTROL_SUBSTEPS, DEFAULT_SCENE_PATH, SimulationEnvironment


class MuJoCoBackend(RobotBackend):
    """Adapts ``SimulationEnvironment`` to the ``RobotBackend`` interface.

    Args:
        env: An existing ``SimulationEnvironment`` to wrap. If ``None``, a
            new one is constructed with the given kwargs (mirroring
            ``SimulationEnvironment``'s own defaults, in particular
            ``control_substeps=10`` -- README "Low-level physics
            frequency": the ROS2/backend layer must not silently change
            this).
    """

    def __init__(
        self,
        env: SimulationEnvironment = None,
        scene_path: Path = DEFAULT_SCENE_PATH,
        control_substeps: int = DEFAULT_CONTROL_SUBSTEPS,
    ) -> None:
        self.env = env if env is not None else SimulationEnvironment(
            scene_path=scene_path, control_substeps=control_substeps
        )
        self._owns_env = env is None

    def get_observation(self) -> Observation:
        return self.env.get_observation()

    def execute_action(self, action: RobotAction) -> None:
        self.env.step(action)

    def reset(self) -> None:
        self.env.reset()

    def close(self) -> None:
        if self._owns_env:
            self.env.close()

    # -- Privileged / evaluation-only passthrough (NOT part of RobotBackend) --
    # A future real-robot backend has no equivalent for any of these; only
    # evaluation/diagnostic code (never the policy loop) may call them.

    def get_object_position(self, body_name: str) -> np.ndarray:
        return self.env.get_object_position(body_name)

    def set_object_position(self, body_name: str, position: np.ndarray) -> None:
        self.env.set_object_position(body_name, position)

    def get_robot_state(self):
        return self.env.get_robot_state()

    def get_end_effector_jacobian(self) -> tuple:
        return self.env.get_end_effector_jacobian()

    def get_joint_range(self) -> np.ndarray:
        """Authoritative per-joint position bounds, shape ``(7, 2)``.
        NOT privileged in the same sense as the methods above (it's a
        static robot spec, not scene state) -- a ``SafetySupervisor`` may
        read this even though it isn't part of the core ``RobotBackend``
        ABC, since not every backend can supply it (e.g. ``FakeRobotBackend``)."""
        return self.env.get_joint_range()
