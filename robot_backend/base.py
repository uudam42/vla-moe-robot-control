"""``RobotBackend``: the simulator/hardware-agnostic execution interface.

The learned policy's contract has always been
``Observation -> RobotAction`` (``models/policy.py`` et al.). Through Step
8, whatever DROVE that loop (``evaluation/closed_loop.py``,
``dataset/generate_dataset.py``, ``dagger/collector.py``) talked to
``simulation.environment.SimulationEnvironment`` directly. ``RobotBackend``
factors that "how do I get an Observation / how do I execute a
RobotAction" responsibility out into a small interface so the same policy
loop can run against MuJoCo today (``MuJoCoBackend``), a test double
(``FakeRobotBackend``), or a real robot later (see
``ros_integration/future_hardware_backend.py`` for the documented,
unimplemented extension point).

Only ``execute_action`` and ``get_observation``/``reset``/``close`` are on
this interface -- privileged, evaluation-only operations (ground-truth
object position, Jacobians, MuJoCo IDs) are deliberately NOT part of it.
Those remain direct ``SimulationEnvironment`` calls made by
evaluation/diagnostic code, never by policy-facing code.
"""

from abc import ABC, abstractmethod

from control.action import RobotAction
from observations.observation import Observation


class RobotBackend(ABC):
    """Minimal simulator/hardware-agnostic robot execution interface."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Current sensor observation: RGB + 23D proprioceptive state + timestamp."""
        raise NotImplementedError

    @abstractmethod
    def execute_action(self, action: RobotAction) -> None:
        """Command the robot to move toward ``action``. Blocking/synchronous."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Return the robot (and, in simulation, the scene) to its initial state."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release any owned resources (renderer, hardware connection, ...)."""
        raise NotImplementedError

    def __enter__(self) -> "RobotBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
