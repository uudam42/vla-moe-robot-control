"""Documented extension point for a future real-robot backend (README
"Future hardware backend stub").

**Nothing in this file talks to real hardware.** No Franka/UR5 driver, no
MoveIt planning, no real-time control loop is implemented here or
anywhere else in this milestone (README "Do NOT implement real robot
control yet" / "Avoid overclaiming"). This class exists solely to make
the INTENDED future contract explicit and to prove, by inheriting from
the same ``RobotBackend`` ABC as ``MuJoCoBackend``, that the policy/ROS2
layers built in Step 9 require no changes to target real hardware later
-- only a new backend implementation.

A real implementation would need, at minimum: a hardware driver
connection (e.g. ``franka_ros2`` / a UR RTDE client), real joint-limit and
velocity/torque safety enforcement (Step 9's ``ros_integration.command_validator``
is a simulation-appropriate starting point, not a certified safety
system), and a real success/termination signal (there is no
``get_object_position`` equivalent on real hardware -- see
``robot_backend.mujoco_backend.MuJoCoBackend``'s privileged passthrough
docstring). None of that exists yet.
"""

from control.action import RobotAction
from observations.observation import Observation
from robot_backend.base import RobotBackend


class FutureHardwareBackend(RobotBackend):
    """Unimplemented on purpose. Documents the extension point; raises if
    actually used, so it can never be silently mistaken for a working
    real-robot backend."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "FutureHardwareBackend is a documented interface stub, not a real implementation "
            "(README 'Future hardware backend stub'). Real robot control is out of scope through "
            "at least Step 9/10 -- see this module's docstring for what a real implementation needs."
        )

    def get_observation(self) -> Observation:  # pragma: no cover - documentation only
        raise NotImplementedError

    def execute_action(self, action: RobotAction) -> None:  # pragma: no cover - documentation only
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - documentation only
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - documentation only
        raise NotImplementedError
