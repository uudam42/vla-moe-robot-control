"""Step 9: simulator/hardware-agnostic robot backend abstraction.

Nothing upstream of ``RobotBackend`` (the learned policy, the closed-loop
control loop, the ROS2 nodes in ``ros2_ws/``) may depend on
``simulation.environment.SimulationEnvironment`` directly -- only
``MuJoCoBackend`` (``robot_backend/mujoco_backend.py``) is allowed to know
MuJoCo exists. This is what makes the same policy code usable unchanged
against a future real-robot backend.
"""

from robot_backend.base import RobotBackend

__all__ = ["RobotBackend"]
