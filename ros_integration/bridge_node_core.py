"""rclpy-independent control-loop logic for the MuJoCo bridge node
(README "MuJoCo Bridge Node" / "Action validation" / "Minimal safety
boundary" / "Watchdog").

The real ``mujoco_bridge_node`` (``ros2_ws/``) is a thin ``rclpy.Node``
wrapper around this class -- it is the ONLY ROS2-layer code allowed to
touch a ``RobotBackend``/``SimulationEnvironment`` (README "the bridge is
the only ROS2 node allowed to know MuJoCo internals"). Subscriber
callbacks for the action topic call ``on_action_received``; a timer at the
backend's execution cadence calls ``tick``.
"""

from dataclasses import dataclass

from ros_integration.command_validator import CommandValidator, ValidationResult
from ros_integration.watchdog import CommandWatchdog


@dataclass
class BridgeStats:
    commands_received: int = 0
    commands_rejected: int = 0
    ticks_executed: int = 0
    watchdog_holds: int = 0


class MuJoCoBridgeNodeCore:
    """Owns exactly one ``RobotBackend``. Validates every incoming action
    before it reaches ``backend.execute_action`` and enforces the command
    watchdog on every execution tick."""

    def __init__(self, backend, validator: CommandValidator = None, watchdog: CommandWatchdog = None) -> None:
        self.backend = backend
        self.validator = validator or CommandValidator()
        self.watchdog = watchdog or CommandWatchdog()
        self.stats = BridgeStats()
        self._last_joint_positions = None

    def on_action_received(self, joint_targets, gripper_target, now: float) -> ValidationResult:
        """Validate and, if valid, record the command with the watchdog.
        Returns the ``ValidationResult`` so the caller (the real node) can
        log a structured diagnostic on rejection (README "reject command
        and log a structured diagnostic")."""
        self.stats.commands_received += 1
        result = self.validator.validate(
            joint_targets, gripper_target, previous_joint_positions=self._last_joint_positions
        )
        if not result.valid:
            self.stats.commands_rejected += 1
            return result

        action = self.validator.build_action(joint_targets, gripper_target, previous_joint_positions=None)
        self.watchdog.record_command(action, now)
        self._last_joint_positions = action.joint_targets
        return result

    def tick(self, now: float):
        """Called once per backend-execution tick. Executes the current
        watchdog-selected action (fresh or held-safe) against the
        backend, or does nothing if no command has ever been received.
        Returns ``(action_or_None, timed_out)``."""
        action, timed_out = self.watchdog.get_action(now)
        if timed_out:
            self.stats.watchdog_holds += 1
        if action is None:
            return None, timed_out
        self.backend.execute_action(action)
        self.stats.ticks_executed += 1
        return action, timed_out

    def reset(self) -> None:
        self.backend.reset()
        self.watchdog.reset()
        self._last_joint_positions = None
