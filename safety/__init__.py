"""Step 10: runtime safety supervision for the learned VLA policy's
output, between ``policy.predict()`` and ``RobotBackend.execute_action()``.

**This is runtime safety supervision for simulation and research
deployment. It is NOT functional safety certification, SIL-rated safety,
ISO 10218 certification, hardware emergency-stop certification, or real
robot collision certification** -- see ``SafetySupervisor``'s docstring.

Composes, rather than duplicates, Step 9's
``ros_integration.command_validator.CommandValidator`` and
``ros_integration.watchdog.CommandWatchdog`` -- both remain the single
source of truth for "is this action well-formed" / "when is a command
stale", reused here and by the ROS2 bridge node alike.
"""

from safety.supervisor import SafetyDecision, SafetyEvent, SafetyReason, SafetySupervisor

__all__ = ["SafetyDecision", "SafetyEvent", "SafetyReason", "SafetySupervisor"]
