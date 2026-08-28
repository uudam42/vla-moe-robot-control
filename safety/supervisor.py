"""``SafetySupervisor``: the single place a predicted ``RobotAction`` is
checked and, if necessary, corrected before it reaches a ``RobotBackend``.

```
Observation -> policy.predict() -> RobotAction -> SafetySupervisor.process()
    -> safe RobotAction -> RobotBackend.execute_action()
```

**Important disclaimer**: this is runtime safety supervision for
simulation and research deployment. It is **NOT** functional safety
certification, SIL-rated safety, ISO 10218 industrial-robot certification,
hardware emergency-stop certification, or real-robot collision
certification. It catches malformed/out-of-bounds/stale model output in a
simulated or research setting; it has not been validated for, and must
never be represented as, physical-hardware safety.

Composes (does not duplicate) Step 9's
``ros_integration.command_validator.CommandValidator`` (shape/finite/
gripper-range/max-joint-delta checks) and
``ros_integration.watchdog.CommandWatchdog`` (hold-last-safe-action
pattern) -- both stay the single source of truth for those checks,
reused identically here and by the ROS2 bridge node
(``ros_integration.bridge_node_core.MuJoCoBridgeNodeCore`` also composes
``CommandValidator``/``CommandWatchdog`` directly, following the exact
same reason-coded intervention pattern this module formalizes).
"""

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from control.action import RobotAction
from ros_integration.command_validator import CommandValidator

DEFAULT_NUM_JOINTS = 7
DEFAULT_MAX_JOINT_DELTA = 0.5  # radians per control step
DEFAULT_MAX_CONSECUTIVE_INTERVENTIONS = 5


class SafetyDecision(str, Enum):
    ACCEPT = "ACCEPT"  # action passed through unmodified
    CLAMP = "CLAMP"  # action was numerically corrected (e.g. joint/gripper clipped)
    HOLD = "HOLD"  # action discarded; last known-safe action reused instead
    REJECT = "REJECT"  # action discarded, unrepairable (e.g. NaN/wrong shape); last safe action reused
    STOP_EPISODE = "STOP_EPISODE"  # too many consecutive interventions; caller should end the episode


class SafetyReason(str, Enum):
    NONE = "NONE"
    NONFINITE_ACTION = "NONFINITE_ACTION"
    INVALID_SHAPE = "INVALID_SHAPE"
    INVALID_GRIPPER = "INVALID_GRIPPER"
    MAX_JOINT_DELTA = "MAX_JOINT_DELTA"
    JOINT_LIMIT = "JOINT_LIMIT"
    STALE_OBSERVATION = "STALE_OBSERVATION"
    COMMAND_TIMEOUT = "COMMAND_TIMEOUT"
    BACKEND_NOT_READY = "BACKEND_NOT_READY"
    REPEATED_INTERVENTION = "REPEATED_INTERVENTION"


def _action_to_dict(action: RobotAction) -> dict:
    if action is None:
        return None
    return {"joint_targets": np.asarray(action.joint_targets).tolist(), "gripper_target": float(action.gripper_target)}


@dataclass
class SafetyEvent:
    """One supervisor decision. Referenced by (episode_id, step_id) from
    telemetry rather than embedding large data -- ``original_action``/
    ``safe_action`` are small (8 floats each), never RGB frames."""

    step_id: int
    timestamp: float
    decision: SafetyDecision
    reason: SafetyReason
    original_action: dict
    safe_action: dict
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "decision": self.decision.value,
            "reason": self.reason.value,
            "original_action": self.original_action,
            "safe_action": self.safe_action,
            "metadata": self.metadata,
        }


class SafetySupervisor:
    """Simulator/hardware-agnostic action safety supervisor.

    Args:
        num_joints: Expected joint count.
        max_joint_delta: Maximum allowed per-joint change (radians) from
            the previous EXECUTED action, when a previous action is known.
        joint_range: Optional ``(num_joints, 2)`` array of authoritative
            ``[min, max]`` joint bounds (e.g. ``MuJoCoBackend.get_joint_range()``)
            -- NOT duplicated/hardcoded here; ``None`` disables this check
            (e.g. for a backend that can't supply it).
        stale_observation_sec: If given, ``process(..., observation_timestamp=...)``
            HOLDs (reuses the last safe action) rather than acting on an
            observation older than this many seconds.
        command_timeout_sec: If given, ``process(..., inference_latency_sec=...)``
            HOLDs if the policy took longer than this to produce the action
            -- the "policy timeout" case (README "Safety responsibilities").
        max_consecutive_interventions: Number of consecutive non-ACCEPT
            decisions before escalating to ``STOP_EPISODE`` (the caller,
            not this class, is responsible for actually ending the episode).
    """

    def __init__(
        self,
        num_joints: int = DEFAULT_NUM_JOINTS,
        max_joint_delta: float = DEFAULT_MAX_JOINT_DELTA,
        joint_range: np.ndarray = None,
        stale_observation_sec: float = None,
        command_timeout_sec: float = None,
        max_consecutive_interventions: int = DEFAULT_MAX_CONSECUTIVE_INTERVENTIONS,
    ) -> None:
        self.validator = CommandValidator(num_joints=num_joints, max_joint_delta=max_joint_delta)
        self.joint_range = np.asarray(joint_range, dtype=np.float64) if joint_range is not None else None
        self.stale_observation_sec = stale_observation_sec
        self.command_timeout_sec = command_timeout_sec
        self.max_consecutive_interventions = max_consecutive_interventions

        self._last_safe_action: RobotAction = None
        self._last_executed_joint_positions: np.ndarray = None
        self._step_id = 0
        self._consecutive_interventions = 0
        self.events: list = []

    def reset(self) -> None:
        """Call at episode start -- clears history without resetting the
        cumulative ``events`` log (callers that want a per-episode log
        should read/clear ``events`` themselves, e.g. via the recorder)."""
        self._last_safe_action = None
        self._last_executed_joint_positions = None
        self._step_id = 0
        self._consecutive_interventions = 0

    def process(
        self,
        action: RobotAction,
        observation_timestamp: float = None,
        inference_latency_sec: float = None,
        backend_ready: bool = True,
        now: float = None,
    ) -> tuple:
        """Returns ``(safe_action, SafetyEvent)``. ``safe_action`` may be
        ``None`` only if no safe action has ever been established yet
        (first-ever tick AND the very first action is itself invalid) --
        callers must handle that (e.g. skip execution this tick)."""
        now = now if now is not None else time.time()
        step_id = self._step_id
        self._step_id += 1

        if not backend_ready:
            return self._intervene(step_id, now, SafetyDecision.HOLD, SafetyReason.BACKEND_NOT_READY, action)

        if observation_timestamp is not None and self.stale_observation_sec is not None:
            age = now - observation_timestamp
            if age > self.stale_observation_sec:
                return self._intervene(
                    step_id, now, SafetyDecision.HOLD, SafetyReason.STALE_OBSERVATION, action,
                    metadata={"observation_age_sec": age},
                )

        if inference_latency_sec is not None and self.command_timeout_sec is not None:
            if inference_latency_sec > self.command_timeout_sec:
                return self._intervene(
                    step_id, now, SafetyDecision.HOLD, SafetyReason.COMMAND_TIMEOUT, action,
                    metadata={"inference_latency_sec": inference_latency_sec},
                )

        joint_targets = getattr(action, "joint_targets", None) if action is not None else None
        gripper_target = getattr(action, "gripper_target", None) if action is not None else None
        result = self.validator.validate(joint_targets, gripper_target, self._last_executed_joint_positions)

        if not result.valid:
            reason = self._classify_validation_failure(result.reason)
            if reason in (SafetyReason.NONFINITE_ACTION, SafetyReason.INVALID_SHAPE):
                # Unrepairable -- can't safely clamp garbage/wrong-shaped data.
                return self._intervene(
                    step_id, now, SafetyDecision.REJECT, reason, action, metadata={"validator_reason": result.reason}
                )
            if reason == SafetyReason.INVALID_GRIPPER:
                clamped = RobotAction(
                    joint_targets=np.asarray(joint_targets, dtype=np.float64),
                    gripper_target=float(np.clip(gripper_target, 0.0, 1.0)),
                )
                return self._intervene(step_id, now, SafetyDecision.CLAMP, reason, action, safe_override=clamped)
            if reason == SafetyReason.MAX_JOINT_DELTA:
                delta = np.clip(
                    np.asarray(joint_targets, dtype=np.float64) - self._last_executed_joint_positions,
                    -self.validator.max_joint_delta, self.validator.max_joint_delta,
                )
                clamped = RobotAction(
                    joint_targets=self._last_executed_joint_positions + delta,
                    gripper_target=float(np.clip(gripper_target, 0.0, 1.0)),
                )
                return self._intervene(step_id, now, SafetyDecision.CLAMP, reason, action, safe_override=clamped)

        # Valid per CommandValidator -- final check: authoritative joint bounds.
        joint_targets_final = np.asarray(joint_targets, dtype=np.float64)
        gripper_final = float(np.clip(gripper_target, 0.0, 1.0))
        if self.joint_range is not None:
            clamped_joints = np.clip(joint_targets_final, self.joint_range[:, 0], self.joint_range[:, 1])
            if not np.allclose(clamped_joints, joint_targets_final):
                safe = RobotAction(joint_targets=clamped_joints, gripper_target=gripper_final)
                return self._intervene(step_id, now, SafetyDecision.CLAMP, SafetyReason.JOINT_LIMIT, action, safe_override=safe)

        safe_action = RobotAction(joint_targets=joint_targets_final, gripper_target=gripper_final)
        self._consecutive_interventions = 0
        event = SafetyEvent(
            step_id=step_id, timestamp=now, decision=SafetyDecision.ACCEPT, reason=SafetyReason.NONE,
            original_action=_action_to_dict(action), safe_action=_action_to_dict(safe_action),
        )
        self.events.append(event)
        self._last_safe_action = safe_action
        self._last_executed_joint_positions = safe_action.joint_targets
        return safe_action, event

    def _classify_validation_failure(self, validator_reason: str) -> SafetyReason:
        if "non-finite" in validator_reason:
            return SafetyReason.NONFINITE_ACTION
        if "joint count" in validator_reason or "could not be converted" in validator_reason:
            return SafetyReason.INVALID_SHAPE
        if "gripper_target" in validator_reason:
            return SafetyReason.INVALID_GRIPPER
        if "joint delta" in validator_reason:
            return SafetyReason.MAX_JOINT_DELTA
        return SafetyReason.INVALID_SHAPE

    def _intervene(
        self, step_id: int, now: float, decision: SafetyDecision, reason: SafetyReason, original_action,
        safe_override: RobotAction = None, metadata: dict = None,
    ) -> tuple:
        self._consecutive_interventions += 1
        if self._consecutive_interventions > self.max_consecutive_interventions:
            decision = SafetyDecision.STOP_EPISODE

        safe_action = safe_override if safe_override is not None else self._last_safe_action
        event = SafetyEvent(
            step_id=step_id, timestamp=now, decision=decision, reason=reason,
            original_action=_action_to_dict(original_action) if original_action is not None else None,
            safe_action=_action_to_dict(safe_action), metadata=metadata or {},
        )
        self.events.append(event)
        if safe_action is not None:
            self._last_safe_action = safe_action
            self._last_executed_joint_positions = safe_action.joint_targets
        return safe_action, event
