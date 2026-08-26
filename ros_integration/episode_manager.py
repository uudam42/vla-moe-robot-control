"""Episode reset orchestration (README "Reset API" / "Temporal policy
reset" / "Reset behavior ordering").

The ``/reset_episode`` ROS2 service (``ros2_ws/``) is a thin wrapper
around ``EpisodeManager.reset()`` -- kept here so the ordering
(backend -> policy -> metrics) is unit-testable without ``rclpy``, and so
a future service/lifecycle-node implementation can't silently reorder it.
"""

from dataclasses import dataclass, field


@dataclass
class EpisodeMetrics:
    """Minimal per-episode counters reset alongside backend/policy state."""

    ticks: int = 0
    actions_executed: int = 0
    watchdog_timeouts_this_episode: int = 0


class EpisodeManager:
    """Owns one backend + one policy and coordinates episode resets.

    Reset ordering (README "Document ordering"): backend FIRST (so any
    policy-triggered inference immediately after reset sees the
    post-reset scene, not a stale one), then policy (clears history --
    critical for ``TemporalDenseVLAPolicy``, README "Temporal policy
    reset": must not leak history across resets), then metrics.
    """

    def __init__(self, backend, policy) -> None:
        self.backend = backend
        self.policy = policy
        self.metrics = EpisodeMetrics()

    def reset(self) -> None:
        self.backend.reset()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.metrics = EpisodeMetrics()

    def record_tick(self) -> None:
        self.metrics.ticks += 1

    def record_action_executed(self) -> None:
        self.metrics.actions_executed += 1
