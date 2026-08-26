"""rclpy-independent control-loop logic for the VLA policy node (README
"VLA Policy Node" / "Observation synchronization" / "Stale data handling"
/ "Instruction handling" / "Control loop").

The real ``vla_policy_node`` (``ros2_ws/``) is a thin ``rclpy.Node``
wrapper: subscriber callbacks call ``on_image``/``on_state``/
``on_instruction`` here, and a timer at the configured control frequency
calls ``tick(now)`` and publishes whatever it returns. Every decision --
whether an observation is synchronized, whether it's stale, which
instruction to use, when to actually call ``policy.predict()`` -- lives
here so it's fully unit-testable without ``rclpy`` installed (README
"Dependency isolation").
"""

from observations.observation import Observation
from ros_integration.instruction_cache import InstructionCache
from ros_integration.sync import LatestMessageSynchronizer, StalenessChecker


class VLAPolicyNodeCore:
    """Owns exactly one policy instance; never imports MuJoCo (README "The
    node must not import MuJoCo" -- enforced by
    ``tests/test_ros_policy_node_core.py``, since this module has no
    MuJoCo import at all)."""

    def __init__(
        self,
        policy,
        synchronizer: LatestMessageSynchronizer = None,
        staleness_checker: StalenessChecker = None,
        instruction_cache: InstructionCache = None,
    ) -> None:
        self.policy = policy
        self.sync = synchronizer or LatestMessageSynchronizer()
        self.staleness = staleness_checker or StalenessChecker()
        self.instructions = instruction_cache or InstructionCache()
        self.stats = {"ticks": 0, "skipped_unsynced": 0, "skipped_stale": 0, "inferences": 0}

    def on_image(self, rgb, timestamp: float) -> None:
        self.sync.update_image(rgb, timestamp)

    def on_state(self, state, timestamp: float) -> None:
        self.sync.update_state(state, timestamp)

    def on_instruction(self, instruction: str) -> bool:
        return self.instructions.update(instruction)

    def reset(self) -> None:
        """README "Reset ordering" for the policy node's own state:
        clears the synchronizer (an in-flight stale image/state pair must
        not carry into the next episode) and the policy's own history
        (critical for ``TemporalDenseVLAPolicy`` -- README "Temporal
        policy reset"). Does NOT reset the instruction cache -- the task
        instruction is a runtime/launch-level setting, not per-episode
        state."""
        self.sync.reset()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def tick(self, now: float):
        """Called once per control-loop timer tick. Returns a
        ``RobotAction`` to publish, or ``None`` if inference was skipped
        this tick (no synchronized observation yet, or it's stale --
        README "skip policy inference + log warning")."""
        self.stats["ticks"] += 1
        synced = self.sync.get_synced()
        if synced is None:
            self.stats["skipped_unsynced"] += 1
            return None
        if self.staleness.is_stale(synced, now):
            self.stats["skipped_stale"] += 1
            return None

        observation = Observation(rgb=synced.rgb, state=synced.state, timestamp=synced.state_timestamp)
        action = self.policy.predict(observation=observation, instruction=self.instructions.get())
        self.stats["inferences"] += 1
        return action
