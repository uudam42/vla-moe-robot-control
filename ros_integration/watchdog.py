"""Command watchdog (README "Watchdog"): if the policy node stops
publishing actions, the backend must not keep applying an arbitrarily
stale command forever. Simulated-only policy: hold the last known-safe
action once the timeout is exceeded, rather than continuing to execute
whatever was last received indefinitely or (worse) a freshly-stale one
re-published as if current.

Pure state machine, no ``rclpy``/timers -- a node just calls
``record_command()`` on every received action and ``get_action(now)`` on
every backend-execution tick, where ``now`` is that node's own monotonic
clock reading.
"""

from control.action import RobotAction

DEFAULT_COMMAND_TIMEOUT_SEC = 0.5


class CommandWatchdog:
    """Tracks command freshness and provides the hold-last-safe-action behavior.

    Args:
        timeout_sec: Maximum age (seconds) a received command may reach
            before it's considered stale.
    """

    def __init__(self, timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC) -> None:
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be > 0, got {timeout_sec}")
        self.timeout_sec = timeout_sec
        self._last_command_time: float = None
        self._last_safe_action: RobotAction = None
        self._currently_timed_out = False
        self.timeout_count = 0

    def record_command(self, action: RobotAction, now: float) -> None:
        """Call once per accepted (validated) command received."""
        self._last_command_time = now
        self._last_safe_action = action
        self._currently_timed_out = False

    def get_action(self, now: float) -> tuple:
        """Returns ``(action_to_execute_or_None, timed_out)``.

        * No command ever received: ``(None, False)`` -- caller should not
          execute anything yet (this is "missing observation"/not-started,
          not a timeout).
        * Fresh command: ``(last_command, False)``.
        * Stale command: ``(last_safe_action, True)`` -- HOLDS the last
          safe action rather than doing nothing or accepting a late one.
          ``timeout_count`` increments exactly once per stale transition
          (not once per tick spent stale).
        """
        if self._last_command_time is None:
            return None, False

        timed_out = (now - self._last_command_time) > self.timeout_sec
        if timed_out and not self._currently_timed_out:
            self.timeout_count += 1
            self._currently_timed_out = True
        return self._last_safe_action, timed_out

    def reset(self) -> None:
        """Call on episode reset -- clears command history without
        touching the cumulative ``timeout_count`` diagnostic."""
        self._last_command_time = None
        self._last_safe_action = None
        self._currently_timed_out = False
