"""Observation synchronization + staleness policy (README "Observation
synchronization" / "Stale data handling").

Camera and robot-state messages arrive on separate topics at independent
rates; this module implements the "simple explicit latest-message
synchronization" option README lists as an alternative to
``message_filters.ApproximateTimeSynchronizer`` -- chosen here because it
has zero ``rclpy``/``message_filters`` dependency and is fully testable in
this environment (README "Dependency isolation"). The real
``vla_policy_node`` (``ros2_ws/``) uses this class internally; swapping in
``ApproximateTimeSynchronizer`` later would only change how ``update_image``/
``update_state`` get called, not this policy.

Strategy: track only the LATEST image and LATEST state message. A
synchronized observation is available only when both exist AND their
timestamps are within ``max_sync_delta`` of each other -- prevents
blindly combining an old image with a fresh state (or vice versa).
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_SYNC_DELTA_SEC = 0.1
DEFAULT_MAX_OBSERVATION_AGE_SEC = 0.5


@dataclass
class SyncedObservation:
    rgb: np.ndarray
    state: np.ndarray
    image_timestamp: float
    state_timestamp: float
    sync_delta: float


class LatestMessageSynchronizer:
    """Explicit latest-image/latest-state synchronization with a max time delta.

    Args:
        max_sync_delta_sec: Maximum allowed ``|image_timestamp - state_timestamp|``
            for the pair to be considered synchronized.
    """

    def __init__(self, max_sync_delta_sec: float = DEFAULT_MAX_SYNC_DELTA_SEC) -> None:
        self.max_sync_delta_sec = max_sync_delta_sec
        self._latest_image: tuple = None  # (rgb, timestamp)
        self._latest_state: tuple = None  # (state, timestamp)

    def update_image(self, rgb: np.ndarray, timestamp: float) -> None:
        self._latest_image = (rgb, float(timestamp))

    def update_state(self, state: np.ndarray, timestamp: float) -> None:
        self._latest_state = (state, float(timestamp))

    def get_synced(self) -> SyncedObservation:
        """Returns a ``SyncedObservation`` if both streams are present and
        within ``max_sync_delta_sec`` of each other, else ``None``."""
        if self._latest_image is None or self._latest_state is None:
            return None
        rgb, image_timestamp = self._latest_image
        state, state_timestamp = self._latest_state
        delta = abs(image_timestamp - state_timestamp)
        if delta > self.max_sync_delta_sec:
            return None
        return SyncedObservation(
            rgb=rgb, state=state, image_timestamp=image_timestamp, state_timestamp=state_timestamp, sync_delta=delta
        )

    def reset(self) -> None:
        self._latest_image = None
        self._latest_state = None


class StalenessChecker:
    """README "Stale data handling": is a synchronized observation too old
    to act on, relative to the current (wall-clock) time?

    Args:
        max_age_sec: Maximum allowed age of the NEWER of the two source
            timestamps before the observation is considered stale.
    """

    def __init__(self, max_age_sec: float = DEFAULT_MAX_OBSERVATION_AGE_SEC) -> None:
        self.max_age_sec = max_age_sec

    def is_stale(self, synced: SyncedObservation, now: float) -> bool:
        newest_timestamp = max(synced.image_timestamp, synced.state_timestamp)
        return (now - newest_timestamp) > self.max_age_sec
