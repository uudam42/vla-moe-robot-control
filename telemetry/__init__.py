"""Step 10: structured per-episode runtime telemetry, recording, and replay.

Distinct from ``dataset.recorder.EpisodeRecorder`` (Step 3, which records
EXPERT demonstrations as training data) -- this package records RUNTIME
rollouts of an already-trained policy for observability/diagnostics/replay,
under ``outputs/episodes/`` rather than ``data/demonstrations/``.
"""

from telemetry.recorder import EpisodeTelemetryRecorder

__all__ = ["EpisodeTelemetryRecorder"]
