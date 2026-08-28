"""``EpisodeTelemetryRecorder``: structured per-tick telemetry + episode
archive for a runtime rollout (README "Telemetry System" / "Episode
Recorder" / "Metadata" / "Telemetry Format" / "Runtime Metrics").

```
outputs/episodes/episode_<YYYYMMDD_HHMMSS>/
├── metadata.json      episode-level config/outcome (README "Metadata")
├── telemetry.jsonl    one independently-parseable JSON object per control tick
├── metrics.json       aggregated episode metrics (README "Runtime Metrics")
├── frames/000000.png  optional, if recording frames
└── video.mp4          optional, built from frames/ at finalize() if imageio is available
```

Never stores privileged diagnostic state as if it were policy input --
``cube_height_delta`` etc. are logged for EVALUATION ONLY (README
"Preserve no-privileged-input rule"); nothing here is read back by
``policy.predict()``.
"""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation.metrics import latency_stats
from evaluation.temporal_diagnostics import gripper_switch_count

DEFAULT_OUTPUT_ROOT = "outputs/episodes"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _action_to_dict(action) -> dict:
    if action is None:
        return None
    return {"joint_targets": np.asarray(action.joint_targets).tolist(), "gripper_target": float(action.gripper_target)}


@dataclass
class EpisodeConfig:
    policy_type: str
    checkpoint: str
    instruction: str
    seed: int = None
    cube_xy_randomization: float = None
    backend: str = "mujoco"
    device: str = None
    max_steps: int = None
    control_substeps: int = None


class EpisodeTelemetryRecorder:
    """One instance per recorded episode.

    Usage:
        recorder = EpisodeTelemetryRecorder(config, output_root="outputs/episodes")
        recorder.start()
        for tick in range(max_steps):
            ...
            recorder.record_step(...)
        recorder.finalize(success=..., failure_reason=..., cube_lift_delta=...)
    """

    def __init__(self, config: EpisodeConfig, output_root: str = DEFAULT_OUTPUT_ROOT) -> None:
        self.config = config
        self.output_root = Path(output_root)
        self.episode_dir: Path = None
        self._telemetry_file = None
        self._num_steps = 0
        self._gripper_probabilities = []
        self._inference_latencies_ms = []
        self._safety_intervention_count = 0
        self._stale_observation_count = 0
        self._watchdog_trigger_count = 0
        self._start_wall_time = None
        self._frames_dir: Path = None
        self._save_frames = False

    def start(self, save_frames: bool = False) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = self.output_root / f"episode_{timestamp}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self._telemetry_file = open(self.episode_dir / "telemetry.jsonl", "w", encoding="utf-8")
        self._start_wall_time = datetime.now(timezone.utc)
        self._save_frames = save_frames
        if save_frames:
            self._frames_dir = self.episode_dir / "frames"
            self._frames_dir.mkdir(parents=True, exist_ok=True)
        return self.episode_dir

    def save_frame(self, step_id: int, rgb: np.ndarray) -> None:
        if not self._save_frames:
            return
        from PIL import Image

        Image.fromarray(rgb, mode="RGB").save(self._frames_dir / f"{step_id:06d}.png")

    def record_step(
        self,
        step_id: int,
        wall_clock_timestamp: float,
        observation_timestamp: float,
        prediction_start: float,
        prediction_end: float,
        original_action,
        executed_action,
        safety_decision: str,
        safety_reason: str,
        backend_execute_latency_sec: float = None,
        simulation_timestamp: float = None,
        cube_height_delta: float = None,
        episode_state: str = "running",
        metadata: dict = None,
    ) -> None:
        inference_latency_sec = prediction_end - prediction_start
        record = {
            "step_id": step_id,
            "wall_clock_timestamp": wall_clock_timestamp,
            "simulation_timestamp": simulation_timestamp,
            "instruction": self.config.instruction,
            "policy_type": self.config.policy_type,
            "observation_timestamp": observation_timestamp,
            "prediction_start": prediction_start,
            "prediction_end": prediction_end,
            "inference_latency_sec": inference_latency_sec,
            "original_action": _action_to_dict(original_action),
            "executed_action": _action_to_dict(executed_action),
            "gripper_command": float(executed_action.gripper_target) if executed_action is not None else None,
            "safety_decision": safety_decision,
            "safety_reason": safety_reason,
            "backend_execute_latency_sec": backend_execute_latency_sec,
            "cube_height_delta": cube_height_delta,  # diagnostic-only, never fed to the policy
            "episode_state": episode_state,
            "metadata": metadata or {},
        }
        self._telemetry_file.write(json.dumps(record) + "\n")
        self._telemetry_file.flush()

        self._num_steps += 1
        if executed_action is not None:
            self._gripper_probabilities.append(float(executed_action.gripper_target))
        self._inference_latencies_ms.append(inference_latency_sec * 1000.0)
        if safety_decision != "ACCEPT":
            self._safety_intervention_count += 1
        if safety_reason == "STALE_OBSERVATION":
            self._stale_observation_count += 1
        if safety_reason == "COMMAND_TIMEOUT":
            self._watchdog_trigger_count += 1

    def finalize(
        self,
        success: bool,
        failure_reason: str = None,
        cube_lift_delta: float = None,
        max_cube_lift: float = None,
        extra_metadata: dict = None,
        build_video: bool = True,
    ) -> dict:
        """Writes metadata.json + metrics.json (and video.mp4 if frames
        were recorded and imageio is available). Returns the metrics dict."""
        self._telemetry_file.close()
        end_wall_time = datetime.now(timezone.utc)

        metadata = {
            **asdict(self.config),
            "start_time": self._start_wall_time.isoformat(),
            "end_time": end_wall_time.isoformat(),
            "git_commit": _git_commit(),
            "success": bool(success),
            "failure_reason": failure_reason,
            **(extra_metadata or {}),
        }
        with open(self.episode_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        latency = latency_stats(self._inference_latencies_ms)
        metrics = {
            "success": bool(success),
            "episode_steps": self._num_steps,
            "episode_duration_sec": (end_wall_time - self._start_wall_time).total_seconds(),
            "max_cube_lift": max_cube_lift,
            "final_cube_lift": cube_lift_delta,
            "gripper_switch_count": gripper_switch_count(self._gripper_probabilities) if self._gripper_probabilities else None,
            "inference_latency_ms": latency,
            "safety_intervention_count": self._safety_intervention_count,
            "stale_observation_count": self._stale_observation_count,
            "watchdog_trigger_count": self._watchdog_trigger_count,
        }
        with open(self.episode_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if build_video and self._save_frames:
            self._build_video()

        return metrics

    def _build_video(self) -> None:
        try:
            import imageio.v2 as imageio
        except ImportError:
            return  # README "Video": no proprietary tooling required, but imageio is optional
        frame_paths = sorted(self._frames_dir.glob("*.png"))
        if not frame_paths:
            return
        frames = [imageio.imread(p) for p in frame_paths]
        imageio.mimwrite(self.episode_dir / "video.mp4", frames, fps=10, macro_block_size=None)
