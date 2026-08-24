"""Records one episode's Observation/RobotAction stream to disk.

``EpisodeRecorder`` only observes the existing control loop (see
``dataset/generate_dataset.py`` for where it's wired in) -- it never
computes IK, never decides controller transitions, and never touches
MuJoCo. It writes to a ``.tmp`` directory and only becomes a real,
loader-visible ``episode_NNNNNN/`` directory on :meth:`finalize`, so a
crash mid-episode never leaves a half-written directory the loader could
mistake for valid data (see ``dataset/episode.py`` for the on-disk format,
and README "Dataset schema").
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from control.action import RobotAction
from dataset import DATASET_VERSION
from dataset.episode import METADATA_FILENAME, RGB_DIRNAME, TRAJECTORY_FILENAME, frame_path
from observations.observation import Observation
from observations.robot_state import EEF_ORIENTATION_SLICE, EEF_POSITION_SLICE


class EpisodeRecorder:
    """Accumulates one episode's timesteps and writes it out transactionally.

    Args:
        staging_root: Directory .tmp episode directories are created under
            while recording (e.g. the dataset output root itself, or a
            dedicated scratch subdirectory).
        episode_id: Zero-padded into the episode directory name.
    """

    def __init__(self, staging_root: Path, episode_id: int) -> None:
        self.episode_id = episode_id
        self._tmp_dir = Path(staging_root) / f"episode_{episode_id:06d}.tmp"
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir)
        (self._tmp_dir / RGB_DIRNAME).mkdir(parents=True)

        self._states = []
        self._joint_targets = []
        self._gripper_targets = []
        self._timestamps = []
        self._controller_stages = []
        self._eef_positions = []
        self._eef_orientations = []
        self._step = 0
        self._finalized = False

    @property
    def num_steps(self) -> int:
        return self._step

    def record(self, observation: Observation, action: RobotAction, controller_stage: str) -> None:
        """Record one timestep. Call BEFORE ``env.step(action)`` -- see module docstring.

        ``observation`` must be the observation the expert used to compute
        ``action`` (i.e. state_t, not state_t+1), which is exactly what
        ``dataset/generate_dataset.py``'s recording order guarantees.
        """
        Image.fromarray(observation.rgb, mode="RGB").save(frame_path(self._tmp_dir, self._step))

        state = observation.state
        self._states.append(state.copy())
        self._joint_targets.append(action.joint_targets.copy())
        self._gripper_targets.append(float(action.gripper_target))
        self._timestamps.append(float(observation.timestamp))
        self._controller_stages.append(str(controller_stage))
        # Diagnostics only, sliced from the same canonical state vector --
        # not a separate state representation (see observations/robot_state.py).
        self._eef_positions.append(state[EEF_POSITION_SLICE].copy())
        self._eef_orientations.append(state[EEF_ORIENTATION_SLICE].copy())

        self._step += 1

    def finalize(
        self,
        destination_root: Path,
        instruction: str,
        success: bool,
        extra_metadata: dict = None,
    ) -> Path:
        """Write trajectory.npz + metadata.json and atomically publish the episode.

        Returns the final episode directory path
        (``destination_root/episode_{episode_id:06d}``).
        """
        if self._finalized:
            raise RuntimeError(f"Episode {self.episode_id} already finalized")
        if self._step == 0:
            raise ValueError(f"Cannot finalize episode {self.episode_id} with zero recorded steps")

        np.savez(
            self._tmp_dir / TRAJECTORY_FILENAME,
            states=np.asarray(self._states, dtype=np.float64),
            joint_targets=np.asarray(self._joint_targets, dtype=np.float64),
            gripper_targets=np.asarray(self._gripper_targets, dtype=np.float64),
            timestamps=np.asarray(self._timestamps, dtype=np.float64),
            controller_stage=np.asarray(self._controller_stages, dtype="<U20"),
            eef_positions=np.asarray(self._eef_positions, dtype=np.float64),
            eef_orientations=np.asarray(self._eef_orientations, dtype=np.float64),
        )

        metadata = {
            "episode_id": self.episode_id,
            "instruction": instruction,
            "success": bool(success),
            "episode_length": self._step,
            "dataset_version": DATASET_VERSION,
            **(extra_metadata or {}),
        }
        with open(self._tmp_dir / METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        destination_root = Path(destination_root)
        destination_root.mkdir(parents=True, exist_ok=True)
        final_dir = destination_root / f"episode_{self.episode_id:06d}"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        self._tmp_dir.rename(final_dir)

        self._finalized = True
        return final_dir

    def abort(self) -> None:
        """Discard the episode: delete the .tmp directory without publishing it."""
        if self._finalized:
            raise RuntimeError(f"Episode {self.episode_id} already finalized, cannot abort")
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
