"""On-disk episode format: one directory per episode.

```
episode_000000/
    rgb/000000.png, 000001.png, ...
    trajectory.npz   (states, joint_targets, gripper_targets, timestamps,
                       controller_stage, eef_positions, eef_orientations)
    metadata.json    (episode-level info: instruction, success, ...)
```

This module only knows the file format -- it doesn't run the simulator or
the controller (that's ``dataset.recorder`` / ``dataset.generate_dataset``)
and doesn't build a multi-episode index (that's ``dataset.loader``).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RGB_DIRNAME = "rgb"
TRAJECTORY_FILENAME = "trajectory.npz"
METADATA_FILENAME = "metadata.json"

# Arrays stored in trajectory.npz that must all share the same leading
# dimension T (episode length). "state"/"joint_targets"/"gripper_targets"
# are the training-relevant fields; "controller_stage"/"eef_positions"/
# "eef_orientations" are diagnostics only (see README "Dataset schema").
TRAJECTORY_ARRAY_NAMES = (
    "states",
    "joint_targets",
    "gripper_targets",
    "timestamps",
    "controller_stage",
    "eef_positions",
    "eef_orientations",
)


def frame_path(episode_dir: Path, step: int) -> Path:
    """Path to the RGB frame for a given timestep within an episode directory."""
    return episode_dir / RGB_DIRNAME / f"{step:06d}.png"


@dataclass
class Episode:
    """An episode loaded back from disk.

    Attributes:
        episode_dir: Directory this episode was loaded from.
        instruction: The single language instruction for the whole episode.
        states: Training input, shape ``(T, 23)`` -- see
            ``observations.robot_state.RobotState.as_vector()``.
        joint_targets: Training label component, shape ``(T, 7)``.
        gripper_targets: Training label component, shape ``(T,)``.
        timestamps: Simulation time at each recorded step, shape ``(T,)``.
        controller_stage: Diagnostic only -- ``ScriptedController.stage``
            name at each step, shape ``(T,)``, string dtype.
        eef_positions: Diagnostic only, shape ``(T, 3)``.
        eef_orientations: Diagnostic only, shape ``(T, 4)``.
        rgb_frame_paths: Length-``T`` list of per-timestep frame paths.
        metadata: Parsed ``metadata.json`` contents.
    """

    episode_dir: Path
    instruction: str
    states: np.ndarray
    joint_targets: np.ndarray
    gripper_targets: np.ndarray
    timestamps: np.ndarray
    controller_stage: np.ndarray
    eef_positions: np.ndarray
    eef_orientations: np.ndarray
    rgb_frame_paths: list
    metadata: dict

    @property
    def length(self) -> int:
        return int(self.states.shape[0])

    def action_vector(self, t: int) -> np.ndarray:
        """8D training label at timestep ``t``: ``[joint_targets, gripper_target]``."""
        return np.concatenate(
            [self.joint_targets[t], [self.gripper_targets[t]]]
        ).astype(np.float64)


def load_episode(episode_dir: Path) -> Episode:
    """Load one episode directory back into memory (arrays + metadata only, not RGB pixels)."""
    episode_dir = Path(episode_dir)
    metadata_path = episode_dir / METADATA_FILENAME
    trajectory_path = episode_dir / TRAJECTORY_FILENAME
    rgb_dir = episode_dir / RGB_DIRNAME

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {METADATA_FILENAME} in {episode_dir}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing {TRAJECTORY_FILENAME} in {episode_dir}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing {RGB_DIRNAME}/ directory in {episode_dir}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with np.load(trajectory_path, allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in TRAJECTORY_ARRAY_NAMES}

    rgb_frame_paths = sorted(rgb_dir.glob("*.png"))

    instruction = metadata.get("instruction", "")

    return Episode(
        episode_dir=episode_dir,
        instruction=instruction,
        states=arrays["states"],
        joint_targets=arrays["joint_targets"],
        gripper_targets=arrays["gripper_targets"],
        timestamps=arrays["timestamps"],
        controller_stage=arrays["controller_stage"],
        eef_positions=arrays["eef_positions"],
        eef_orientations=arrays["eef_orientations"],
        rgb_frame_paths=rgb_frame_paths,
        metadata=metadata,
    )
