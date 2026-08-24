"""Dataset validation utilities.

Pure checks over on-disk episodes -- no simulation, no training. Used both
by ``dataset/generate_dataset.py`` (sanity-check what it just wrote) and
as a standalone ``python -m dataset.validation <root>`` command.
"""

from pathlib import Path

import numpy as np

from observations.robot_state import STATE_DIM

from .episode import load_episode


def validate_episode(episode_dir: Path, require_success: bool = True) -> list:
    """Check one episode directory. Returns a list of problem strings (empty = valid)."""
    episode_dir = Path(episode_dir)
    problems = []

    try:
        episode = load_episode(episode_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return [f"failed to load episode: {exc}"]

    T = episode.length

    if len(episode.rgb_frame_paths) != T:
        problems.append(f"RGB frame count {len(episode.rgb_frame_paths)} != T={T}")
    else:
        expected_names = {f"{i:06d}.png" for i in range(T)}
        actual_names = {p.name for p in episode.rgb_frame_paths}
        if actual_names != expected_names:
            problems.append("RGB frame numbers are not contiguous 000000..T-1")

    if episode.states.shape != (T, STATE_DIM):
        problems.append(f"states.shape {episode.states.shape} != (T={T}, {STATE_DIM})")
    if episode.joint_targets.shape != (T, 7):
        problems.append(f"joint_targets.shape {episode.joint_targets.shape} != (T={T}, 7)")
    if episode.gripper_targets.shape != (T,):
        problems.append(f"gripper_targets.shape {episode.gripper_targets.shape} != (T={T},)")
    if episode.timestamps.shape != (T,):
        problems.append(f"timestamps.shape {episode.timestamps.shape} != (T={T},)")

    for name, array in (
        ("states", episode.states),
        ("joint_targets", episode.joint_targets),
        ("gripper_targets", episode.gripper_targets),
        ("timestamps", episode.timestamps),
    ):
        if array.size and not np.all(np.isfinite(array)):
            problems.append(f"{name} contains non-finite values")

    if episode.gripper_targets.size and (
        np.any(episode.gripper_targets < 0.0) or np.any(episode.gripper_targets > 1.0)
    ):
        problems.append("gripper_targets outside [0, 1]")

    if episode.timestamps.size > 1 and np.any(np.diff(episode.timestamps) < 0):
        problems.append("timestamps are not monotonically non-decreasing")

    if not episode.instruction or not episode.instruction.strip():
        problems.append("instruction is empty")

    if require_success and not episode.metadata.get("success", False):
        problems.append("metadata.success is not true")

    return problems


def validate_dataset(root: Path, split_subdir: str = "successful", require_success: bool = True) -> dict:
    """Validate every episode under ``root/split_subdir``. Returns a summary dict."""
    episodes_root = Path(root) / split_subdir
    episode_dirs = sorted(p for p in episodes_root.glob("episode_*") if p.is_dir())

    problems_by_episode = {}
    for episode_dir in episode_dirs:
        problems = validate_episode(episode_dir, require_success=require_success)
        if problems:
            problems_by_episode[episode_dir.name] = problems

    return {
        "root": str(episodes_root),
        "num_episodes": len(episode_dirs),
        "num_valid": len(episode_dirs) - len(problems_by_episode),
        "num_invalid": len(problems_by_episode),
        "problems": problems_by_episode,
    }


if __name__ == "__main__":
    import sys

    root_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/demonstrations")
    report = validate_dataset(root_arg)
    print(f"Validated {report['num_episodes']} episodes under {report['root']}")
    print(f"Valid:   {report['num_valid']}")
    print(f"Invalid: {report['num_invalid']}")
    for name, problems in report["problems"].items():
        print(f"  {name}:")
        for problem in problems:
            print(f"    - {problem}")
