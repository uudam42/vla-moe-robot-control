"""Tests for dataset/validation.py."""

import json

import numpy as np

from control.action import RobotAction
from dataset.episode import METADATA_FILENAME, TRAJECTORY_FILENAME
from dataset.recorder import EpisodeRecorder
from dataset.validation import validate_episode
from observations.observation import Observation


def make_observation(t: int) -> Observation:
    rgb = np.full((4, 4, 3), t % 256, dtype=np.uint8)
    state = np.arange(23, dtype=np.float64) + t
    return Observation(rgb=rgb, state=state, timestamp=float(t) * 0.02)


def write_valid_episode(tmp_path, length=5):
    recorder = EpisodeRecorder(tmp_path, episode_id=0)
    for t in range(length):
        recorder.record(
            make_observation(t), RobotAction(np.full(7, 0.1 * t), 0.5), "TEST"
        )
    return recorder.finalize(tmp_path / "successful", "Pick up the red cube.", True, {})


def test_valid_episode_has_no_problems(tmp_path):
    episode_dir = write_valid_episode(tmp_path)
    assert validate_episode(episode_dir) == []


def test_detects_missing_rgb_frame(tmp_path):
    episode_dir = write_valid_episode(tmp_path)
    (episode_dir / "rgb" / "000002.png").unlink()
    problems = validate_episode(episode_dir)
    assert any("frame count" in p for p in problems)


def test_detects_non_finite_state(tmp_path):
    episode_dir = write_valid_episode(tmp_path)
    with np.load(episode_dir / TRAJECTORY_FILENAME) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["states"][2, 0] = np.nan
    np.savez(episode_dir / TRAJECTORY_FILENAME, **arrays)

    problems = validate_episode(episode_dir)
    assert any("non-finite" in p for p in problems)


def test_detects_gripper_target_out_of_range(tmp_path):
    episode_dir = write_valid_episode(tmp_path)
    with np.load(episode_dir / TRAJECTORY_FILENAME) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["gripper_targets"][0] = 1.5
    np.savez(episode_dir / TRAJECTORY_FILENAME, **arrays)

    problems = validate_episode(episode_dir)
    assert any("[0, 1]" in p for p in problems)


def test_detects_non_monotonic_timestamps(tmp_path):
    episode_dir = write_valid_episode(tmp_path)
    with np.load(episode_dir / TRAJECTORY_FILENAME) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["timestamps"][3] = arrays["timestamps"][1]
    np.savez(episode_dir / TRAJECTORY_FILENAME, **arrays)

    problems = validate_episode(episode_dir)
    assert any("monotonic" in p for p in problems)


def test_detects_empty_instruction(tmp_path):
    episode_dir = write_valid_episode(tmp_path)
    metadata_path = episode_dir / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text())
    metadata["instruction"] = ""
    metadata_path.write_text(json.dumps(metadata))

    problems = validate_episode(episode_dir)
    assert any("instruction is empty" in p for p in problems)


def test_detects_unsuccessful_episode_when_success_required(tmp_path):
    episode_dir = write_valid_episode(tmp_path)
    metadata_path = episode_dir / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text())
    metadata["success"] = False
    metadata_path.write_text(json.dumps(metadata))

    problems = validate_episode(episode_dir, require_success=True)
    assert any("success" in p for p in problems)
    assert validate_episode(episode_dir, require_success=False) == []


def test_missing_episode_directory_reports_problem(tmp_path):
    problems = validate_episode(tmp_path / "does_not_exist")
    assert len(problems) == 1
    assert "failed to load" in problems[0]
