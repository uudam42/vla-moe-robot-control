"""Episode save/load round-trip tests (dataset/episode.py + dataset/recorder.py)."""

import numpy as np

from control.action import RobotAction
from dataset.episode import load_episode
from dataset.recorder import EpisodeRecorder
from observations.observation import Observation


def make_observation(t: int) -> Observation:
    rgb = np.full((8, 8, 3), t % 256, dtype=np.uint8)
    state = np.arange(23, dtype=np.float64) + t
    return Observation(rgb=rgb, state=state, timestamp=float(t) * 0.02)


def write_synthetic_episode(tmp_path, episode_id=0, length=5, instruction="Pick up the red cube.", success=True):
    recorder = EpisodeRecorder(tmp_path, episode_id)
    for t in range(length):
        obs = make_observation(t)
        action = RobotAction(joint_targets=np.full(7, 0.1 * t), gripper_target=0.5)
        recorder.record(obs, action, controller_stage="DESCEND")
    destination = tmp_path / "successful"
    episode_dir = recorder.finalize(destination, instruction, success, {"cube_lift_delta": 0.1})
    return episode_dir


def test_round_trip_arrays_and_instruction_match(tmp_path):
    episode_dir = write_synthetic_episode(tmp_path, length=5)
    episode = load_episode(episode_dir)

    assert episode.length == 5
    assert episode.instruction == "Pick up the red cube."
    assert episode.states.shape == (5, 23)
    assert episode.joint_targets.shape == (5, 7)
    assert episode.gripper_targets.shape == (5,)
    assert episode.timestamps.shape == (5,)
    assert len(episode.rgb_frame_paths) == 5

    for t in range(5):
        expected_state = np.arange(23, dtype=np.float64) + t
        assert np.allclose(episode.states[t], expected_state)
        assert np.allclose(episode.joint_targets[t], 0.1 * t)
        assert episode.gripper_targets[t] == 0.5
        assert np.isclose(episode.timestamps[t], t * 0.02)


def test_round_trip_metadata_matches(tmp_path):
    episode_dir = write_synthetic_episode(tmp_path, episode_id=3, instruction="Grasp the red cube.")
    episode = load_episode(episode_dir)

    assert episode.metadata["episode_id"] == 3
    assert episode.metadata["instruction"] == "Grasp the red cube."
    assert episode.metadata["success"] is True
    assert episode.metadata["episode_length"] == 5
    assert episode.metadata["cube_lift_delta"] == 0.1
    assert "dataset_version" in episode.metadata


def test_action_vector_matches_joint_and_gripper_targets(tmp_path):
    episode_dir = write_synthetic_episode(tmp_path, length=3)
    episode = load_episode(episode_dir)

    action = episode.action_vector(1)
    assert action.shape == (8,)
    assert np.allclose(action[:7], episode.joint_targets[1])
    assert action[7] == episode.gripper_targets[1]


def test_frame_files_are_zero_padded_and_contiguous(tmp_path):
    episode_dir = write_synthetic_episode(tmp_path, length=12)
    names = sorted(p.name for p in (episode_dir / "rgb").glob("*.png"))
    assert names == [f"{i:06d}.png" for i in range(12)]
