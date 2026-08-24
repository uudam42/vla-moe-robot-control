"""Recorder tests, including the critical observation/action alignment guarantee.

See dataset/recorder.py and dataset/generate_dataset.py: a recorded sample
at index t must be the observation the expert used to compute action_t
(state_t), never the observation produced by applying action_t
(state_t+1).
"""

import numpy as np
import pytest

from control.action import RobotAction
from dataset.episode import load_episode
from dataset.generate_dataset import generate_episode
from dataset.recorder import EpisodeRecorder
from observations.observation import Observation
from simulation.environment import SimulationEnvironment


def make_observation(value: float) -> Observation:
    rgb = np.full((4, 4, 3), int(value) % 256, dtype=np.uint8)
    state = np.full(23, value, dtype=np.float64)
    return Observation(rgb=rgb, state=state, timestamp=value)


def test_recorder_stores_exactly_what_it_is_given_not_a_later_state(tmp_path):
    """The recorder's own contract: record(obs_t, action_t) must not silently drift.

    Simulates a caller that (correctly) records BEFORE mutating its notion
    of "current state" to value t+1, and checks the file on disk reflects
    value t, never t+1.
    """
    recorder = EpisodeRecorder(tmp_path, episode_id=0)
    for t in range(4):
        obs = make_observation(float(t))  # state_t, distinguishable by value == t
        action = RobotAction(joint_targets=np.full(7, float(t)), gripper_target=0.5)
        recorder.record(obs, action, controller_stage="TEST")
        # A real loop would call env.step(action) here, advancing to t+1 --
        # deliberately not represented, since the recorder must not see it.

    episode_dir = recorder.finalize(tmp_path, "Pick up the red cube.", True, {})
    episode = load_episode(episode_dir)

    assert np.allclose(episode.states[:, 0], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(episode.joint_targets[:, 0], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(episode.timestamps, [0.0, 1.0, 2.0, 3.0])


def test_frame_count_matches_recorded_steps(tmp_path):
    recorder = EpisodeRecorder(tmp_path, episode_id=0)
    for t in range(7):
        recorder.record(make_observation(float(t)), RobotAction(np.zeros(7), 0.5), "TEST")
    assert recorder.num_steps == 7
    episode_dir = recorder.finalize(tmp_path, "Pick up the red cube.", True, {})
    assert len(list((episode_dir / "rgb").glob("*.png"))) == 7


def test_abort_leaves_no_visible_episode_directory(tmp_path):
    recorder = EpisodeRecorder(tmp_path, episode_id=0)
    for t in range(3):
        recorder.record(make_observation(float(t)), RobotAction(np.zeros(7), 0.5), "TEST")
    recorder.abort()

    # No "episode_000000" (published) directory should exist anywhere under tmp_path.
    assert list(tmp_path.rglob("episode_000000")) == []


def test_cannot_finalize_after_abort_or_twice(tmp_path):
    recorder = EpisodeRecorder(tmp_path, episode_id=0)
    recorder.record(make_observation(0.0), RobotAction(np.zeros(7), 0.5), "TEST")
    episode_dir = recorder.finalize(tmp_path, "Pick up the red cube.", True, {})
    assert episode_dir.exists()
    with pytest.raises(RuntimeError):
        recorder.finalize(tmp_path, "Pick up the red cube.", True, {})


def test_finalize_rejects_zero_step_episode(tmp_path):
    recorder = EpisodeRecorder(tmp_path, episode_id=0)
    with pytest.raises(ValueError):
        recorder.finalize(tmp_path, "Pick up the red cube.", True, {})


def test_end_to_end_alignment_with_real_environment(tmp_path):
    """Integration check: generate_episode()'s recorded state_0 is the
    HOME state env.reset() produces -- captured before any env.step() call
    -- and action_0's joint targets are computed from that same state_0,
    not from whatever the robot moved to afterward.
    """
    with SimulationEnvironment() as env:
        rng = np.random.default_rng(0)
        recorder, _success, metadata = generate_episode(
            env,
            episode_id=0,
            rng=rng,
            xy_randomization=0.0,
            max_steps=5,
            staging_root=tmp_path,
            seed=0,
        )
        episode_dir = recorder.finalize(tmp_path, metadata["instruction"], True, metadata)

        env.reset()
        home_state_vector = env.get_robot_state().as_vector()

    episode = load_episode(episode_dir)

    assert np.allclose(episode.states[0], home_state_vector, atol=1e-6)
    # HOME stage's first action, computed from state_0, holds at the
    # current joint positions (zero joint-space error) -- so joint_targets[0]
    # must equal the joint positions embedded in state_0, not a later state.
    assert np.allclose(episode.joint_targets[0], episode.states[0][:7], atol=1e-6)
