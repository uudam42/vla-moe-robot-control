"""Tests for tools/replay_episode.py (README "Replay Integrity": must use
recorded data only, never call the VLA policy again)."""

import inspect

import numpy as np
import torch

from robot_backend.fake_backend import FakeRobotBackend
from runtime.run_episode import run_episode
from safety.supervisor import SafetySupervisor
from telemetry.recorder import EpisodeConfig, EpisodeTelemetryRecorder
from tools.replay_episode import load_episode, replay


def _record_episode(tmp_path, tiny_temporal_vla_checkpoint, max_steps=5):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    backend = FakeRobotBackend()
    supervisor = SafetySupervisor()
    config = EpisodeConfig(
        policy_type="temporal", checkpoint=str(tiny_temporal_vla_checkpoint), instruction="Pick up the red cube.",
        seed=42, cube_xy_randomization=0.03, backend="fake", device="cpu", max_steps=max_steps, control_substeps=10,
    )
    recorder = EpisodeTelemetryRecorder(config, output_root=tmp_path)
    run_episode(backend, policy, supervisor, recorder, "Pick up the red cube.", max_steps=max_steps)
    return recorder.episode_dir


def test_load_episode_reads_back_all_three_files(tmp_path, tiny_temporal_vla_checkpoint):
    episode_dir = _record_episode(tmp_path, tiny_temporal_vla_checkpoint, max_steps=4)
    episode = load_episode(episode_dir)
    assert episode["metadata"]["policy_type"] == "temporal"
    assert episode["metrics"]["episode_steps"] == 4
    assert len(episode["steps"]) == 4


def test_replay_prints_summary_without_raising(tmp_path, tiny_temporal_vla_checkpoint, capsys):
    episode_dir = _record_episode(tmp_path, tiny_temporal_vla_checkpoint, max_steps=3)
    replay(episode_dir)
    captured = capsys.readouterr()
    assert "Episode:" in captured.out
    assert "step 0000" in captured.out
    assert "Success:" in captured.out


def test_replay_json_mode_matches_loaded_episode(tmp_path, tiny_temporal_vla_checkpoint, capsys):
    episode_dir = _record_episode(tmp_path, tiny_temporal_vla_checkpoint, max_steps=2)
    import json

    replay(episode_dir, as_json=True)
    captured = capsys.readouterr()
    dumped = json.loads(captured.out)
    assert dumped["metrics"]["episode_steps"] == 2


def test_replay_limit_truncates_output(tmp_path, tiny_temporal_vla_checkpoint, capsys):
    episode_dir = _record_episode(tmp_path, tiny_temporal_vla_checkpoint, max_steps=10)
    replay(episode_dir, limit=3)
    captured = capsys.readouterr()
    assert captured.out.count("step ") == 3
    assert "more steps not shown" in captured.out


def test_replay_never_imports_a_policy_class():
    """Replay integrity: must reconstruct the story from telemetry.jsonl
    alone, never re-instantiate/call a policy."""
    import tools.replay_episode as module

    source = inspect.getsource(module)
    for forbidden in ("DenseVLAPolicy", "MoEVLAPolicy", "TemporalDenseVLAPolicy", "policy.predict", "build_policy"):
        assert forbidden not in source
    assert not hasattr(module, "torch")


def test_replay_never_imports_mujoco():
    import tools.replay_episode as module

    assert not hasattr(module, "mujoco")
    assert "import mujoco" not in inspect.getsource(module)
