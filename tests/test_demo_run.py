"""Smoke test for demo/run.py -- the Step 10 showcase entry point.
Uses the tiny random-weight checkpoint and a short episode so it stays fast."""

from pathlib import Path

from demo.run import main as demo_main


def test_demo_run_smoke_no_record(tmp_path, tiny_temporal_vla_checkpoint):
    metrics = demo_main([
        "--backend", "mujoco", "--policy", "temporal",
        "--checkpoint", str(tiny_temporal_vla_checkpoint),
        "--instruction", "Pick up the red cube.",
        "--seed", "0", "--max-steps", "5", "--output", str(tmp_path),
    ])
    assert metrics["episode_steps"] == 5

    episode_dirs = list(Path(tmp_path).glob("episode_*"))
    assert len(episode_dirs) == 1
    assert (episode_dirs[0] / "metadata.json").exists()
    assert (episode_dirs[0] / "telemetry.jsonl").exists()
    assert not (episode_dirs[0] / "frames").exists()  # --record not passed


def test_demo_run_smoke_with_record_saves_frames(tmp_path, tiny_temporal_vla_checkpoint):
    metrics = demo_main([
        "--backend", "mujoco", "--policy", "temporal",
        "--checkpoint", str(tiny_temporal_vla_checkpoint),
        "--seed", "1", "--max-steps", "3", "--output", str(tmp_path), "--record",
    ])
    episode_dirs = list(Path(tmp_path).glob("episode_*"))
    frames_dir = episode_dirs[0] / "frames"
    assert frames_dir.is_dir()
    assert len(list(frames_dir.glob("*.png"))) == 3


def test_demo_run_rejects_unsupported_backend(tiny_temporal_vla_checkpoint):
    import pytest

    with pytest.raises(SystemExit):
        demo_main(["--backend", "real_robot", "--policy", "temporal", "--checkpoint", str(tiny_temporal_vla_checkpoint)])
