"""Tests for telemetry/recorder.py."""

import json

import numpy as np
import pytest

from control.action import RobotAction
from telemetry.recorder import EpisodeConfig, EpisodeTelemetryRecorder


def make_config(**overrides):
    defaults = dict(
        policy_type="temporal", checkpoint="outputs/training/temporal_dense_vla_run_001/best.pt",
        instruction="Pick up the red cube.", seed=42, cube_xy_randomization=0.03,
        backend="mujoco", device="cpu", max_steps=10, control_substeps=10,
    )
    defaults.update(overrides)
    return EpisodeConfig(**defaults)


def make_action(value=0.0, gripper=1.0):
    return RobotAction(joint_targets=np.full(7, value), gripper_target=gripper)


def test_start_creates_expected_directory_structure(tmp_path):
    recorder = EpisodeTelemetryRecorder(make_config(), output_root=tmp_path)
    episode_dir = recorder.start(save_frames=True)
    assert episode_dir.exists()
    assert (episode_dir / "telemetry.jsonl").exists()
    assert (episode_dir / "frames").is_dir()


def test_record_step_writes_one_jsonl_line_per_call(tmp_path):
    recorder = EpisodeTelemetryRecorder(make_config(), output_root=tmp_path)
    episode_dir = recorder.start()
    for step in range(3):
        recorder.record_step(
            step_id=step, wall_clock_timestamp=float(step), observation_timestamp=float(step),
            prediction_start=float(step), prediction_end=float(step) + 0.01,
            original_action=make_action(step), executed_action=make_action(step),
            safety_decision="ACCEPT", safety_reason="NONE",
        )
    recorder.finalize(success=False)

    lines = (episode_dir / "telemetry.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["step_id"] == i
        assert record["safety_decision"] == "ACCEPT"
        assert record["executed_action"]["joint_targets"] == [float(i)] * 7


def test_record_step_never_stores_raw_frames_inline(tmp_path):
    """README "Do not store huge RGB frames directly inside each safety
    event" -- verified here for telemetry too: no 'rgb' key anywhere."""
    recorder = EpisodeTelemetryRecorder(make_config(), output_root=tmp_path)
    episode_dir = recorder.start()
    recorder.record_step(
        step_id=0, wall_clock_timestamp=0.0, observation_timestamp=0.0,
        prediction_start=0.0, prediction_end=0.01,
        original_action=make_action(), executed_action=make_action(),
        safety_decision="ACCEPT", safety_reason="NONE",
    )
    recorder.finalize(success=False)
    line = (episode_dir / "telemetry.jsonl").read_text().strip()
    assert "rgb" not in json.loads(line)


def test_metadata_json_has_required_fields(tmp_path):
    recorder = EpisodeTelemetryRecorder(make_config(), output_root=tmp_path)
    episode_dir = recorder.start()
    recorder.finalize(success=True, failure_reason=None)

    metadata = json.loads((episode_dir / "metadata.json").read_text())
    for key in ["policy_type", "checkpoint", "instruction", "seed", "cube_xy_randomization",
                "backend", "device", "max_steps", "control_substeps", "start_time", "end_time",
                "success", "failure_reason"]:
        assert key in metadata
    assert metadata["success"] is True
    assert metadata["policy_type"] == "temporal"


def test_metrics_json_aggregates_latency_and_gripper_switches(tmp_path):
    recorder = EpisodeTelemetryRecorder(make_config(), output_root=tmp_path)
    episode_dir = recorder.start()
    grippers = [1.0, 1.0, 0.0, 0.0, 1.0]  # 2 switches
    for step, g in enumerate(grippers):
        recorder.record_step(
            step_id=step, wall_clock_timestamp=float(step), observation_timestamp=float(step),
            prediction_start=float(step), prediction_end=float(step) + 0.005,
            original_action=make_action(gripper=g), executed_action=make_action(gripper=g),
            safety_decision="ACCEPT", safety_reason="NONE",
        )
    metrics = recorder.finalize(success=True, cube_lift_delta=0.05, max_cube_lift=0.06)

    assert metrics["episode_steps"] == 5
    assert metrics["gripper_switch_count"] == 2
    assert metrics["final_cube_lift"] == 0.05
    assert metrics["max_cube_lift"] == 0.06
    assert metrics["inference_latency_ms"]["n"] == 5
    assert metrics["inference_latency_ms"]["mean_ms"] == pytest.approx(5.0, abs=0.1)


def test_safety_intervention_counts_tracked(tmp_path):
    recorder = EpisodeTelemetryRecorder(make_config(), output_root=tmp_path)
    recorder.start()
    decisions = ["ACCEPT", "CLAMP", "HOLD", "ACCEPT", "REJECT"]
    reasons = ["NONE", "JOINT_LIMIT", "STALE_OBSERVATION", "NONE", "NONFINITE_ACTION"]
    for step, (d, r) in enumerate(zip(decisions, reasons)):
        recorder.record_step(
            step_id=step, wall_clock_timestamp=float(step), observation_timestamp=float(step),
            prediction_start=float(step), prediction_end=float(step) + 0.005,
            original_action=make_action(), executed_action=make_action(),
            safety_decision=d, safety_reason=r,
        )
    metrics = recorder.finalize(success=False)
    assert metrics["safety_intervention_count"] == 3  # CLAMP, HOLD, REJECT
    assert metrics["stale_observation_count"] == 1


def test_finalize_records_git_commit_or_none(tmp_path):
    recorder = EpisodeTelemetryRecorder(make_config(), output_root=tmp_path)
    episode_dir = recorder.start()
    recorder.finalize(success=False)
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    assert "git_commit" in metadata  # value may be a hash or None, but key must exist


def test_telemetry_module_never_imports_mujoco():
    import inspect

    import telemetry.recorder as module

    assert not hasattr(module, "mujoco")
    assert "import mujoco" not in inspect.getsource(module)
