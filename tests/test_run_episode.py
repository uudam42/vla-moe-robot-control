"""Tests for runtime/run_episode.py -- the Step 10 production runtime loop
(policy -> SafetySupervisor -> RobotBackend, with telemetry)."""

import json

import numpy as np
import torch

from robot_backend.fake_backend import FakeRobotBackend
from robot_backend.mujoco_backend import MuJoCoBackend
from runtime.run_episode import run_episode
from safety.supervisor import SafetySupervisor
from simulation.environment import SimulationEnvironment
from telemetry.recorder import EpisodeConfig, EpisodeTelemetryRecorder


def make_config(tmp_checkpoint):
    return EpisodeConfig(
        policy_type="temporal", checkpoint=str(tmp_checkpoint), instruction="Pick up the red cube.",
        seed=42, cube_xy_randomization=0.03, backend="fake", device="cpu", max_steps=8, control_substeps=10,
    )


def test_run_episode_against_fake_backend_no_privileged_success_detection(tmp_path, tiny_temporal_vla_checkpoint):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    backend = FakeRobotBackend()
    supervisor = SafetySupervisor()
    recorder = EpisodeTelemetryRecorder(make_config(tiny_temporal_vla_checkpoint), output_root=tmp_path)

    metrics = run_episode(backend, policy, supervisor, recorder, "Pick up the red cube.", max_steps=8)

    assert metrics["episode_steps"] == 8
    assert metrics["success"] is False  # FakeRobotBackend has no cube -- success detection skipped entirely
    assert metrics["max_cube_lift"] is None
    assert len(backend.executed_actions) == 8


def test_run_episode_writes_telemetry_and_metadata(tmp_path, tiny_temporal_vla_checkpoint):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    backend = FakeRobotBackend()
    supervisor = SafetySupervisor()
    recorder = EpisodeTelemetryRecorder(make_config(tiny_temporal_vla_checkpoint), output_root=tmp_path)

    run_episode(backend, policy, supervisor, recorder, "Pick up the red cube.", max_steps=5)

    episode_dir = recorder.episode_dir
    assert (episode_dir / "metadata.json").exists()
    assert (episode_dir / "metrics.json").exists()
    lines = (episode_dir / "telemetry.jsonl").read_text().strip().split("\n")
    assert len(lines) == 5
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["step_id"] == i
        assert record["policy_type"] == "temporal"
        assert record["safety_decision"] in ("ACCEPT", "CLAMP", "HOLD", "REJECT", "STOP_EPISODE")


def test_run_episode_policy_receives_only_observation_and_instruction(tmp_path, tiny_temporal_vla_checkpoint, monkeypatch):
    from models.temporal_policy import TemporalDenseVLAPolicy
    from observations.observation import Observation

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    calls = []
    original_predict = TemporalDenseVLAPolicy.predict

    def spy_predict(self, observation, instruction):
        calls.append((observation, instruction))
        return original_predict(self, observation, instruction)

    monkeypatch.setattr(TemporalDenseVLAPolicy, "predict", spy_predict)

    backend = FakeRobotBackend()
    supervisor = SafetySupervisor()
    recorder = EpisodeTelemetryRecorder(make_config(tiny_temporal_vla_checkpoint), output_root=tmp_path)
    run_episode(backend, policy, supervisor, recorder, "Pick up the red cube.", max_steps=3)

    assert len(calls) == 3
    for observation, instruction in calls:
        assert isinstance(observation, Observation)
        assert instruction == "Pick up the red cube."


def test_run_episode_resets_policy_and_safety_between_episodes(tmp_path, tiny_temporal_vla_checkpoint):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    backend = FakeRobotBackend()
    supervisor = SafetySupervisor()

    recorder1 = EpisodeTelemetryRecorder(make_config(tiny_temporal_vla_checkpoint), output_root=tmp_path)
    run_episode(backend, policy, supervisor, recorder1, "Pick up the red cube.", max_steps=4)
    assert policy._tick == 4

    recorder2 = EpisodeTelemetryRecorder(make_config(tiny_temporal_vla_checkpoint), output_root=tmp_path)
    run_episode(backend, policy, supervisor, recorder2, "Pick up the red cube.", max_steps=2)
    assert policy._tick == 2  # not 6 -- history did not leak across episodes


def test_run_episode_with_mujoco_backend_smoke(tmp_path, tiny_temporal_vla_checkpoint):
    """Real MuJoCo backend, privileged success-detection path exercised
    (even though the tiny random-weight policy won't actually succeed)."""
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    with SimulationEnvironment() as env:
        backend = MuJoCoBackend(env=env)
        supervisor = SafetySupervisor(joint_range=env.get_joint_range())
        recorder = EpisodeTelemetryRecorder(make_config(tiny_temporal_vla_checkpoint), output_root=tmp_path)

        metrics = run_episode(backend, policy, supervisor, recorder, "Pick up the red cube.", max_steps=6)

    assert metrics["episode_steps"] == 6
    assert metrics["max_cube_lift"] is not None  # privileged success detection WAS exercised


def test_run_episode_module_never_imports_mujoco_directly():
    import inspect

    import runtime.run_episode as module

    assert not hasattr(module, "mujoco")
    assert "import mujoco" not in inspect.getsource(module)
