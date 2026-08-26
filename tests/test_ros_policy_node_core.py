"""Tests for ros_integration/policy_node_core.py -- the VLA policy node's
rclpy-independent control loop."""

import inspect

import numpy as np
import torch

from ros_integration.policy_node_core import VLAPolicyNodeCore
from ros_integration.sync import StalenessChecker


def make_rgb():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_tick_skips_inference_when_unsynchronized(tiny_temporal_vla_checkpoint):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    core = VLAPolicyNodeCore(policy)
    action = core.tick(now=1.0)
    assert action is None
    assert core.stats["skipped_unsynced"] == 1
    assert core.stats["inferences"] == 0


def test_tick_skips_inference_when_stale(tiny_temporal_vla_checkpoint):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    core = VLAPolicyNodeCore(policy, staleness_checker=StalenessChecker(max_age_sec=0.1))
    core.on_image(make_rgb(), timestamp=1.0)
    core.on_state(np.zeros(23), timestamp=1.0)

    action = core.tick(now=5.0)  # way past the 0.1s staleness threshold
    assert action is None
    assert core.stats["skipped_stale"] == 1
    assert core.stats["inferences"] == 0


def test_tick_predicts_when_synced_and_fresh(tiny_temporal_vla_checkpoint):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    core = VLAPolicyNodeCore(policy, staleness_checker=StalenessChecker(max_age_sec=10.0))
    core.on_image(make_rgb(), timestamp=1.0)
    core.on_state(np.zeros(23), timestamp=1.0)

    action = core.tick(now=1.05)
    assert action is not None
    assert np.all(np.isfinite(action.joint_targets))
    assert core.stats["inferences"] == 1


def test_instruction_cached_and_used_by_predict(tiny_temporal_vla_checkpoint, monkeypatch):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    core = VLAPolicyNodeCore(policy, staleness_checker=StalenessChecker(max_age_sec=10.0))
    core.on_instruction("Grasp the red cube.")
    core.on_image(make_rgb(), timestamp=1.0)
    core.on_state(np.zeros(23), timestamp=1.0)

    captured = {}
    original_predict = TemporalDenseVLAPolicy.predict

    def spy_predict(self, observation, instruction):
        captured["instruction"] = instruction
        return original_predict(self, observation, instruction)

    monkeypatch.setattr(TemporalDenseVLAPolicy, "predict", spy_predict)
    core.tick(now=1.05)
    assert captured["instruction"] == "Grasp the red cube."


def test_reset_clears_sync_state_and_policy_history(tiny_temporal_vla_checkpoint):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    core = VLAPolicyNodeCore(policy, staleness_checker=StalenessChecker(max_age_sec=10.0))
    core.on_image(make_rgb(), timestamp=1.0)
    core.on_state(np.zeros(23), timestamp=1.0)
    core.tick(now=1.05)
    assert policy._tick == 1

    core.reset()
    assert policy._tick == 0
    assert core.sync.get_synced() is None  # stale in-flight pair cleared


def test_policy_node_core_never_imports_mujoco():
    import ros_integration.policy_node_core as module

    assert not hasattr(module, "mujoco")
    assert "import mujoco" not in inspect.getsource(module)
    assert "SimulationEnvironment" not in inspect.getsource(module)
