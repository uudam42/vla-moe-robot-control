"""Tests for ros_integration/episode_manager.py + instruction_cache.py --
reset ordering (README "Document ordering") and Temporal policy.reset()
propagation (README "Temporal policy reset" -- must not leak history).
"""

import torch

from robot_backend.fake_backend import FakeRobotBackend
from ros_integration.episode_manager import EpisodeManager
from ros_integration.instruction_cache import DEFAULT_INSTRUCTION, InstructionCache


class _OrderRecordingBackend(FakeRobotBackend):
    def __init__(self, log):
        super().__init__()
        self._log = log

    def reset(self):
        self._log.append("backend")
        super().reset()


class _OrderRecordingPolicy:
    def __init__(self, log):
        self._log = log

    def reset(self):
        self._log.append("policy")


def test_reset_order_is_backend_then_policy():
    log = []
    manager = EpisodeManager(_OrderRecordingBackend(log), _OrderRecordingPolicy(log))
    manager.reset()
    assert log == ["backend", "policy"]


def test_reset_clears_metrics():
    manager = EpisodeManager(FakeRobotBackend(), _OrderRecordingPolicy([]))
    manager.record_tick()
    manager.record_action_executed()
    assert manager.metrics.ticks == 1
    assert manager.metrics.actions_executed == 1

    manager.reset()
    assert manager.metrics.ticks == 0
    assert manager.metrics.actions_executed == 0


def test_reset_is_a_noop_for_policies_without_reset():
    class NoResetPolicy:
        pass

    manager = EpisodeManager(FakeRobotBackend(), NoResetPolicy())
    manager.reset()  # must not raise


def test_temporal_policy_history_does_not_leak_across_episode_manager_resets(tiny_temporal_vla_checkpoint):
    from models.temporal_policy import TemporalDenseVLAPolicy

    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    manager = EpisodeManager(FakeRobotBackend(), policy)

    for _ in range(4):
        observation = manager.backend.get_observation()
        action = policy.predict(observation=observation, instruction="Pick up the red cube.")
        manager.backend.execute_action(action)
        manager.record_tick()

    assert policy._tick == 4
    assert len(policy._issued_actions_by_index) == 4

    manager.reset()

    assert policy._tick == 0
    assert policy._issued_actions_by_index == {}
    assert manager.metrics.ticks == 0


# --- InstructionCache ---


def test_instruction_cache_defaults_before_any_update():
    cache = InstructionCache()
    assert cache.get() == DEFAULT_INSTRUCTION


def test_instruction_cache_accepts_valid_instruction():
    cache = InstructionCache()
    assert cache.update("Grasp the red cube.") is True
    assert cache.get() == "Grasp the red cube."


def test_instruction_cache_rejects_empty_or_whitespace():
    cache = InstructionCache()
    assert cache.update("") is False
    assert cache.update("   ") is False
    assert cache.get() == DEFAULT_INSTRUCTION  # unchanged


def test_instruction_cache_reset_restores_default():
    cache = InstructionCache()
    cache.update("Lift the red cube.")
    cache.reset()
    assert cache.get() == DEFAULT_INSTRUCTION


def test_episode_manager_module_never_imports_rclpy():
    import inspect

    import ros_integration.episode_manager as module

    assert not hasattr(module, "rclpy")
    assert "import rclpy" not in inspect.getsource(module)
