"""Tests for models/temporal_policy.py::TemporalDenseVLAPolicy -- the
history-buffer runtime adapter.
"""

import numpy as np
import torch

from control.action import RobotAction
from models.temporal_history import NO_ACTION_VECTOR
from models.temporal_policy import TemporalDenseVLAPolicy
from observations.observation import Observation


def make_observation(seed: int) -> Observation:
    rgb = np.random.default_rng(seed).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    state = np.random.default_rng(seed + 100).normal(size=23)
    return Observation(rgb=rgb, state=state, timestamp=float(seed))


def test_predict_returns_valid_robot_action(tiny_temporal_vla_checkpoint):
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    action = policy.predict(make_observation(0), "Pick up the red cube.")

    assert isinstance(action, RobotAction)
    assert action.joint_targets.shape == (7,)
    assert np.all(np.isfinite(action.joint_targets))
    assert 0.0 <= action.gripper_target <= 1.0


def test_reset_clears_all_history(tiny_temporal_vla_checkpoint):
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    for i in range(5):
        policy.predict(make_observation(i), "Pick up the red cube.")

    assert len(policy._rgb_history) > 0
    assert len(policy._issued_actions_by_index) > 0
    assert policy._tick > 0

    policy.reset()

    assert policy._rgb_history == []
    assert policy._state_history == []
    assert policy._issued_actions_by_index == {}
    assert policy._tick == 0


def test_first_prediction_after_reset_uses_all_masked_history(tiny_temporal_vla_checkpoint, monkeypatch):
    """First tick of an episode: no real previous action exists yet -- the
    model must be called with an all-NO_ACTION previous-action window
    (matches training's t=0 sample -- see tests/test_temporal_dataset.py)."""
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    captured = {}
    original_forward = policy.model.forward

    def spy_forward(pixel_values, states, previous_actions, input_ids, attention_mask):
        captured["previous_actions"] = previous_actions.clone()
        return original_forward(pixel_values, states, previous_actions, input_ids, attention_mask)

    monkeypatch.setattr(policy.model, "forward", spy_forward)
    policy.predict(make_observation(0), "Pick up the red cube.")

    previous_actions = captured["previous_actions"][0].numpy()  # (H, 8)
    for slot in range(previous_actions.shape[0]):
        assert np.allclose(previous_actions[slot], NO_ACTION_VECTOR)


def test_policy_history_uses_its_own_issued_actions_not_expert_actions(tiny_temporal_vla_checkpoint, monkeypatch):
    """README "Teacher-forcing distinction": at inference, previous-action
    history must be built from the policy's OWN previous outputs, never
    any externally-supplied "expert" action."""
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))

    action_0 = policy.predict(make_observation(0), "Pick up the red cube.")
    expected_vector = np.concatenate(
        [policy.action_normalizer.normalize_joints(action_0.joint_targets), [action_0.gripper_target]]
    )

    assert 0 in policy._issued_actions_by_index
    assert np.allclose(policy._issued_actions_by_index[0], expected_vector)

    captured = {}
    original_forward = policy.model.forward

    def spy_forward(pixel_values, states, previous_actions, input_ids, attention_mask):
        captured["previous_actions"] = previous_actions.clone()
        return original_forward(pixel_values, states, previous_actions, input_ids, attention_mask)

    monkeypatch.setattr(policy.model, "forward", spy_forward)
    policy.predict(make_observation(1), "Pick up the red cube.")

    # Second tick's history: slots 0,1 masked (before episode start), slot 2
    # = the policy's own first issued action, slot 3 (current) masked.
    previous_actions = captured["previous_actions"][0].numpy()
    assert np.allclose(previous_actions[2], expected_vector, atol=1e-5)
    assert np.allclose(previous_actions[3], NO_ACTION_VECTOR)


def test_history_buffer_caps_at_history_length(tiny_temporal_vla_checkpoint):
    policy = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    for i in range(10):
        policy.predict(make_observation(i), "Pick up the red cube.")
    assert len(policy._rgb_history) == policy.history_length
    assert len(policy._state_history) == policy.history_length


def test_predict_never_imports_or_calls_mujoco(tiny_temporal_vla_checkpoint):
    import inspect

    import models.temporal_policy as policy_module

    assert not hasattr(policy_module, "mujoco")
    assert "import mujoco" not in inspect.getsource(policy_module)
