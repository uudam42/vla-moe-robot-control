"""Tests for models/policy.py::DenseVLAPolicy -- the inference adapter.

Verifies the boundary that matters for Step 5: predict() must return a
valid, existing RobotAction and must never touch MuJoCo (see README
"No closed-loop claims yet").
"""

import inspect

import numpy as np
import torch

from control.action import RobotAction
from models.dense_vla import DenseVLA, DenseVLAConfig
from models.policy import DenseVLAPolicy
from observations.observation import Observation
from training.checkpoint import save_checkpoint
from training.normalization import ActionNormalizer, StateNormalizer


def make_policy(tmp_path) -> DenseVLAPolicy:
    config = DenseVLAConfig(hidden_dim=64, num_layers=2, num_heads=4, ffn_dim=128)
    model = DenseVLA(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state_normalizer = StateNormalizer(mean=np.zeros(23), std=np.ones(23) * 2.0)
    action_normalizer = ActionNormalizer(joint_mean=np.zeros(7), joint_std=np.ones(7) * 0.5)

    checkpoint_path = tmp_path / "policy_test.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, epoch=0, config=config,
        state_normalizer=state_normalizer, action_normalizer=action_normalizer, val_metrics={},
    )
    return DenseVLAPolicy.from_checkpoint(checkpoint_path, device=torch.device("cpu"))


def make_observation() -> Observation:
    rgb = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    state = np.random.default_rng(1).normal(size=23)
    return Observation(rgb=rgb, state=state, timestamp=1.23)


def test_predict_returns_valid_robot_action(tmp_path):
    policy = make_policy(tmp_path)
    action = policy.predict(make_observation(), "Pick up the red cube.")

    assert isinstance(action, RobotAction)
    assert action.joint_targets.shape == (7,)
    assert np.all(np.isfinite(action.joint_targets))
    assert 0.0 <= action.gripper_target <= 1.0
    assert np.isfinite(action.gripper_target)


def test_predict_is_deterministic_in_eval_mode(tmp_path):
    policy = make_policy(tmp_path)
    observation = make_observation()

    action_1 = policy.predict(observation, "Pick up the red cube.")
    action_2 = policy.predict(observation, "Pick up the red cube.")

    assert np.allclose(action_1.joint_targets, action_2.joint_targets)
    assert action_1.gripper_target == action_2.gripper_target


def test_predict_never_imports_or_calls_mujoco(tmp_path):
    """Enforces README "No closed-loop claims yet": predict() must be pure
    inference, with no MuJoCo dependency anywhere in models/policy.py."""
    import models.policy as policy_module

    assert not hasattr(policy_module, "mujoco")
    source = inspect.getsource(policy_module)
    assert "import mujoco" not in source


def test_predict_accepts_different_instructions(tmp_path):
    policy = make_policy(tmp_path)
    observation = make_observation()
    for instruction in ["Pick up the red cube.", "Grasp the red cube.", "Lift the red cube."]:
        action = policy.predict(observation, instruction)
        assert isinstance(action, RobotAction)
