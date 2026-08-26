"""Tests for models/moe_policy.py::MoEVLAPolicy -- the MoE inference adapter."""

import inspect

import numpy as np
import torch

from control.action import RobotAction
from models.moe_policy import MoEVLAPolicy
from observations.observation import Observation


def make_observation() -> Observation:
    rgb = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    state = np.random.default_rng(1).normal(size=23)
    return Observation(rgb=rgb, state=state, timestamp=1.23)


def test_predict_returns_valid_robot_action(tiny_moe_vla_checkpoint):
    policy = MoEVLAPolicy.from_checkpoint(tiny_moe_vla_checkpoint, device=torch.device("cpu"))
    action = policy.predict(make_observation(), "Pick up the red cube.")

    assert isinstance(action, RobotAction)
    assert action.joint_targets.shape == (7,)
    assert np.all(np.isfinite(action.joint_targets))
    assert 0.0 <= action.gripper_target <= 1.0
    assert np.isfinite(action.gripper_target)


def test_predict_is_deterministic_in_eval_mode(tiny_moe_vla_checkpoint):
    policy = MoEVLAPolicy.from_checkpoint(tiny_moe_vla_checkpoint, device=torch.device("cpu"))
    observation = make_observation()

    action_1 = policy.predict(observation, "Pick up the red cube.")
    action_2 = policy.predict(observation, "Pick up the red cube.")

    assert np.allclose(action_1.joint_targets, action_2.joint_targets)
    assert action_1.gripper_target == action_2.gripper_target


def test_predict_with_routing_returns_action_and_diagnostics(tiny_moe_vla_checkpoint):
    policy = MoEVLAPolicy.from_checkpoint(tiny_moe_vla_checkpoint, device=torch.device("cpu"))
    action, layer_diagnostics = policy.predict_with_routing(make_observation(), "Pick up the red cube.")

    assert isinstance(action, RobotAction)
    assert set(layer_diagnostics.keys()) == {1, 3}
    for diagnostics in layer_diagnostics.values():
        assert diagnostics["routing_indices"].shape[1] == 4  # 4 fusion tokens


def test_predict_and_predict_with_routing_agree_on_action(tiny_moe_vla_checkpoint):
    policy = MoEVLAPolicy.from_checkpoint(tiny_moe_vla_checkpoint, device=torch.device("cpu"))
    observation = make_observation()

    plain_action = policy.predict(observation, "Pick up the red cube.")
    routed_action, _ = policy.predict_with_routing(observation, "Pick up the red cube.")

    assert np.allclose(plain_action.joint_targets, routed_action.joint_targets)
    assert plain_action.gripper_target == routed_action.gripper_target


def test_predict_never_imports_or_calls_mujoco(tiny_moe_vla_checkpoint):
    import models.moe_policy as policy_module

    assert not hasattr(policy_module, "mujoco")
    source = inspect.getsource(policy_module)
    assert "import mujoco" not in source
