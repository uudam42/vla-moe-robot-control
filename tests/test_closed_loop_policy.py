"""Policy runtime test: a real MuJoCo Observation through DenseVLAPolicy.predict()."""

import numpy as np
import torch

from control.action import RobotAction
from models.policy import DenseVLAPolicy
from simulation.environment import SimulationEnvironment
from training.config import resolve_device


def test_predict_on_real_observation_returns_valid_action(tiny_vla_checkpoint):
    policy = DenseVLAPolicy.from_checkpoint(tiny_vla_checkpoint, device=torch.device("cpu"))

    with SimulationEnvironment() as env:
        env.reset()
        observation = env.get_observation()
        action = policy.predict(observation=observation, instruction="Pick up the red cube.")

    assert isinstance(action, RobotAction)
    assert action.joint_targets.shape == (7,)
    assert np.all(np.isfinite(action.joint_targets))
    assert np.isfinite(action.gripper_target)
    assert 0.0 <= action.gripper_target <= 1.0


def test_policy_model_is_in_eval_mode_after_loading(tiny_vla_checkpoint):
    policy = DenseVLAPolicy.from_checkpoint(tiny_vla_checkpoint, device=torch.device("cpu"))
    assert policy.model.training is False


def test_device_resolution_prefers_available_accelerator():
    device = resolve_device(None)
    assert device.type in ("cuda", "mps", "cpu")
    if torch.backends.mps.is_available():
        assert device.type in ("cuda", "mps")
