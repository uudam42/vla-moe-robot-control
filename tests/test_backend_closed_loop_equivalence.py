"""README "Functional equivalence test" (partial, pre-ROS2): the SAME
policy, same seed, same checkpoint must produce IDENTICAL behavior whether
driven directly through ``SimulationEnvironment``
(``evaluation.closed_loop.run_closed_loop_episode``) or through
``MuJoCoBackend`` (``robot_backend.backend_closed_loop.run_closed_loop_episode_via_backend``).
This is the "refactor one direct runner to use RobotBackend; verify no
behavior regression" step, proven by direct comparison rather than by
inspection.
"""

import numpy as np
import torch

from evaluation.closed_loop import run_closed_loop_episode
from models.temporal_policy import TemporalDenseVLAPolicy
from robot_backend.backend_closed_loop import run_closed_loop_episode_via_backend
from robot_backend.mujoco_backend import MuJoCoBackend
from simulation.environment import SimulationEnvironment


def test_backend_and_direct_paths_produce_identical_rollouts(tiny_temporal_vla_checkpoint):
    cube_xy_offset = np.array([0.01, -0.015])
    instruction = "Pick up the red cube."

    policy_direct = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    with SimulationEnvironment() as env:
        direct_result = run_closed_loop_episode(
            env, policy_direct, instruction, cube_xy_offset=cube_xy_offset, max_steps=15
        )

    policy_backend = TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))
    with MuJoCoBackend() as backend:
        backend_result = run_closed_loop_episode_via_backend(
            backend, policy_backend, instruction, cube_xy_offset=cube_xy_offset, max_steps=15
        )

    assert direct_result.episode_length == backend_result.episode_length
    assert direct_result.success == backend_result.success
    assert np.allclose(direct_result.initial_cube_position, backend_result.initial_cube_position)
    assert np.allclose(direct_result.final_cube_position, backend_result.final_cube_position)
    assert np.allclose(direct_result.gripper_probabilities, backend_result.gripper_probabilities, atol=1e-6)
    assert np.isclose(direct_result.cube_lift_delta, backend_result.cube_lift_delta, atol=1e-9)


def test_backend_path_never_imports_mujoco_outside_the_backend_module():
    import inspect

    import robot_backend.backend_closed_loop as backend_closed_loop_module

    assert not hasattr(backend_closed_loop_module, "mujoco")
    assert "import mujoco" not in inspect.getsource(backend_closed_loop_module)
