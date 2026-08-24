"""Integration tests for SimulationEnvironment (require MuJoCo rendering)."""

from pathlib import Path

import numpy as np
import pytest

from control.action import RobotAction
from simulation.environment import SimulationEnvironment

SCENE_PATH = Path(__file__).parent.parent / "simulation" / "scene.xml"


@pytest.fixture
def env():
    environment = SimulationEnvironment(scene_path=SCENE_PATH, width=320, height=240)
    yield environment
    environment.close()


def test_missing_scene_raises():
    with pytest.raises(FileNotFoundError):
        SimulationEnvironment(scene_path=Path("does_not_exist.xml"))


def test_unknown_camera_raises():
    with pytest.raises(ValueError):
        SimulationEnvironment(scene_path=SCENE_PATH, camera_name="nonexistent_camera")


def test_render_dimensions_and_variance(env):
    obs = env.get_observation()
    assert obs.rgb.shape == (240, 320, 3)
    assert obs.rgb.dtype == np.uint8
    assert obs.rgb.size > 0
    # A meaningfully lit scene should not render as a flat, uniform image.
    assert obs.rgb.std() > 1.0


def test_robot_state_shapes_and_finiteness(env):
    state = env.get_robot_state()
    assert state.joint_positions.shape == (7,)
    assert state.joint_velocities.shape == (7,)
    assert state.end_effector_position.shape == (3,)
    assert state.end_effector_orientation.shape == (4,)
    assert state.gripper_position.shape == (2,)
    assert np.all(np.isfinite(state.joint_positions))
    assert np.all(np.isfinite(state.joint_velocities))
    assert np.all(np.isfinite(state.end_effector_position))


def test_observation_state_is_robot_proprioception(env):
    obs = env.get_observation()
    robot_state = env.get_robot_state()
    assert obs.state.shape == (23,)
    assert np.allclose(obs.state, robot_state.as_vector())


def test_step_changes_robot_state(env):
    state_1 = env.get_robot_state()
    nudge = np.zeros(7)
    nudge[0] = 0.3
    env.step(RobotAction(joint_targets=state_1.joint_positions + nudge, gripper_target=1.0))
    state_2 = env.get_robot_state()

    assert not np.allclose(state_1.joint_positions, state_2.joint_positions)


def test_step_rejects_non_robot_action(env):
    with pytest.raises(TypeError):
        env.step("not an action")


def test_reset_restores_initial_state(env):
    state_initial = env.get_robot_state()
    nudge = np.zeros(7)
    nudge[0] = 0.3
    env.step(RobotAction(joint_targets=state_initial.joint_positions + nudge, gripper_target=1.0))
    state_after_step = env.get_robot_state()
    assert not np.allclose(state_initial.joint_positions, state_after_step.joint_positions)

    env.reset()
    state_after_reset = env.get_robot_state()

    assert np.allclose(
        state_initial.joint_positions, state_after_reset.joint_positions, atol=1e-6
    )


def test_get_object_position_returns_cube_position(env):
    cube_position = env.get_object_position("red_cube")
    assert cube_position.shape == (3,)
    assert np.all(np.isfinite(cube_position))


def test_get_object_position_unknown_body_raises(env):
    with pytest.raises(ValueError):
        env.get_object_position("nonexistent_body")


def test_jacobian_shape_and_finiteness(env):
    jacobian = env.get_jacobian()
    assert jacobian.shape == (3, 7)
    assert np.all(np.isfinite(jacobian))


def test_end_effector_jacobian_shape_and_finiteness(env):
    jacobian_position, jacobian_rotation = env.get_end_effector_jacobian()
    assert jacobian_position.shape == (3, 7)
    assert jacobian_rotation.shape == (3, 7)
    assert np.all(np.isfinite(jacobian_position))
    assert np.all(np.isfinite(jacobian_rotation))
    # get_jacobian() is the translational half of the same computation.
    assert np.allclose(jacobian_position, env.get_jacobian())


def test_set_object_position_moves_the_cube(env):
    new_position = np.array([0.6, 0.2, 0.5])
    env.set_object_position("red_cube", new_position)
    assert np.allclose(env.get_object_position("red_cube"), new_position)


def test_set_object_position_unknown_body_raises(env):
    with pytest.raises(ValueError):
        env.set_object_position("nonexistent_body", np.zeros(3))


def test_observation_state_unaffected_by_cube_position(env):
    """Observation.state must stay pure robot proprioception (Step 2.5 section 25).

    A future VLA trains on RGB + robot state + language, never simulator
    ground truth -- moving the cube must not change Observation.state at
    all, since the robot itself hasn't moved.
    """
    state_before = env.get_observation().state.copy()
    env.set_object_position("red_cube", np.array([0.9, -0.4, 0.6]))
    state_after = env.get_observation().state

    assert np.array_equal(state_before, state_after)
