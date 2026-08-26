"""Tests for dagger/corrective_expert.py -- the stateless DAgger labeler.

Deliberately MuJoCo-free for the phase-inference unit tests (mirrors
tests/test_scripted_controller.py's synthetic-state style); a small number
of MuJoCo-backed tests validate that the labeler produces finite actions on
perturbed physical states and can drive a real closed-loop recovery.
"""

import numpy as np
import pytest

from control.scripted_controller import GRASP_HEIGHT_OFFSET, TOP_DOWN_GRASP_QUATERNION, Stage
from control.success import sustained_lift_success
from dagger.corrective_expert import compute_corrective_action, infer_phase
from observations.robot_state import RobotState
from simulation.environment import SimulationEnvironment

TOY_JACOBIAN_POS = np.hstack([np.eye(3), np.zeros((3, 4))])
TOY_JACOBIAN_ROT = np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))])
TOY_JACOBIAN = (TOY_JACOBIAN_POS, TOY_JACOBIAN_ROT)


def make_robot_state(
    end_effector_position,
    end_effector_orientation=None,
    gripper_position=None,
    joint_positions=None,
) -> RobotState:
    return RobotState(
        joint_positions=np.zeros(7) if joint_positions is None else joint_positions,
        joint_velocities=np.zeros(7),
        end_effector_position=np.asarray(end_effector_position, dtype=np.float64),
        end_effector_orientation=(
            TOP_DOWN_GRASP_QUATERNION.copy() if end_effector_orientation is None else end_effector_orientation
        ),
        gripper_position=np.array([0.04, 0.04]) if gripper_position is None else gripper_position,
    )


# --- Stateless phase inference (no MuJoCo) --------------------------------


def test_far_above_cube_infers_above_cube_phase():
    cube_position = np.array([0.5, 0.0, 0.02])
    state = make_robot_state(end_effector_position=cube_position + np.array([0.1, 0.05, 0.3]))
    assert infer_phase(state, cube_position, initial_cube_z=0.02) is Stage.ABOVE_CUBE


def test_aligned_but_high_infers_descend_phase():
    cube_position = np.array([0.5, 0.0, 0.02])
    state = make_robot_state(end_effector_position=cube_position + np.array([0.0, 0.0, 0.08]))
    assert infer_phase(state, cube_position, initial_cube_z=0.02) is Stage.DESCEND


def test_aligned_low_gripper_open_infers_close_gripper_phase():
    cube_position = np.array([0.5, 0.0, 0.02])
    state = make_robot_state(
        end_effector_position=cube_position + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET]),
        gripper_position=np.array([0.04, 0.04]),
    )
    assert infer_phase(state, cube_position, initial_cube_z=0.02) is Stage.CLOSE_GRIPPER


def test_closed_and_high_infers_lift_phase_regardless_of_xy_drift():
    """Once closed and elevated, a secured cube moves WITH the gripper, so
    xy drift alone must not bounce the phase back to ABOVE_CUBE."""
    cube_position = np.array([0.5, 0.0, 0.06])  # cube has risen with the gripper
    state = make_robot_state(
        end_effector_position=cube_position + np.array([0.05, 0.0, 0.05]),  # some xy drift
        gripper_position=np.array([0.02, 0.02]),
    )
    assert infer_phase(state, cube_position, initial_cube_z=0.02) is Stage.LIFT


def test_closed_but_far_and_low_infers_above_cube_not_lift():
    """A closed-empty gripper far from the cube should reopen and
    re-approach, not attempt to 'lift' nothing."""
    cube_position = np.array([0.5, 0.0, 0.02])
    state = make_robot_state(
        end_effector_position=cube_position + np.array([0.2, 0.0, 0.0]),
        gripper_position=np.array([0.005, 0.005]),
    )
    assert infer_phase(state, cube_position, initial_cube_z=0.02) is Stage.ABOVE_CUBE


# --- compute_corrective_action: finiteness on perturbed states -----------


@pytest.mark.parametrize(
    "eef_offset,gripper_position",
    [
        pytest.param((0.10, 0.0, 0.15), (0.04, 0.04), id="slightly_off_approach_trajectory"),
        pytest.param((0.06, -0.04, 0.10), (0.04, 0.04), id="slightly_beside_cube"),
        pytest.param((0.0, 0.0, -0.05), (0.04, 0.04), id="too_low"),
        pytest.param((0.0, 0.0, GRASP_HEIGHT_OFFSET), (0.028, 0.028), id="gripper_partly_closed"),
        pytest.param((0.0, 0.0, 0.08), (0.02, 0.02), id="post_contact_elevated"),
        pytest.param((0.3, 0.3, 0.4), (0.04, 0.04), id="far_off_workspace_corner"),
    ],
)
def test_corrective_action_is_finite_on_perturbed_states(eef_offset, gripper_position):
    cube_position = np.array([0.5, 0.0, 0.02])
    state = make_robot_state(
        end_effector_position=cube_position + np.array(eef_offset),
        gripper_position=np.array(gripper_position),
    )
    action, phase = compute_corrective_action(state, TOY_JACOBIAN, cube_position, initial_cube_z=0.02)

    assert np.all(np.isfinite(action.joint_targets))
    assert np.isfinite(action.gripper_target)
    assert 0.0 <= action.gripper_target <= 1.0
    assert isinstance(phase, Stage)


def test_corrective_action_cube_displaced_from_expected_still_finite():
    """The privileged cube_position always reflects the TRUE current cube
    location; a labeler call with the eef far from a *displaced* cube must
    still produce a finite, sensible (re-approach) action."""
    true_cube_position = np.array([0.55, -0.02, 0.02])
    eef_near_stale_position = np.array([0.5, 0.0, 0.12])
    state = make_robot_state(end_effector_position=eef_near_stale_position)

    action, phase = compute_corrective_action(state, TOY_JACOBIAN, true_cube_position, initial_cube_z=0.02)

    assert np.all(np.isfinite(action.joint_targets))
    assert phase is Stage.ABOVE_CUBE


# --- MuJoCo-backed: closed-loop recovery ----------------------------------


def test_corrective_expert_recovers_cube_from_home_in_closed_loop():
    """Driving the corrective expert itself in a closed loop (optional per
    README "Validate corrective expert") from the unperturbed HOME state
    must reach real task success -- confirms the labeler's phase logic is
    sound, not just locally finite."""
    with SimulationEnvironment() as env:
        env.reset()
        cube_position = env.get_object_position("red_cube")
        initial_cube_z = float(cube_position[2])
        cube_z_history = []
        success = False
        for _ in range(400):
            robot_state = env.get_robot_state()
            cube_position = env.get_object_position("red_cube")
            jacobian = env.get_end_effector_jacobian()
            action, _ = compute_corrective_action(robot_state, jacobian, cube_position, initial_cube_z)
            env.step(action)
            cube_z_history.append(float(env.get_object_position("red_cube")[2]))
            if sustained_lift_success(cube_z_history, initial_cube_z):
                success = True
                break
        assert success


def test_corrective_expert_recovers_from_xy_perturbed_cube():
    """Same closed-loop check, but with a +/-3cm cube offset (the DAgger
    Round 1 collection distribution) -- the labeler must remain reliable
    across the actual collection distribution, not just the nominal case."""
    with SimulationEnvironment() as env:
        env.reset()
        cube_position = env.get_object_position("red_cube")
        cube_position[0] += 0.02
        cube_position[1] -= 0.015
        env.set_object_position("red_cube", cube_position)
        cube_position = env.get_object_position("red_cube")
        initial_cube_z = float(cube_position[2])
        cube_z_history = []
        success = False
        for _ in range(400):
            robot_state = env.get_robot_state()
            cube_position = env.get_object_position("red_cube")
            jacobian = env.get_end_effector_jacobian()
            action, _ = compute_corrective_action(robot_state, jacobian, cube_position, initial_cube_z)
            env.step(action)
            cube_z_history.append(float(env.get_object_position("red_cube")[2]))
            if sustained_lift_success(cube_z_history, initial_cube_z):
                success = True
                break
        assert success


def test_corrective_expert_module_never_imports_mujoco():
    import inspect

    import dagger.corrective_expert as expert_module

    assert not hasattr(expert_module, "mujoco")
    assert "import mujoco" not in inspect.getsource(expert_module)
