"""Tests for orientation math, pose IK, and the ScriptedController state machine.

These are deliberately MuJoCo-free: state-machine tests construct
RobotState objects directly and drive the controller with a synthetic
Jacobian pair, so the logic can be verified in isolation from physics/
contacts. See tests/test_simulation.py for the MuJoCo-backed integration
tests (real Jacobians, real rendering).
"""

import numpy as np

from control.kinematics import (
    damped_least_squares_ik,
    orientation_error,
    quaternion_multiply,
    rotation_vector_degrees,
    solve_pose_ik,
)
from control.scripted_controller import (
    APPROACH_HEIGHT,
    GRASP_HEIGHT_OFFSET,
    GRIPPER_CLOSE_STEPS,
    TOP_DOWN_GRASP_QUATERNION,
    ScriptedController,
    Stage,
)
from observations.robot_state import RobotState

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# Toy Jacobian pair: joints 0:3 map 1:1 onto Cartesian xyz, joints 3:6 map
# 1:1 onto an angular-velocity command, joint 6 is unused (a redundant DoF,
# like the real 7-DoF arm has past position+orientation's 6 constraints).
TOY_JACOBIAN_POS = np.hstack([np.eye(3), np.zeros((3, 4))])
TOY_JACOBIAN_ROT = np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))])
TOY_JACOBIAN = (TOY_JACOBIAN_POS, TOY_JACOBIAN_ROT)


def make_robot_state(
    joint_positions: np.ndarray,
    end_effector_position: np.ndarray = None,
    end_effector_orientation: np.ndarray = None,
    gripper_position: np.ndarray = None,
) -> RobotState:
    return RobotState(
        joint_positions=joint_positions,
        joint_velocities=np.zeros(7),
        end_effector_position=(
            joint_positions[:3].copy() if end_effector_position is None else end_effector_position
        ),
        end_effector_orientation=(
            TOP_DOWN_GRASP_QUATERNION.copy()
            if end_effector_orientation is None
            else end_effector_orientation
        ),
        gripper_position=np.array([0.04, 0.04]) if gripper_position is None else gripper_position,
    )


def integrate_orientation(current_quat: np.ndarray, rotation_vector: np.ndarray) -> np.ndarray:
    """Compose a small rotation-vector step onto a quaternion (test helper only)."""
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return current_quat
    axis = rotation_vector / angle
    delta_quat = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])
    new_quat = quaternion_multiply(delta_quat, current_quat)
    return new_quat / np.linalg.norm(new_quat)


# --- Orientation error math (control/kinematics.py) --------------------------


def test_identical_quaternion_gives_near_zero_error():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    error = orientation_error(q, q)
    assert error.shape == (3,)
    assert np.all(np.isfinite(error))
    assert np.linalg.norm(error) < 1e-9


def test_negated_quaternion_is_the_same_orientation():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    error_same_sign = orientation_error(q, q)
    error_flipped_sign = orientation_error(q, -q)
    assert np.linalg.norm(error_same_sign) < 1e-9
    assert np.linalg.norm(error_flipped_sign) < 1e-9


def test_known_small_rotation_gives_sensible_error():
    # 90-degree rotation about the world x axis.
    angle = np.pi / 2
    target = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    error = orientation_error(IDENTITY_QUAT, target)
    assert error.shape == (3,)
    assert np.all(np.isfinite(error))
    assert np.isclose(np.linalg.norm(error), angle, atol=1e-6)
    # Rotation axis should point along +x.
    assert np.allclose(error / np.linalg.norm(error), [1.0, 0.0, 0.0], atol=1e-6)


def test_rotation_vector_degrees_matches_radians():
    error = np.array([np.pi / 4, 0.0, 0.0])
    assert np.isclose(rotation_vector_degrees(error), 45.0)


# --- Pose IK (control/kinematics.py) ------------------------------------------


def test_pose_ik_output_shape_and_finiteness():
    rng = np.random.default_rng(1)
    jp = rng.normal(size=(3, 7))
    jr = rng.normal(size=(3, 7))
    dq = solve_pose_ik(
        position_error=np.array([0.1, -0.05, 0.02]),
        orientation_error_vector=np.array([0.05, 0.0, -0.02]),
        jacobian_position=jp,
        jacobian_rotation=jr,
    )
    assert dq.shape == (7,)
    assert np.all(np.isfinite(dq))


def test_pose_ik_zero_error_gives_near_zero_update():
    dq = solve_pose_ik(
        position_error=np.zeros(3),
        orientation_error_vector=np.zeros(3),
        jacobian_position=TOY_JACOBIAN_POS,
        jacobian_rotation=TOY_JACOBIAN_ROT,
    )
    assert np.allclose(dq, 0.0, atol=1e-9)


def test_pose_ik_respects_max_joint_delta():
    dq = solve_pose_ik(
        position_error=np.array([10.0, 10.0, 10.0]),
        orientation_error_vector=np.array([10.0, 10.0, 10.0]),
        jacobian_position=TOY_JACOBIAN_POS,
        jacobian_rotation=TOY_JACOBIAN_ROT,
        max_joint_delta=0.1,
    )
    assert np.all(np.abs(dq) <= 0.1 + 1e-9)


def test_pose_ik_converges_on_toy_system():
    """Repeated pose-IK steps should drive both position and orientation to target."""
    q = np.zeros(7)
    orientation = IDENTITY_QUAT.copy()
    target_position = np.array([0.2, -0.1, 0.15])
    target_orientation = TOP_DOWN_GRASP_QUATERNION.copy()

    for _ in range(200):
        position_error = target_position - q[:3]
        rotation_error = orientation_error(orientation, target_orientation)
        dq = solve_pose_ik(
            position_error,
            rotation_error,
            TOY_JACOBIAN_POS,
            TOY_JACOBIAN_ROT,
            orientation_weight=1.0,
            damping=0.05,
            max_joint_delta=0.1,
        )
        q = q + dq
        orientation = integrate_orientation(orientation, dq[3:6])

    assert np.linalg.norm(target_position - q[:3]) < 1e-2
    assert rotation_vector_degrees(orientation_error(orientation, target_orientation)) < 5.0


def test_position_only_ik_still_works_unchanged():
    """Step 2's position-only solver must remain intact (section 9 requirement)."""
    dq = damped_least_squares_ik(TOY_JACOBIAN_POS, np.array([0.1, 0.0, 0.0]))
    assert dq.shape == (7,)
    assert np.all(np.isfinite(dq))


# --- ScriptedController state machine -----------------------------------------


def test_above_cube_does_not_transition_on_position_alone():
    """Position converged but orientation is wrong -> must NOT advance to DESCEND."""
    controller = ScriptedController(home_joint_positions=np.zeros(7))
    controller.stage = Stage.ABOVE_CUBE
    cube_position = np.array([1.0, 0.0, 0.0])
    target_position = cube_position + np.array([0.0, 0.0, APPROACH_HEIGHT])

    state = make_robot_state(
        joint_positions=np.zeros(7),
        end_effector_position=target_position,  # exact position match
        end_effector_orientation=IDENTITY_QUAT,  # very wrong orientation
    )

    controller.compute_action(state, TOY_JACOBIAN, cube_position)

    assert controller.stage is Stage.ABOVE_CUBE


def test_above_cube_transitions_when_position_and_orientation_converge():
    controller = ScriptedController(home_joint_positions=np.zeros(7))
    controller.stage = Stage.ABOVE_CUBE
    cube_position = np.array([1.0, 0.0, 0.0])
    target_position = cube_position + np.array([0.0, 0.0, APPROACH_HEIGHT])

    state = make_robot_state(
        joint_positions=np.zeros(7),
        end_effector_position=target_position,
        end_effector_orientation=TOP_DOWN_GRASP_QUATERNION,
    )

    controller.compute_action(state, TOY_JACOBIAN, cube_position)

    assert controller.stage is Stage.DESCEND


def test_descend_targets_grasp_height_with_top_down_orientation():
    controller = ScriptedController(home_joint_positions=np.zeros(7))
    controller.stage = Stage.DESCEND
    cube_position = np.array([1.0, 0.0, 0.0])
    state = make_robot_state(
        joint_positions=np.zeros(7),
        end_effector_position=np.array([1.0, 0.0, 0.5]),
        end_effector_orientation=TOP_DOWN_GRASP_QUATERNION,
    )

    controller.compute_action(state, TOY_JACOBIAN, cube_position)

    expected_target = cube_position + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
    assert np.allclose(controller.last_target_position, expected_target)


def test_descend_does_not_transition_if_orientation_drifts():
    controller = ScriptedController(home_joint_positions=np.zeros(7))
    controller.stage = Stage.DESCEND
    cube_position = np.array([1.0, 0.0, 0.0])
    target_position = cube_position + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])

    state = make_robot_state(
        joint_positions=np.zeros(7),
        end_effector_position=target_position,
        end_effector_orientation=IDENTITY_QUAT,  # drifted away from top-down
    )

    controller.compute_action(state, TOY_JACOBIAN, cube_position)

    assert controller.stage is Stage.DESCEND


def test_close_gripper_holds_grasp_pose_and_advances_after_fixed_steps():
    controller = ScriptedController(home_joint_positions=np.zeros(7))
    controller.stage = Stage.CLOSE_GRIPPER
    controller._gripper_close_counter = 0
    controller._grasp_target = np.array([1.0, 0.0, 0.045])
    state = make_robot_state(
        joint_positions=np.zeros(7),
        end_effector_position=np.array([1.0, 0.0, 0.045]),
        end_effector_orientation=TOP_DOWN_GRASP_QUATERNION,
    )
    cube_position = np.array([1.0, 0.0, 0.0])

    for _ in range(GRIPPER_CLOSE_STEPS - 1):
        action = controller.compute_action(state, TOY_JACOBIAN, cube_position)
        assert controller.stage is Stage.CLOSE_GRIPPER
        assert action.gripper_target == 0.0
        # Pose is actively re-solved toward the grasp target every tick, not
        # frozen -- with zero error the commanded joint targets should stay
        # at the current joint positions.
        assert np.allclose(action.joint_targets, state.joint_positions, atol=1e-6)

    controller.compute_action(state, TOY_JACOBIAN, cube_position)
    assert controller.stage is Stage.LIFT


def test_lift_preserves_orientation_and_requires_it_to_finish():
    controller = ScriptedController(home_joint_positions=np.zeros(7))
    controller.stage = Stage.LIFT
    controller._lift_target = np.array([1.0, 0.0, 0.2])

    state_wrong_orientation = make_robot_state(
        joint_positions=np.zeros(7),
        end_effector_position=np.array([1.0, 0.0, 0.2]),  # position already there
        end_effector_orientation=IDENTITY_QUAT,
    )
    controller.compute_action(state_wrong_orientation, TOY_JACOBIAN, np.array([1.0, 0.0, 0.045]))
    assert controller.stage is Stage.LIFT  # must not finish with bad orientation

    state_correct = make_robot_state(
        joint_positions=np.zeros(7),
        end_effector_position=np.array([1.0, 0.0, 0.2]),
        end_effector_orientation=TOP_DOWN_GRASP_QUATERNION,
    )
    controller.compute_action(state_correct, TOY_JACOBIAN, np.array([1.0, 0.0, 0.045]))
    assert controller.stage is Stage.DONE


def test_done_stage_holds_joint_positions_and_implies_no_success_metric():
    """Reaching DONE is controller completion, not task success (section 16)."""
    controller = ScriptedController(home_joint_positions=np.zeros(7))
    controller.stage = Stage.DONE
    state = make_robot_state(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))

    action = controller.compute_action(state, TOY_JACOBIAN, cube_position=np.zeros(3))

    assert controller.done is True
    assert np.allclose(action.joint_targets, state.joint_positions)
    assert action.gripper_target == 0.0
    # `done` is a state-machine flag only -- it carries no lift/grasp outcome.
    assert not hasattr(controller, "task_success")
