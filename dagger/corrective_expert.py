"""Corrective (stateless) expert labeler for DAgger data collection.

``control.scripted_controller.ScriptedController`` is unsuitable for
labeling arbitrary model-induced states: its ``stage`` field only advances
forward (HOME -> ABOVE_CUBE -> ... -> DONE) under the assumption that IT
generated every previous action. During DAgger collection the Temporal VLA
drives the rollout instead, so the physical state at any tick may not match
what the stateful controller's internal stage assumes -- e.g. the eef could
be near the cube while the controller (if reused naively) still thinks it's
in ABOVE_CUBE holding a stale ``_grasp_target`` from a earlier attempt.

``infer_phase`` / ``compute_corrective_action`` solve this by re-deriving a
reasonable phase from the CURRENT physical state every single call --
nothing here persists across ticks. This makes the labeler robust to being
invoked on any off-expert state, at the cost of not exactly reproducing
``ScriptedController``'s original hold/settle timing (see
``GRIPPER_CLOSE_STEPS`` in ``scripted_controller.py``) -- that timing exists
for closed-loop physical stability during data GENERATION and isn't needed
for one-shot corrective LABELING.

Teacher-only: this module (like ``ScriptedController``) is allowed
privileged access to the cube's ground-truth position. It must never be
imported on the learned policy's execution path (see
``tests/test_dagger_expert_not_executed.py`` /
``tests/test_dagger_no_privileged_runtime.py``).
"""

import numpy as np

from control.action import RobotAction
from control.kinematics import orientation_error, solve_pose_ik
from control.scripted_controller import (
    APPROACH_HEIGHT,
    GRASP_HEIGHT_OFFSET,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    IK_DAMPING,
    IK_MAX_JOINT_DELTA,
    LIFT_HEIGHT,
    LIFT_MAX_JOINT_DELTA,
    ORIENTATION_WEIGHT,
    TOP_DOWN_GRASP_QUATERNION,
    Stage,
)
from observations.robot_state import RobotState

# Calibrated against the Step 3 expert dataset AND direct measurement of
# the gripper actuator's closing dynamics (see module docstring / Step 8
# report): per-finger opening starts at ~0.040m (open) and, once the
# gripper target flips to closed, drops through ~0.030m within 1-2 ticks --
# well before the fingers have actually settled onto the cube -- and only
# reaches its final resting value (~0.020-0.021m, closed on the cube's
# 0.02m half-width) a few ticks later. Using the higher 0.03 threshold
# triggered "gripper_closed" (and therefore LIFT) mid-transient, before
# real grip force had built up, letting the cube slip out during lift
# (verified empirically: closed-loop success only appeared once this was
# tightened close to the settled value).
GRIPPER_CLOSED_OPENING_THRESHOLD = 0.022  # meters, per-finger

# xy alignment required before considering the arm "positioned over the
# cube" (measured expert DESCEND/CLOSE_GRIPPER xy error stays under ~0.01m;
# ABOVE_CUBE ranges up to ~0.09m while still approaching).
APPROACH_XY_TOLERANCE = 0.025  # meters

# Height above the cube below which the arm is considered "descended enough"
# to attempt closing (measured expert CLOSE_GRIPPER z_above stays under
# ~0.03m; DESCEND covers roughly 0.03-0.12m).
CLOSE_Z_TOLERANCE = 0.035  # meters

def infer_phase(
    robot_state: RobotState,
    cube_position: np.ndarray,
    initial_cube_z: float,
) -> Stage:
    """Derive a corrective phase from the CURRENT physical state alone.

    Stateless by construction -- see module docstring. Only the four
    "active" ``ScriptedController`` stages are ever returned; HOME/DONE are
    not meaningful corrective labels for an arbitrary off-expert state.

    Deliberately does NOT use live cube-height gain to decide "grasp
    secured": a momentary bump from finger/cube contact during CLOSING can
    lift the cube a few millimeters before the fingers have actually
    settled around it (the same false-positive ``control.success`` guards
    against with a sustained-ticks requirement -- see its docstring). Using
    that signal here caused the phase to flap CLOSE_GRIPPER<->LIFT<->DESCEND
    on a premature bump and never complete a grasp (verified empirically).
    Instead, "secured" is read purely from gripper closure + Cartesian
    position, which is stable under momentary contact noise.
    """
    eef_position = robot_state.end_effector_position
    xy_distance = float(np.linalg.norm(eef_position[:2] - cube_position[:2]))
    z_above = float(eef_position[2] - cube_position[2])
    gripper_opening = float(np.mean(robot_state.gripper_position))
    gripper_closed = gripper_opening < GRIPPER_CLOSED_OPENING_THRESHOLD

    if gripper_closed:
        # Already closed and above the grasp height -> assume a grasp is in
        # progress/complete and keep lifting, regardless of xy drift (a
        # secured cube moves with the gripper, so xy naturally tracks it).
        if z_above > CLOSE_Z_TOLERANCE:
            return Stage.LIFT
        # Closed, still down at grasp height, and well-aligned -> settling
        # onto the grasp; hold/continue toward LIFT.
        if xy_distance <= APPROACH_XY_TOLERANCE:
            return Stage.LIFT
        # Closed but far from the cube and still low: almost certainly a
        # closed-empty gripper somewhere it shouldn't be -- reopen and
        # re-approach rather than trying to "lift" nothing.
        return Stage.ABOVE_CUBE

    if xy_distance > APPROACH_XY_TOLERANCE:
        return Stage.ABOVE_CUBE
    if z_above > CLOSE_Z_TOLERANCE:
        return Stage.DESCEND
    return Stage.CLOSE_GRIPPER


def _pose_action(
    robot_state: RobotState,
    jacobian: tuple,
    target_position: np.ndarray,
    gripper_target: float,
    max_joint_delta: float = IK_MAX_JOINT_DELTA,
) -> RobotAction:
    """One damped-least-squares pose-IK step toward ``target_position``,
    holding the top-down grasp orientation -- same math as
    ``ScriptedController._pose_ik_step``, without any stage-transition
    bookkeeping."""
    jacobian_position, jacobian_rotation = jacobian
    position_error = target_position - robot_state.end_effector_position
    rotation_error = orientation_error(robot_state.end_effector_orientation, TOP_DOWN_GRASP_QUATERNION)

    dq = solve_pose_ik(
        position_error, rotation_error, jacobian_position, jacobian_rotation,
        orientation_weight=ORIENTATION_WEIGHT, damping=IK_DAMPING, max_joint_delta=max_joint_delta,
    )
    joint_targets = robot_state.joint_positions + dq
    return RobotAction(joint_targets=joint_targets, gripper_target=gripper_target)


def compute_corrective_action(
    robot_state: RobotState,
    jacobian: tuple,
    cube_position: np.ndarray,
    initial_cube_z: float,
) -> tuple:
    """Teacher-only corrective label for the current state.

    Returns ``(action, phase)``. Uses privileged ``cube_position``
    (ground-truth) exactly as ``ScriptedController`` does for data
    GENERATION -- the returned action must never be executed by the
    learned policy, only stored as a supervised target (README "Critical
    runtime distinction").
    """
    phase = infer_phase(robot_state, cube_position, initial_cube_z)
    cube_position = np.asarray(cube_position, dtype=np.float64)

    if phase is Stage.ABOVE_CUBE:
        target = cube_position + np.array([0.0, 0.0, APPROACH_HEIGHT])
        action = _pose_action(robot_state, jacobian, target, GRIPPER_OPEN)
    elif phase is Stage.DESCEND:
        target = cube_position + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
        action = _pose_action(robot_state, jacobian, target, GRIPPER_OPEN)
    elif phase is Stage.CLOSE_GRIPPER:
        target = cube_position + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
        action = _pose_action(robot_state, jacobian, target, GRIPPER_CLOSED)
    else:  # Stage.LIFT
        # Fixed absolute height target (anchored to the EPISODE's initial
        # cube z, not the live/current cube z) -- using the live cube z
        # here would create positive feedback once grasped (cube rises with
        # the gripper, pushing the target up by the same amount every
        # tick). xy still tracks the live cube position to correct for any
        # grasp drift.
        target = np.array(
            [cube_position[0], cube_position[1], initial_cube_z + GRASP_HEIGHT_OFFSET + LIFT_HEIGHT]
        )
        action = _pose_action(robot_state, jacobian, target, GRIPPER_CLOSED, max_joint_delta=LIFT_MAX_JOINT_DELTA)

    return action, phase
