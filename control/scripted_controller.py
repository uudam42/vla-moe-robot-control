"""Deterministic scripted pick-sequence controller with pose-controlled IK.

``ScriptedController`` is a "data-generation expert": it is allowed
privileged access to the cube's ground-truth position (passed in by the
caller), which a future learned VLA will not have. Both must produce the
same :class:`~control.action.RobotAction` type, so swapping this controller
for a VLA model requires no change to the simulation/control boundary.

This module never touches MjModel/MjData/renderer or MuJoCo IDs -- it
consumes plain :class:`~observations.robot_state.RobotState`, a
``(J_pos, J_rot)`` Jacobian pair, and a target position, and returns a
RobotAction.

Step 2.5 change: the controller now drives full 6D pose (position +
orientation) rather than position only. Step 2's position-only IK
(``damped_least_squares_ik``) let the wrist orientation drift during
descent, which made the gripper approach the cube at an inconsistent
angle and push it instead of enclosing it. Every Cartesian-tracking stage
(ABOVE_CUBE, DESCEND, CLOSE_GRIPPER, LIFT) now also holds
``TOP_DOWN_GRASP_QUATERNION``.
"""

import enum

import numpy as np

from control.action import RobotAction
from control.kinematics import orientation_error, rotation_vector_degrees, solve_pose_ik
from observations.robot_state import RobotState

POSITION_TOLERANCE = 0.02  # meters; Cartesian target considered "reached"
ORIENTATION_TOLERANCE = 0.15  # radians (~8.6deg); orientation considered "aligned"

APPROACH_HEIGHT = 0.10  # meters above cube center for the ABOVE_CUBE target
GRASP_HEIGHT_OFFSET = 0.012  # meters above cube center for the DESCEND target.
# Recalibrated in Step 2.5 (was 0.045 in Step 2). The Step 2 value was
# calibrated with position-only IK, where wrist orientation drifted during
# descent -- the table contact that "calibration" detected was some other,
# tilted part of the gripper grazing the table, not the fingertip pads.
# With orientation now held level, direct measurement (see NOTICE.txt /
# README) shows the fingertip pads sit only ~5-6mm below the end_effector
# site, not ~49mm. 0.012 was empirically verified (fixed cube, held
# top-down) to: (a) close on the cube around its mid-height -- gripper
# settles at ~0.020m per finger, matching the cube's 0.02m half-width,
# confirming actual contact rather than closing past it -- and (b) clear
# the table (no finger/table contact observed).
LIFT_HEIGHT = 0.15  # meters to raise above grasp height during LIFT
GRIPPER_CLOSE_STEPS = 70  # controller ticks to hold CLOSE_GRIPPER; also acts
# as a settle period (grasp pose held under active pose IK) before LIFT
# starts -- empirically, lifting immediately after closing let the cube
# slip back out; holding the closed grasp for longer first fixed that.
IK_DAMPING = 0.08
IK_MAX_JOINT_DELTA = 0.15  # radians per controller tick (all stages but LIFT)
LIFT_MAX_JOINT_DELTA = 0.03  # radians per controller tick during LIFT only;
# lifting at the normal speed was dynamically fast enough to shake the cube
# out of the grasp before the arm reached the lift target. A slower lift
# (this alone, with the settle period above) reliably kept the cube in
# the grasp through a full lift in repeated manual trials.
HOME_TOLERANCE = 0.05  # radians; joint-space "close enough to home"

# Position (meters) vs. orientation (radians) live on different scales;
# this weight is the one place that trade-off is tuned. Conservative by
# design: strong enough that the wrist visibly holds its downward pose,
# gentle enough that it doesn't fight the position objective or destabilize
# the arm. Tune here only -- do not duplicate this value elsewhere.
ORIENTATION_WEIGHT = 0.3

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0

# Measured via env.reset() -> env.get_robot_state().end_effector_orientation
# at the HOME keyframe (see tests/test_scripted_controller.py). At HOME the
# end_effector site's local +z axis already points along world (0, 0, -1)
# -- i.e. the gripper faces straight down with fingers opposing along
# world x/y -- exactly the top-down grasp pose we want to hold through
# approach, descent, closing, and lift. Quaternion convention: (w, x, y, z),
# matching MuJoCo (see control/kinematics.py).
TOP_DOWN_GRASP_QUATERNION = np.array([0.0, 0.70710678, 0.70710678, 0.0])


class Stage(enum.Enum):
    HOME = "HOME"
    ABOVE_CUBE = "ABOVE_CUBE"
    DESCEND = "DESCEND"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    LIFT = "LIFT"
    DONE = "DONE"


class ScriptedController:
    """HOME -> ABOVE_CUBE -> DESCEND -> CLOSE_GRIPPER -> LIFT -> DONE.

    Args:
        home_joint_positions: The robot's arm joint positions at its home
            configuration, shape ``(7,)``. Typically read from
            ``env.get_robot_state().joint_positions`` right after
            ``env.reset()`` -- the controller itself has no notion of
            "home" beyond this caller-supplied reference.
        target_orientation: Orientation to hold during approach/descend/
            close/lift, shape ``(4,)``, ``(w, x, y, z)``. Defaults to
            :data:`TOP_DOWN_GRASP_QUATERNION`.
    """

    def __init__(
        self,
        home_joint_positions: np.ndarray,
        target_orientation: np.ndarray = None,
    ) -> None:
        self._home_joint_positions = np.asarray(home_joint_positions, dtype=np.float64)
        self._target_orientation = np.asarray(
            TOP_DOWN_GRASP_QUATERNION if target_orientation is None else target_orientation,
            dtype=np.float64,
        )
        self.stage = Stage.HOME
        self._gripper_close_counter = 0
        self._lift_target: np.ndarray = None
        self._grasp_target: np.ndarray = None

        # Diagnostics exposed for logging/demo use, updated every tick that
        # tracks a Cartesian pose target. None in HOME/DONE.
        self.last_target_position: np.ndarray = None
        self.last_position_error: float = None
        self.last_orientation_error_deg: float = None

    @property
    def done(self) -> bool:
        return self.stage is Stage.DONE

    def compute_action(
        self,
        robot_state: RobotState,
        jacobian: tuple,
        cube_position: np.ndarray,
    ) -> RobotAction:
        """Advance the state machine by one tick and produce a RobotAction.

        Args:
            robot_state: Current proprioception.
            jacobian: ``(J_pos, J_rot)`` pair, each shape ``(3, 7)`` (see
                ``SimulationEnvironment.get_end_effector_jacobian()``).
            cube_position: Ground-truth cube position, shape ``(3,)``
                (privileged; see ``SimulationEnvironment.get_object_position``).
        """
        if self.stage is Stage.HOME:
            return self._tick_home(robot_state)
        if self.stage is Stage.ABOVE_CUBE:
            target = cube_position + np.array([0.0, 0.0, APPROACH_HEIGHT])
            return self._tick_pose(robot_state, jacobian, target, GRIPPER_OPEN, Stage.DESCEND)
        if self.stage is Stage.DESCEND:
            target = cube_position + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
            return self._tick_descend(robot_state, jacobian, target)
        if self.stage is Stage.CLOSE_GRIPPER:
            return self._tick_close_gripper(robot_state, jacobian)
        if self.stage is Stage.LIFT:
            return self._tick_lift(robot_state, jacobian)
        return self._tick_done(robot_state)

    def _tick_home(self, robot_state: RobotState) -> RobotAction:
        self._clear_pose_diagnostics()
        error = self._home_joint_positions - robot_state.joint_positions
        if np.max(np.abs(error)) < HOME_TOLERANCE:
            self.stage = Stage.ABOVE_CUBE
        delta = np.clip(error, -IK_MAX_JOINT_DELTA, IK_MAX_JOINT_DELTA)
        joint_targets = robot_state.joint_positions + delta
        return RobotAction(joint_targets=joint_targets, gripper_target=GRIPPER_OPEN)

    def _tick_pose(
        self,
        robot_state: RobotState,
        jacobian: tuple,
        target_position: np.ndarray,
        gripper_target: float,
        next_stage: Stage,
    ) -> RobotAction:
        distance, orientation_distance, joint_targets = self._pose_ik_step(
            robot_state, jacobian, target_position
        )
        if distance < POSITION_TOLERANCE and orientation_distance < ORIENTATION_TOLERANCE:
            self.stage = next_stage
        return RobotAction(joint_targets=joint_targets, gripper_target=gripper_target)

    def _tick_descend(
        self, robot_state: RobotState, jacobian: tuple, target_position: np.ndarray
    ) -> RobotAction:
        distance, orientation_distance, joint_targets = self._pose_ik_step(
            robot_state, jacobian, target_position
        )
        if distance < POSITION_TOLERANCE and orientation_distance < ORIENTATION_TOLERANCE:
            self.stage = Stage.CLOSE_GRIPPER
            self._gripper_close_counter = 0
            self._grasp_target = target_position.copy()
        return RobotAction(joint_targets=joint_targets, gripper_target=GRIPPER_OPEN)

    def _tick_close_gripper(self, robot_state: RobotState, jacobian: tuple) -> RobotAction:
        """Hold the grasp pose (position + orientation) while closing.

        Re-solves pose IK toward the fixed pose reached at the end of
        DESCEND every tick, rather than freezing joint targets, so the arm
        actively resists drift from contact forces as the fingers close
        instead of passively holding whatever the last commanded targets
        were.
        """
        self._gripper_close_counter += 1
        _, _, joint_targets = self._pose_ik_step(robot_state, jacobian, self._grasp_target)
        if self._gripper_close_counter >= GRIPPER_CLOSE_STEPS:
            self.stage = Stage.LIFT
            self._lift_target = self._grasp_target + np.array([0.0, 0.0, LIFT_HEIGHT])
        return RobotAction(joint_targets=joint_targets, gripper_target=GRIPPER_CLOSED)

    def _tick_lift(self, robot_state: RobotState, jacobian: tuple) -> RobotAction:
        distance, orientation_distance, joint_targets = self._pose_ik_step(
            robot_state, jacobian, self._lift_target, max_joint_delta=LIFT_MAX_JOINT_DELTA
        )
        if distance < POSITION_TOLERANCE and orientation_distance < ORIENTATION_TOLERANCE:
            self.stage = Stage.DONE
        return RobotAction(joint_targets=joint_targets, gripper_target=GRIPPER_CLOSED)

    def _tick_done(self, robot_state: RobotState) -> RobotAction:
        self._clear_pose_diagnostics()
        return RobotAction(
            joint_targets=robot_state.joint_positions.copy(), gripper_target=GRIPPER_CLOSED
        )

    def _pose_ik_step(
        self,
        robot_state: RobotState,
        jacobian: tuple,
        target_position: np.ndarray,
        max_joint_delta: float = IK_MAX_JOINT_DELTA,
    ) -> tuple:
        jacobian_position, jacobian_rotation = jacobian
        position_error = target_position - robot_state.end_effector_position
        rotation_error = orientation_error(
            robot_state.end_effector_orientation, self._target_orientation
        )

        distance = float(np.linalg.norm(position_error))
        orientation_distance = float(np.linalg.norm(rotation_error))

        dq = solve_pose_ik(
            position_error,
            rotation_error,
            jacobian_position,
            jacobian_rotation,
            orientation_weight=ORIENTATION_WEIGHT,
            damping=IK_DAMPING,
            max_joint_delta=max_joint_delta,
        )
        joint_targets = robot_state.joint_positions + dq

        self.last_target_position = target_position
        self.last_position_error = distance
        self.last_orientation_error_deg = rotation_vector_degrees(rotation_error)

        return distance, orientation_distance, joint_targets

    def _clear_pose_diagnostics(self) -> None:
        self.last_target_position = None
        self.last_position_error = None
        self.last_orientation_error_deg = None
