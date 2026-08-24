"""Minimal damped least-squares IK math.

This module is pure NumPy linear algebra: it consumes Jacobians and
Cartesian/rotational errors and returns joint-space deltas. It does not
know about MuJoCo, joint names, or actuators -- ``SimulationEnvironment``
computes Jacobians (a simulator-specific operation, via MuJoCo's
``mj_jacSite``) and hands them here as plain arrays, so this module would
work unchanged against Jacobians from any other kinematics source.

Quaternion convention throughout this module: ``(w, x, y, z)``, matching
MuJoCo's convention (``mju_mat2Quat`` et al.) and
``RobotState.end_effector_orientation``.
"""

import numpy as np


def damped_least_squares_ik(
    jacobian: np.ndarray,
    position_error: np.ndarray,
    damping: float = 0.05,
    max_joint_delta: float = 0.2,
) -> np.ndarray:
    """Compute a joint-space delta that reduces a Cartesian position error.

    Implements ``dq = J^T (J J^T + lambda^2 I)^-1 e``, a standard damped
    least-squares (Levenberg-Marquardt-style) pseudoinverse step. This is a
    single small step, not a full IK solve to convergence -- callers
    (the scripted controller) are expected to call this repeatedly in a
    closed loop as the robot moves.

    Args:
        jacobian: Translational end-effector Jacobian, shape ``(3, N)``
            where ``N`` is the number of controlled joints.
        position_error: Cartesian position error (target - current), shape
            ``(3,)``.
        damping: Damping factor lambda. Larger values are more stable near
            singularities but converge more slowly.
        max_joint_delta: Maximum absolute change allowed per joint, radians.
            Prevents a single step from commanding a large, potentially
            unsafe joint jump.

    Returns:
        Joint-space delta, shape ``(N,)``, clamped to
        ``[-max_joint_delta, max_joint_delta]`` per element.
    """
    if jacobian.ndim != 2 or jacobian.shape[0] != 3:
        raise ValueError(f"jacobian must have shape (3, N), got {jacobian.shape}")
    if position_error.shape != (3,):
        raise ValueError(
            f"position_error must have shape (3,), got {position_error.shape}"
        )

    jjt = jacobian @ jacobian.T
    damped = jjt + (damping**2) * np.eye(3)
    dq = jacobian.T @ np.linalg.solve(damped, position_error)

    if not np.all(np.isfinite(dq)):
        raise ValueError("IK produced a non-finite joint delta")

    return np.clip(dq, -max_joint_delta, max_joint_delta)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 * q2``, quaternions as ``(w, x, y, z)``."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (== inverse, for a unit quaternion) of ``q = (w, x, y, z)``."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def orientation_error(current_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    """Small rotation vector (axis * angle, radians) taking current -> target.

    Computes the relative rotation ``q_err = q_target * conjugate(q_current)``
    and converts it to a 3D rotation vector. Handles the quaternion
    double-cover (``q`` and ``-q`` represent the same orientation) by
    flipping ``q_err`` to have a non-negative scalar part before conversion,
    which guarantees the shortest-path rotation (angle in ``[0, pi]``) and
    avoids a discontinuous jump when a quaternion's sign flips between
    control ticks.

    Args:
        current_quat: Current orientation, shape ``(4,)``, ``(w, x, y, z)``.
        target_quat: Target orientation, shape ``(4,)``, ``(w, x, y, z)``.

    Returns:
        Rotation vector, shape ``(3,)``, radians. Near-zero when the two
        orientations are (approximately) identical.
    """
    if current_quat.shape != (4,) or target_quat.shape != (4,):
        raise ValueError(
            "orientation_error requires shape (4,) quaternions, got "
            f"{current_quat.shape} and {target_quat.shape}"
        )

    current = current_quat / np.linalg.norm(current_quat)
    target = target_quat / np.linalg.norm(target_quat)

    q_err = quaternion_multiply(target, quaternion_conjugate(current))
    if q_err[0] < 0.0:
        q_err = -q_err  # shortest path: q and -q are the same rotation

    w = float(np.clip(q_err[0], -1.0, 1.0))
    v = q_err[1:]
    v_norm = float(np.linalg.norm(v))

    if v_norm < 1e-9:
        return np.zeros(3)

    angle = 2.0 * np.arctan2(v_norm, w)
    axis = v / v_norm
    return axis * angle


def rotation_vector_degrees(rotation_vector: np.ndarray) -> float:
    """Magnitude of a rotation vector (see :func:`orientation_error`), in degrees."""
    return float(np.degrees(np.linalg.norm(rotation_vector)))


def solve_pose_ik(
    position_error: np.ndarray,
    orientation_error_vector: np.ndarray,
    jacobian_position: np.ndarray,
    jacobian_rotation: np.ndarray,
    orientation_weight: float = 0.3,
    damping: float = 0.08,
    max_joint_delta: float = 0.15,
) -> np.ndarray:
    """6D damped least-squares pose IK step (position + orientation).

    Stacks the translational and (weighted) rotational Jacobians/errors
    into a single 6xN system and takes one damped least-squares step:
    ``dq = J^T (J J^T + lambda^2 I)^-1 e``, the same formulation as
    :func:`damped_least_squares_ik`, generalized from 3 to 6 rows.

    Args:
        position_error: Cartesian position error (target - current), ``(3,)``.
        orientation_error_vector: Rotation vector error, ``(3,)`` radians
            (see :func:`orientation_error`).
        jacobian_position: Translational Jacobian, ``(3, N)``.
        jacobian_rotation: Rotational Jacobian, ``(3, N)``.
        orientation_weight: Scales the orientation error/Jacobian rows
            before stacking, to balance position (meters) against
            orientation (radians) in one combined least-squares solve.
        damping: Damping factor lambda.
        max_joint_delta: Maximum absolute change allowed per joint, radians.

    Returns:
        Joint-space delta, shape ``(N,)``, clamped to
        ``[-max_joint_delta, max_joint_delta]`` per element.
    """
    if jacobian_position.ndim != 2 or jacobian_position.shape[0] != 3:
        raise ValueError(
            f"jacobian_position must have shape (3, N), got {jacobian_position.shape}"
        )
    if jacobian_rotation.shape != jacobian_position.shape:
        raise ValueError(
            "jacobian_rotation must have the same shape as jacobian_position, "
            f"got {jacobian_rotation.shape} vs {jacobian_position.shape}"
        )
    if position_error.shape != (3,):
        raise ValueError(f"position_error must have shape (3,), got {position_error.shape}")
    if orientation_error_vector.shape != (3,):
        raise ValueError(
            f"orientation_error_vector must have shape (3,), got {orientation_error_vector.shape}"
        )

    error = np.concatenate([position_error, orientation_weight * orientation_error_vector])
    jacobian = np.vstack([jacobian_position, orientation_weight * jacobian_rotation])

    jjt = jacobian @ jacobian.T
    damped = jjt + (damping**2) * np.eye(6)
    dq = jacobian.T @ np.linalg.solve(damped, error)

    if not np.all(np.isfinite(dq)):
        raise ValueError("Pose IK produced a non-finite joint delta")

    return np.clip(dq, -max_joint_delta, max_joint_delta)
