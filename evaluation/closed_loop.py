"""Closed-loop Dense VLA rollout: Observation -> DenseVLAPolicy -> RobotAction -> env.step().

This is the ONLY place in Step 5 that drives the robot with the learned
policy. Per the input-contract boundary (see README "Step 5"), the policy
is called with exactly ``(observation, instruction)`` -- never cube
position, Jacobian, or controller stage. Those may only be read from
``env`` afterward, for evaluation/diagnostics, never fed back into
``policy.predict()``.

Task success is measured the same way as Step 2.5/3 -- sustained physical
cube lift (``control.success.sustained_lift_success``) -- never a model
"done" signal (the model doesn't have one), and never
``ScriptedController`` (it is not instantiated anywhere in this module).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from control.action import RobotAction
from control.success import cube_lift_delta, sustained_lift_success
from models.policy import DenseVLAPolicy
from simulation.environment import SimulationEnvironment

CUBE_BODY_NAME = "red_cube"
DEFAULT_MAX_STEPS = 350


@dataclass
class StepRecord:
    """One control tick of optional trajectory diagnostics (evaluation-only)."""

    step: int
    inference_latency_ms: float
    env_step_latency_ms: float
    gripper_probability: float
    eef_position: list
    cube_position: list
    cube_height_delta: float
    joint_targets: list


@dataclass
class RolloutResult:
    success: bool
    termination_reason: str  # "success" | "timeout" | "runtime_error"
    episode_length: int
    instruction: str
    cube_xy_offset: list
    initial_cube_position: list
    final_cube_position: list
    cube_lift_delta: float
    inference_latencies_ms: list = field(default_factory=list)
    env_step_latencies_ms: list = field(default_factory=list)
    gripper_probabilities: list = field(default_factory=list)
    steps: list = field(default_factory=list)  # list[StepRecord], only if record_trajectory=True

    def to_summary_dict(self) -> dict:
        """Episode record without the (potentially large) per-step trajectory."""
        data = {k: v for k, v in self.__dict__.items() if k != "steps"}
        return data


def _validate_action(action: RobotAction) -> None:
    """Defensive re-check at the closed-loop integration boundary.

    ``RobotAction.__post_init__`` already enforces all of this at
    construction time inside ``policy.predict()`` -- this exists so the
    contract is explicit and visible right where the action crosses from
    "model output" to "commanded to the robot" (see README "Action safety
    validation"), not because it can currently fail silently.
    """
    if action.joint_targets.shape != (7,):
        raise ValueError(f"policy produced joint_targets shape {action.joint_targets.shape}, expected (7,)")
    if not np.all(np.isfinite(action.joint_targets)):
        raise ValueError("policy produced non-finite joint_targets")
    if not np.isfinite(action.gripper_target):
        raise ValueError("policy produced non-finite gripper_target")
    if not (0.0 <= action.gripper_target <= 1.0):
        raise ValueError(f"policy produced gripper_target {action.gripper_target} outside [0, 1]")


def _smooth_action(action: RobotAction, previous_action: RobotAction, alpha: float) -> RobotAction:
    """EMA smoothing (README "Action smoothing"): opt-in only, off by default."""
    if previous_action is None:
        return action
    joint_targets = alpha * action.joint_targets + (1.0 - alpha) * previous_action.joint_targets
    gripper_target = alpha * action.gripper_target + (1.0 - alpha) * previous_action.gripper_target
    return RobotAction(joint_targets=joint_targets, gripper_target=gripper_target)


def run_closed_loop_episode(
    env: SimulationEnvironment,
    policy: DenseVLAPolicy,
    instruction: str,
    cube_xy_offset: np.ndarray = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    record_trajectory: bool = False,
    on_step: callable = None,
    smoothing_alpha: float = None,
) -> RolloutResult:
    """Run one closed-loop episode entirely under the learned policy.

    Every action in this rollout comes from ``policy.predict()``. No
    ``ScriptedController`` is instantiated or consulted -- see module
    docstring and ``tests/test_no_privileged_vla_inputs.py``.

    Args:
        on_step: Optional ``(tick, observation, action) -> None`` callback,
            invoked after each ``env.step()`` -- e.g. for the demo runner
            to save periodic frames. Purely observational: it cannot
            influence the action already taken.
        smoothing_alpha: Optional EMA smoothing weight in ``(0, 1]`` applied
            to the raw predicted action before it is sent to
            ``env.step()``: ``smoothed = alpha*pred + (1-alpha)*previous``.
            ``None`` (default) disables smoothing -- the RAW baseline must
            always be measured this way first (see README "Action
            smoothing"); this is an opt-in remedy evaluated separately,
            never the default.
    """
    env.reset()
    if hasattr(policy, "reset"):
        # Stateful policies (e.g. TemporalDenseVLAPolicy) must not leak
        # observation/action history across episodes -- see README
        # "Episode reset". Dense/MoE policies have no reset() and are
        # stateless, so this is a no-op for them.
        policy.reset()
    if cube_xy_offset is not None:
        cube_position = env.get_object_position(CUBE_BODY_NAME)
        cube_position[0] += cube_xy_offset[0]
        cube_position[1] += cube_xy_offset[1]
        env.set_object_position(CUBE_BODY_NAME, cube_position)

    initial_cube_position = env.get_object_position(CUBE_BODY_NAME).copy()
    initial_cube_z = float(initial_cube_position[2])
    cube_z_history = []

    inference_latencies_ms = []
    env_step_latencies_ms = []
    gripper_probabilities = []
    steps = []

    success = False
    step_count = 0
    previous_action = None
    for tick in range(max_steps):
        # The ONLY inputs the policy ever sees: the current Observation
        # (RGB + 23D state) and the fixed instruction string.
        observation = env.get_observation()

        inference_start = time.perf_counter()
        action = policy.predict(observation=observation, instruction=instruction)
        inference_end = time.perf_counter()
        _validate_action(action)

        if smoothing_alpha is not None:
            action = _smooth_action(action, previous_action, smoothing_alpha)
        previous_action = action

        env.step(action)
        step_end = time.perf_counter()

        inference_latencies_ms.append((inference_end - inference_start) * 1000.0)
        env_step_latencies_ms.append((step_end - inference_end) * 1000.0)
        gripper_probabilities.append(action.gripper_target)

        # Privileged state read here is for EVALUATION ONLY -- it is never
        # passed back into policy.predict().
        cube_position = env.get_object_position(CUBE_BODY_NAME)
        cube_z_history.append(float(cube_position[2]))

        if record_trajectory:
            robot_state = env.get_robot_state()
            steps.append(
                StepRecord(
                    step=tick,
                    inference_latency_ms=inference_latencies_ms[-1],
                    env_step_latency_ms=env_step_latencies_ms[-1],
                    gripper_probability=action.gripper_target,
                    eef_position=robot_state.end_effector_position.tolist(),
                    cube_position=cube_position.tolist(),
                    cube_height_delta=float(cube_position[2] - initial_cube_z),
                    joint_targets=action.joint_targets.tolist(),
                )
            )

        if on_step is not None:
            on_step(tick, observation, action)

        step_count = tick + 1
        if sustained_lift_success(cube_z_history, initial_cube_z):
            success = True
            break

    final_cube_position = env.get_object_position(CUBE_BODY_NAME).copy()
    lift_delta = cube_lift_delta(cube_z_history, initial_cube_z)
    termination_reason = "success" if success else "timeout"

    return RolloutResult(
        success=success,
        termination_reason=termination_reason,
        episode_length=step_count,
        instruction=instruction,
        cube_xy_offset=(cube_xy_offset.tolist() if cube_xy_offset is not None else [0.0, 0.0]),
        initial_cube_position=initial_cube_position.tolist(),
        final_cube_position=final_cube_position.tolist(),
        cube_lift_delta=lift_delta,
        inference_latencies_ms=inference_latencies_ms,
        env_step_latencies_ms=env_step_latencies_ms,
        gripper_probabilities=gripper_probabilities,
        steps=steps,
    )
