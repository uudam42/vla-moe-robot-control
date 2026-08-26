"""Closed-loop rollout driven through the ``RobotBackend`` interface
instead of a raw ``SimulationEnvironment`` (README "refactor one direct
runner to use RobotBackend").

Deliberately a NEW function, not a modification of
``evaluation.closed_loop.run_closed_loop_episode`` -- the existing direct
path stays exactly as Steps 5-8 left it (README "Do not remove the direct
execution path" / "Preserve existing research baselines"). This module
reuses that module's ``RolloutResult``/``StepRecord`` schema and
``_validate_action`` so both paths are byte-for-byte comparable, and is
verified to produce IDENTICAL actions to the direct path given the same
seed/checkpoint/instruction by
``tests/test_backend_closed_loop_equivalence.py``.

Task-success detection still needs the ground-truth cube position, which
is deliberately NOT part of the ``RobotBackend`` ABC (README "RobotBackend"
docstring) -- this function therefore only works with a backend that also
exposes ``get_object_position``/`` (i.e. ``MuJoCoBackend`` today). A real
hardware backend would need a different, real success signal; this
function is a simulation-evaluation convenience, not part of the
policy-facing contract.
"""

import time

import numpy as np

from control.success import cube_lift_delta, sustained_lift_success
from evaluation.closed_loop import DEFAULT_MAX_STEPS, RolloutResult, StepRecord, _validate_action
from robot_backend.base import RobotBackend

CUBE_BODY_NAME = "red_cube"


def run_closed_loop_episode_via_backend(
    backend: RobotBackend,
    policy,
    instruction: str,
    cube_xy_offset: np.ndarray = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    record_trajectory: bool = False,
    on_step: callable = None,
) -> RolloutResult:
    """Same rollout/metrics contract as
    ``evaluation.closed_loop.run_closed_loop_episode``, but every
    environment interaction goes through ``backend`` (``get_observation``/
    ``execute_action``/``reset``), never a raw ``SimulationEnvironment``.
    """
    backend.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    if cube_xy_offset is not None:
        cube_position = backend.get_object_position(CUBE_BODY_NAME)
        cube_position[0] += cube_xy_offset[0]
        cube_position[1] += cube_xy_offset[1]
        backend.set_object_position(CUBE_BODY_NAME, cube_position)

    initial_cube_position = backend.get_object_position(CUBE_BODY_NAME).copy()
    initial_cube_z = float(initial_cube_position[2])
    cube_z_history = []

    inference_latencies_ms = []
    env_step_latencies_ms = []
    gripper_probabilities = []
    steps = []

    success = False
    step_count = 0
    for tick in range(max_steps):
        observation = backend.get_observation()

        inference_start = time.perf_counter()
        action = policy.predict(observation=observation, instruction=instruction)
        inference_end = time.perf_counter()
        _validate_action(action)

        backend.execute_action(action)
        step_end = time.perf_counter()

        inference_latencies_ms.append((inference_end - inference_start) * 1000.0)
        env_step_latencies_ms.append((step_end - inference_end) * 1000.0)
        gripper_probabilities.append(action.gripper_target)

        cube_position = backend.get_object_position(CUBE_BODY_NAME)
        cube_z_history.append(float(cube_position[2]))

        if record_trajectory:
            robot_state = backend.get_robot_state()
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

    final_cube_position = backend.get_object_position(CUBE_BODY_NAME).copy()
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
