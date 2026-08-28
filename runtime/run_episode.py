"""``run_episode``: the Step 10 production runtime loop (README "Final
system architecture" / "Call relationship section", direct path):

```
RobotBackend.get_observation()
    -> Observation
    -> policy.predict(observation, instruction)
    -> RobotAction
    -> SafetySupervisor.process(...)
    -> safe RobotAction
    -> RobotBackend.execute_action(...)
```

with structured telemetry recorded every tick via
``telemetry.recorder.EpisodeTelemetryRecorder``. Task-success detection
(privileged, evaluation-only) reuses ``control.success`` unchanged --
never fed to the policy (README "Preserve no-privileged-input rule").
Backends without privileged object-position access (a future hardware
backend, ``FakeRobotBackend``) simply skip success detection --
``get_object_position``/``set_object_position`` are duck-typed, checked
with ``hasattr`` rather than assumed.
"""

import time

import numpy as np

from control.success import cube_lift_delta, sustained_lift_success
from robot_backend.base import RobotBackend
from safety.supervisor import SafetyDecision, SafetySupervisor
from telemetry.recorder import EpisodeTelemetryRecorder

CUBE_BODY_NAME = "red_cube"
DEFAULT_MAX_STEPS = 350


def run_episode(
    backend: RobotBackend,
    policy,
    safety_supervisor: SafetySupervisor,
    recorder: EpisodeTelemetryRecorder,
    instruction: str,
    cube_xy_offset: np.ndarray = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    save_frames: bool = False,
) -> dict:
    """Runs one episode end-to-end and returns the finalized metrics dict
    (same dict ``recorder.finalize()`` returns and writes to ``metrics.json``)."""
    backend.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    safety_supervisor.reset()

    privileged = hasattr(backend, "get_object_position") and hasattr(backend, "set_object_position")
    initial_cube_z = None
    cube_z_history = []
    if privileged:
        if cube_xy_offset is not None:
            cube_position = backend.get_object_position(CUBE_BODY_NAME)
            cube_position[0] += cube_xy_offset[0]
            cube_position[1] += cube_xy_offset[1]
            backend.set_object_position(CUBE_BODY_NAME, cube_position)
        initial_cube_z = float(backend.get_object_position(CUBE_BODY_NAME)[2])

    recorder.start(save_frames=save_frames)

    success = False
    failure_reason = None
    step_count = 0
    for tick in range(max_steps):
        # The ONLY inputs the policy ever sees: the current Observation and instruction.
        observation = backend.get_observation()
        wall_clock_timestamp = time.time()

        prediction_start = time.perf_counter()
        action = policy.predict(observation=observation, instruction=instruction)
        prediction_end = time.perf_counter()

        safe_action, safety_event = safety_supervisor.process(
            action,
            observation_timestamp=observation.timestamp,
            inference_latency_sec=prediction_end - prediction_start,
            now=wall_clock_timestamp,
        )

        backend_execute_latency_sec = None
        if safe_action is not None:
            exec_start = time.perf_counter()
            backend.execute_action(safe_action)
            backend_execute_latency_sec = time.perf_counter() - exec_start

        cube_height_delta = None
        if privileged:
            cube_position = backend.get_object_position(CUBE_BODY_NAME)
            cube_z_history.append(float(cube_position[2]))
            cube_height_delta = float(cube_position[2] - initial_cube_z)

        if save_frames:
            recorder.save_frame(tick, observation.rgb)

        recorder.record_step(
            step_id=tick,
            wall_clock_timestamp=wall_clock_timestamp,
            observation_timestamp=observation.timestamp,
            prediction_start=prediction_start,
            prediction_end=prediction_end,
            original_action=action,
            executed_action=safe_action,
            safety_decision=safety_event.decision.value,
            safety_reason=safety_event.reason.value,
            backend_execute_latency_sec=backend_execute_latency_sec,
            simulation_timestamp=observation.timestamp,
            cube_height_delta=cube_height_delta,
        )

        step_count = tick + 1

        if safety_event.decision == SafetyDecision.STOP_EPISODE:
            failure_reason = "safety_stop"
            break
        if privileged and sustained_lift_success(cube_z_history, initial_cube_z):
            success = True
            break

    if not success and failure_reason is None:
        failure_reason = "timeout"

    max_lift = cube_lift_delta(cube_z_history, initial_cube_z) if cube_z_history else None
    final_lift = (cube_z_history[-1] - initial_cube_z) if cube_z_history else None

    return recorder.finalize(
        success=success,
        failure_reason=None if success else failure_reason,
        cube_lift_delta=final_lift,
        max_cube_lift=max_lift,
        extra_metadata={"episode_length": step_count},
    )
