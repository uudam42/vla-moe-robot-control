"""Aggregate metrics and heuristic failure classification over a batch of
``evaluation.closed_loop.RolloutResult``.
"""

import numpy as np

from evaluation.closed_loop import RolloutResult

# Categories from README "Failure categories". Heuristic, not ground truth --
# useful for prioritizing what to look at, not a precise labeling.
NEVER_REACHED_CUBE = "never_reached_cube"
REACHED_BUT_MISALIGNED = "reached_but_misaligned"
PUSHED_CUBE_AWAY = "pushed_cube_away"
GRIPPER_NEVER_CLOSED = "gripper_never_closed"
GRASPED_BUT_DROPPED = "grasped_but_dropped"
FAILED_TO_LIFT = "failed_to_lift"
TIMEOUT_UNCATEGORIZED = "timeout_uncategorized"
RUNTIME_ERROR = "numerical_or_runtime_failure"

REACH_DISTANCE_THRESHOLD = 0.08  # meters; eef-cube distance considered "reached"
PUSH_DISPLACEMENT_THRESHOLD = 0.03  # meters; cube xy displacement considered "pushed"
PARTIAL_LIFT_THRESHOLD = 0.02  # meters; any measurable lift at all


def latency_stats(latencies_ms: list) -> dict:
    if not latencies_ms:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    array = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)) if array.size >= 20 else None,
    }


def classify_failure(result: RolloutResult) -> str:
    """Best-effort heuristic label for one failed episode. See module docstring."""
    if result.termination_reason == "runtime_error":
        return RUNTIME_ERROR

    gripper_ever_closed = any(p < 0.5 for p in result.gripper_probabilities)
    initial_xy = np.array(result.initial_cube_position[:2])
    final_xy = np.array(result.final_cube_position[:2])
    xy_displacement = float(np.linalg.norm(final_xy - initial_xy))
    initial_z = result.initial_cube_position[2]
    final_z = result.final_cube_position[2]

    min_eef_cube_distance = None
    if result.steps:
        eef_positions = np.array([s.eef_position for s in result.steps])
        cube_positions = np.array([s.cube_position for s in result.steps])
        min_eef_cube_distance = float(np.linalg.norm(eef_positions - cube_positions, axis=1).min())

    if min_eef_cube_distance is not None and min_eef_cube_distance > REACH_DISTANCE_THRESHOLD:
        return NEVER_REACHED_CUBE

    if not gripper_ever_closed:
        return GRIPPER_NEVER_CLOSED

    if result.cube_lift_delta > PARTIAL_LIFT_THRESHOLD and (final_z - initial_z) < 0.01:
        return GRASPED_BUT_DROPPED

    if xy_displacement > PUSH_DISPLACEMENT_THRESHOLD and result.cube_lift_delta < PARTIAL_LIFT_THRESHOLD:
        return PUSHED_CUBE_AWAY

    if min_eef_cube_distance is not None and result.cube_lift_delta < PARTIAL_LIFT_THRESHOLD:
        return REACHED_BUT_MISALIGNED

    if result.cube_lift_delta < PARTIAL_LIFT_THRESHOLD:
        return FAILED_TO_LIFT

    return TIMEOUT_UNCATEGORIZED


def summarize_results(results: list) -> dict:
    """Aggregate a batch of RolloutResult into the closed-loop evaluation summary."""
    num_episodes = len(results)
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    all_inference_latencies = [ms for r in results for ms in r.inference_latencies_ms]
    latency = latency_stats(all_inference_latencies)
    effective_hz = 1000.0 / latency["mean_ms"] if latency["mean_ms"] else None

    failure_counts = {}
    for result in failures:
        category = classify_failure(result)
        failure_counts[category] = failure_counts.get(category, 0) + 1

    return {
        "num_episodes": num_episodes,
        "num_success": len(successes),
        "success_rate": (len(successes) / num_episodes) if num_episodes else 0.0,
        "mean_successful_episode_length": (
            float(np.mean([r.episode_length for r in successes])) if successes else None
        ),
        "mean_cube_lift_delta": float(np.mean([r.cube_lift_delta for r in results])) if results else None,
        "latency_ms": latency,
        "effective_control_hz": effective_hz,
        "failure_counts": failure_counts,
    }


def summarize_by_instruction(results: list) -> dict:
    """Success rate broken out per instruction string (README "Language evaluation")."""
    by_instruction = {}
    for result in results:
        by_instruction.setdefault(result.instruction, []).append(result)
    return {instruction: summarize_results(group) for instruction, group in by_instruction.items()}
