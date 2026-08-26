"""Tests for evaluation/metrics.py -- pure aggregation over synthetic RolloutResults."""

from evaluation.closed_loop import RolloutResult, StepRecord
from evaluation.metrics import (
    GRIPPER_NEVER_CLOSED,
    NEVER_REACHED_CUBE,
    PUSHED_CUBE_AWAY,
    classify_failure,
    latency_stats,
    summarize_by_instruction,
    summarize_results,
)


def make_result(success, episode_length=100, instruction="Pick up the red cube.", lift_delta=0.0,
                 gripper_probabilities=None, latencies=None, initial_xy=(0.45, 0.0), final_xy=(0.45, 0.0),
                 termination_reason=None, steps=None):
    return RolloutResult(
        success=success,
        termination_reason=termination_reason or ("success" if success else "timeout"),
        episode_length=episode_length,
        instruction=instruction,
        cube_xy_offset=[0.0, 0.0],
        initial_cube_position=[initial_xy[0], initial_xy[1], 0.42],
        final_cube_position=[final_xy[0], final_xy[1], 0.42 + lift_delta if success else 0.4199],
        cube_lift_delta=lift_delta,
        inference_latencies_ms=latencies or [10.0] * episode_length,
        gripper_probabilities=gripper_probabilities or [1.0] * episode_length,
        steps=steps or [],
    )


def test_latency_stats_basic():
    stats = latency_stats([10.0, 20.0, 30.0])
    assert stats["n"] == 3
    assert stats["mean_ms"] == 20.0
    assert stats["p50_ms"] == 20.0


def test_latency_stats_empty():
    stats = latency_stats([])
    assert stats["n"] == 0
    assert stats["mean_ms"] is None


def test_summarize_results_success_rate():
    results = [make_result(True), make_result(True), make_result(False)]
    summary = summarize_results(results)
    assert summary["num_episodes"] == 3
    assert summary["num_success"] == 2
    assert summary["success_rate"] == 2 / 3


def test_summarize_results_mean_successful_length():
    results = [make_result(True, episode_length=100), make_result(True, episode_length=200), make_result(False)]
    summary = summarize_results(results)
    assert summary["mean_successful_episode_length"] == 150.0


def test_summarize_results_no_successes_has_none_length():
    results = [make_result(False), make_result(False)]
    summary = summarize_results(results)
    assert summary["mean_successful_episode_length"] is None
    assert summary["success_rate"] == 0.0


def test_summarize_results_effective_control_hz():
    results = [make_result(True, latencies=[10.0, 10.0])]
    summary = summarize_results(results)
    assert summary["effective_control_hz"] == 100.0


def test_summarize_by_instruction_groups_correctly():
    results = [
        make_result(True, instruction="Pick up the red cube."),
        make_result(False, instruction="Pick up the red cube."),
        make_result(True, instruction="Grasp the red cube."),
    ]
    by_instruction = summarize_by_instruction(results)
    assert set(by_instruction) == {"Pick up the red cube.", "Grasp the red cube."}
    assert by_instruction["Pick up the red cube."]["num_episodes"] == 2
    assert by_instruction["Grasp the red cube."]["success_rate"] == 1.0


def test_classify_failure_never_reached_cube():
    steps = [
        StepRecord(step=i, inference_latency_ms=1.0, env_step_latency_ms=1.0, gripper_probability=1.0,
                   eef_position=[0.0, 0.0, 1.0], cube_position=[0.45, 0.0, 0.42], cube_height_delta=0.0,
                   joint_targets=[0.0] * 7)
        for i in range(5)
    ]
    result = make_result(False, lift_delta=0.0, steps=steps)
    assert classify_failure(result) == NEVER_REACHED_CUBE


def test_classify_failure_gripper_never_closed():
    result = make_result(False, lift_delta=0.0, gripper_probabilities=[1.0] * 50)
    assert classify_failure(result) == GRIPPER_NEVER_CLOSED


def test_classify_failure_pushed_cube_away():
    result = make_result(
        False, lift_delta=0.0, gripper_probabilities=[0.0] * 50,
        initial_xy=(0.45, 0.0), final_xy=(0.40, 0.05),
    )
    assert classify_failure(result) == PUSHED_CUBE_AWAY


def test_classify_failure_success_episodes_not_classified_as_failure():
    # classify_failure is only meaningful for failed episodes; summarize_results
    # must not call it on successes.
    results = [make_result(True)]
    summary = summarize_results(results)
    assert summary["failure_counts"] == {}
