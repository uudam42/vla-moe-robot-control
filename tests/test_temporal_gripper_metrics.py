"""Tests for evaluation/temporal_diagnostics.py -- the gripper-oscillation metric."""

from evaluation.temporal_diagnostics import (
    gripper_switch_count,
    gripper_switch_rate,
    summarize_gripper_stability,
    windowed_gripper_switch_count,
)


def test_no_switches_when_constant():
    assert gripper_switch_count([1.0, 1.0, 1.0, 1.0]) == 0
    assert gripper_switch_count([0.0, 0.0, 0.0]) == 0


def test_counts_open_close_transitions():
    # open, closed, open, closed, closed -> 3 transitions
    assert gripper_switch_count([1.0, 0.0, 0.9, 0.1, 0.2]) == 3


def test_threshold_boundary_behavior():
    # exactly at threshold counts as "open" (>=)
    assert gripper_switch_count([0.5, 0.4]) == 1
    assert gripper_switch_count([0.5, 0.5]) == 0


def test_switch_rate_normalizes_by_length():
    sequence = [1.0, 0.0, 1.0, 0.0]  # 3 switches over 4 samples -> 3/3
    assert gripper_switch_rate(sequence) == 1.0


def test_switch_rate_zero_for_short_sequences():
    assert gripper_switch_rate([1.0]) == 0.0
    assert gripper_switch_rate([]) == 0.0


def test_oscillating_sequence_has_high_switch_count():
    """The Step 5/6-diagnosed failure mode: gripper flips every tick."""
    oscillating = [1.0, 0.0] * 20
    assert gripper_switch_count(oscillating) == 39  # every consecutive pair differs


def test_smooth_monotonic_confidence_change_has_zero_switches():
    """Smoothly increasing confidence that crosses the threshold once is a
    single decision, not oscillation."""
    smooth = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
    assert gripper_switch_count(smooth) == 1


def test_windowed_switch_count_restricts_to_local_range():
    sequence = [1.0] * 10 + [1.0, 0.0] * 5 + [0.0] * 10  # oscillation only in the middle
    center = 12
    windowed = windowed_gripper_switch_count(sequence, center_index=center, window=5)
    full = gripper_switch_count(sequence)
    assert windowed <= full
    assert windowed > 0


def test_summarize_gripper_stability_aggregates_across_episodes():
    class FakeResult:
        def __init__(self, probs):
            self.gripper_probabilities = probs

    results = [FakeResult([1.0, 0.0, 1.0]), FakeResult([1.0, 1.0, 1.0])]
    summary = summarize_gripper_stability(results)
    assert summary["mean_switch_count"] == 1.0  # (2 + 0) / 2
    assert summary["max_switch_count"] == 2
