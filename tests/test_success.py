"""Tests for task-success metrics (control/success.py), independent of MuJoCo."""

import numpy as np

from control.success import cube_lift_delta, sustained_lift_success


def test_small_cube_movement_is_not_success():
    # Cube barely moves (e.g. jostled by the gripper), never sustained above threshold.
    history = np.full(50, 0.42) + np.random.default_rng(0).normal(0, 0.001, 50)
    assert sustained_lift_success(history, initial_cube_z=0.42) is False


def test_momentary_bump_is_not_success():
    """A brief spike above threshold that doesn't hold should not count."""
    history = np.full(30, 0.42)
    history[10:12] = 0.50  # 2-tick bump, well under the default sustain window
    assert sustained_lift_success(history, initial_cube_z=0.42, sustain_ticks=10) is False


def test_sustained_lift_is_success():
    history = np.concatenate([np.full(20, 0.42), np.full(20, 0.50)])
    assert sustained_lift_success(history, initial_cube_z=0.42, lift_threshold=0.04, sustain_ticks=10) is True


def test_lift_success_respects_custom_threshold():
    history = np.full(20, 0.45)  # only +0.03 m
    assert sustained_lift_success(history, initial_cube_z=0.42, lift_threshold=0.04, sustain_ticks=5) is False
    assert sustained_lift_success(history, initial_cube_z=0.42, lift_threshold=0.02, sustain_ticks=5) is True


def test_lift_success_false_on_empty_history():
    assert sustained_lift_success(np.array([]), initial_cube_z=0.42) is False


def test_cube_lift_delta_reports_peak_gain():
    history = np.array([0.42, 0.44, 0.50, 0.47])
    assert np.isclose(cube_lift_delta(history, initial_cube_z=0.42), 0.08)


def test_cube_lift_delta_zero_on_empty_history():
    assert cube_lift_delta(np.array([]), initial_cube_z=0.42) == 0.0
