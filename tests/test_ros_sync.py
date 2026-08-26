"""Tests for ros_integration/sync.py -- observation synchronization and staleness."""

import numpy as np

from ros_integration.sync import LatestMessageSynchronizer, StalenessChecker


def make_rgb():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_no_sync_until_both_streams_present():
    synchronizer = LatestMessageSynchronizer(max_sync_delta_sec=0.1)
    assert synchronizer.get_synced() is None
    synchronizer.update_image(make_rgb(), timestamp=1.0)
    assert synchronizer.get_synced() is None  # only image so far
    synchronizer.update_state(np.zeros(23), timestamp=1.02)
    synced = synchronizer.get_synced()
    assert synced is not None
    assert synced.image_timestamp == 1.0
    assert synced.state_timestamp == 1.02
    assert np.isclose(synced.sync_delta, 0.02)


def test_sync_rejected_beyond_max_delta():
    synchronizer = LatestMessageSynchronizer(max_sync_delta_sec=0.1)
    synchronizer.update_image(make_rgb(), timestamp=1.0)
    synchronizer.update_state(np.zeros(23), timestamp=2.0)  # 1.0s apart, way over 0.1s
    assert synchronizer.get_synced() is None


def test_sync_uses_latest_message_only():
    synchronizer = LatestMessageSynchronizer(max_sync_delta_sec=0.1)
    synchronizer.update_image(make_rgb(), timestamp=1.0)
    synchronizer.update_state(np.zeros(23), timestamp=1.0)
    synchronizer.update_image(make_rgb(), timestamp=5.0)  # newer image arrives
    synced = synchronizer.get_synced()
    assert synced is None  # 5.0 vs 1.0 state -> not synced anymore
    synchronizer.update_state(np.ones(23), timestamp=5.05)
    synced = synchronizer.get_synced()
    assert synced is not None
    assert np.allclose(synced.state, 1.0)


def test_reset_clears_both_streams():
    synchronizer = LatestMessageSynchronizer(max_sync_delta_sec=0.1)
    synchronizer.update_image(make_rgb(), timestamp=1.0)
    synchronizer.update_state(np.zeros(23), timestamp=1.0)
    synchronizer.reset()
    assert synchronizer.get_synced() is None


def test_staleness_checker_flags_old_observation():
    synchronizer = LatestMessageSynchronizer(max_sync_delta_sec=0.1)
    synchronizer.update_image(make_rgb(), timestamp=1.0)
    synchronizer.update_state(np.zeros(23), timestamp=1.0)
    synced = synchronizer.get_synced()

    checker = StalenessChecker(max_age_sec=0.5)
    assert checker.is_stale(synced, now=1.1) is False
    assert checker.is_stale(synced, now=2.0) is True


def test_ros_sync_module_never_imports_rclpy():
    import inspect

    import ros_integration.sync as module

    assert not hasattr(module, "rclpy")
    assert "import rclpy" not in inspect.getsource(module)
