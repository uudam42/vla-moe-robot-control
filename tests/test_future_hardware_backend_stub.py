"""README "Future hardware backend stub": documents the extension point
without pretending real hardware support exists."""

import pytest

from robot_backend.base import RobotBackend
from robot_backend.future_hardware_backend import FutureHardwareBackend


def test_future_hardware_backend_is_a_robot_backend_subclass():
    """Proves the intended future contract requires no ABC changes -- a
    real implementation would only need to fill in these four methods."""
    assert issubclass(FutureHardwareBackend, RobotBackend)


def test_future_hardware_backend_cannot_be_instantiated():
    with pytest.raises(NotImplementedError):
        FutureHardwareBackend()
