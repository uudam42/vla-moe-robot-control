"""Unit tests for the Observation data structure itself (no MuJoCo)."""

import numpy as np
import pytest

from observations.observation import Observation


def make_observation(width: int = 64, height: int = 48) -> Observation:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    state = np.zeros(6, dtype=np.float64)
    return Observation(rgb=rgb, state=state, timestamp=0.0)


def test_observation_field_types():
    obs = make_observation()
    assert isinstance(obs.rgb, np.ndarray)
    assert isinstance(obs.state, np.ndarray)
    assert isinstance(obs.timestamp, float)


def test_observation_rgb_shape():
    obs = make_observation(width=64, height=48)
    assert obs.rgb.ndim == 3
    assert obs.rgb.shape[-1] == 3
    assert obs.rgb.shape == (48, 64, 3)


def test_observation_rejects_bad_rgb_shape():
    with pytest.raises(ValueError):
        Observation(
            rgb=np.zeros((10, 10), dtype=np.uint8),
            state=np.zeros(3),
            timestamp=0.0,
        )


def test_observation_rejects_non_array_rgb():
    with pytest.raises(TypeError):
        Observation(rgb=[[1, 2, 3]], state=np.zeros(3), timestamp=0.0)


def test_observation_timestamp_is_numeric():
    obs = make_observation()
    assert isinstance(obs.timestamp, (int, float))
