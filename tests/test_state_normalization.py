"""Tests for training/normalization.py::StateNormalizer."""

import numpy as np

from training.normalization import StateNormalizer


def test_normalize_denormalize_roundtrip():
    rng = np.random.default_rng(0)
    states = rng.normal(loc=1.0, scale=3.0, size=(200, 23))
    normalizer = StateNormalizer.fit(states)

    sample = states[5]
    normalized = normalizer.normalize(sample)
    reconstructed = normalizer.denormalize(normalized)
    assert np.allclose(reconstructed, sample, atol=1e-6)


def test_normalized_states_have_near_zero_mean_and_unit_std():
    rng = np.random.default_rng(2)
    states = rng.normal(loc=-2.0, scale=0.3, size=(1000, 23))
    normalizer = StateNormalizer.fit(states)

    normalized = normalizer.normalize(states)
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-6)


def test_near_zero_std_dimension_is_handled_safely():
    states = np.random.default_rng(0).normal(size=(20, 23))
    states[:, 0] = 5.0  # constant dimension (e.g. a joint that never moves at HOME)
    normalizer = StateNormalizer.fit(states)

    assert normalizer.std[0] == 1.0
    normalized = normalizer.normalize(states)
    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized[:, 0], 0.0)  # (5.0 - 5.0) / 1.0


def test_to_dict_from_dict_roundtrip():
    normalizer = StateNormalizer(mean=np.arange(23, dtype=np.float64), std=np.ones(23) * 2.0)
    restored = StateNormalizer.from_dict(normalizer.to_dict())
    assert np.allclose(restored.mean, normalizer.mean)
    assert np.allclose(restored.std, normalizer.std)
