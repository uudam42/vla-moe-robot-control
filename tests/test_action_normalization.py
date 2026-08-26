"""Tests for training/normalization.py::ActionNormalizer and fit_normalizers_from_split."""

import numpy as np

from training.normalization import ActionNormalizer, fit_normalizers_from_split


def test_fit_uses_only_the_given_joint_targets():
    joint_targets = np.array([[0.0] * 7, [2.0] * 7, [4.0] * 7])
    normalizer = ActionNormalizer.fit(joint_targets)
    assert np.allclose(normalizer.joint_mean, 2.0)
    assert np.allclose(normalizer.joint_std, np.std([0.0, 2.0, 4.0]))


def test_normalize_denormalize_roundtrip():
    rng = np.random.default_rng(0)
    joint_targets = rng.normal(loc=0.5, scale=2.0, size=(50, 7))
    normalizer = ActionNormalizer.fit(joint_targets)

    sample = joint_targets[0]
    normalized = normalizer.normalize_joints(sample)
    reconstructed = normalizer.denormalize_joints(normalized)
    assert np.allclose(reconstructed, sample, atol=1e-6)


def test_normalized_joints_have_near_zero_mean_and_unit_std():
    rng = np.random.default_rng(1)
    joint_targets = rng.normal(loc=3.0, scale=0.7, size=(500, 7))
    normalizer = ActionNormalizer.fit(joint_targets)

    normalized = normalizer.normalize_joints(joint_targets)
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-6)


def test_near_zero_std_dimension_is_handled_safely():
    joint_targets = np.zeros((10, 7))
    joint_targets[:, 3] = 1.0  # one perfectly constant dimension
    normalizer = ActionNormalizer.fit(joint_targets)

    assert normalizer.joint_std[3] == 1.0  # fallback, not ~0
    normalized = normalizer.normalize_joints(joint_targets)
    assert np.all(np.isfinite(normalized))


def test_to_dict_from_dict_roundtrip():
    normalizer = ActionNormalizer(joint_mean=np.arange(7, dtype=np.float64), joint_std=np.ones(7))
    restored = ActionNormalizer.from_dict(normalizer.to_dict())
    assert np.allclose(restored.joint_mean, normalizer.joint_mean)
    assert np.allclose(restored.joint_std, normalizer.joint_std)


def test_fit_normalizers_from_split_uses_only_train_episodes(tmp_path):
    """Regression: normalization stats must come from train only, never val/test."""
    import json

    from control.action import RobotAction
    from dataset.recorder import EpisodeRecorder
    from dataset.splits import save_splits
    from observations.observation import Observation

    def make_obs(t, joint_value):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        state = np.zeros(23, dtype=np.float64)
        return Observation(rgb=rgb, state=state, timestamp=float(t))

    def write_episode(episode_id, joint_value, split_root):
        recorder = EpisodeRecorder(split_root, episode_id)
        for t in range(3):
            recorder.record(
                make_obs(t, joint_value),
                RobotAction(joint_targets=np.full(7, joint_value), gripper_target=0.5),
                "TEST",
            )
        return recorder.finalize(split_root / "successful", "Pick up the red cube.", True, {}).name

    train_name = write_episode(0, joint_value=0.0, split_root=tmp_path)
    val_name = write_episode(1, joint_value=1000.0, split_root=tmp_path)  # wildly different -- must be excluded

    save_splits({"train": [train_name], "val": [val_name], "test": []}, tmp_path / "splits.json")

    state_normalizer, action_normalizer = fit_normalizers_from_split(tmp_path, split="train")

    # If val's episode (mean joint value 1000) leaked in, the mean would be way off 0.
    assert np.allclose(action_normalizer.joint_mean, 0.0, atol=1e-6)
