"""End-to-end tests for dagger/collector.py: on-disk format, retention
bookkeeping, and disagreement diagnostics."""

import numpy as np
import pytest
import torch

from dagger.collector import EXPERT_LABELS_FILENAME, load_expert_labels
from dagger.disagreement import compute_disagreement
from dataset.episode import load_episode
from models.temporal_policy import TemporalDenseVLAPolicy
from simulation.environment import SimulationEnvironment


@pytest.fixture
def policy(tiny_temporal_vla_checkpoint):
    return TemporalDenseVLAPolicy.from_checkpoint(tiny_temporal_vla_checkpoint, device=torch.device("cpu"))


def _collect(tmp_path, policy, max_steps=10, sample_every=3, **kwargs):
    from dagger.collector import collect_episode

    with SimulationEnvironment() as env:
        return collect_episode(
            env, policy, episode_id=0, instruction="Pick up the red cube.",
            episodes_root=tmp_path / "episodes", max_steps=max_steps, sample_every=sample_every, **kwargs,
        )


def test_episode_reuses_step3_on_disk_format(tmp_path, policy):
    episode_dir, metadata = _collect(tmp_path, policy)

    assert (episode_dir / "trajectory.npz").exists()
    assert (episode_dir / "metadata.json").exists()
    assert (episode_dir / "rgb").is_dir()
    assert (episode_dir / EXPERT_LABELS_FILENAME).exists()

    episode = load_episode(episode_dir)  # must load with the UNMODIFIED Step 3 loader
    assert episode.length == metadata["num_candidate_timesteps"]


def test_expert_labels_sidecar_shapes_match_episode_length(tmp_path, policy):
    episode_dir, metadata = _collect(tmp_path, policy, max_steps=12, sample_every=4)
    episode = load_episode(episode_dir)
    labels = load_expert_labels(episode_dir)

    T = episode.length
    assert labels["expert_joint_targets"].shape == (T, 7)
    assert labels["expert_gripper_targets"].shape == (T,)
    assert labels["joint_l2_disagreement"].shape == (T,)
    assert labels["gripper_disagreement"].shape == (T,)
    assert labels["retained"].shape == (T,)
    assert labels["cube_position"].shape == (T, 3)


def test_periodic_sampling_is_always_retained(tmp_path, policy):
    """Every `sample_every`-th tick (0-indexed) must be retained regardless
    of disagreement, per the periodic-backbone component of the retention rule."""
    episode_dir, metadata = _collect(tmp_path, policy, max_steps=15, sample_every=3)
    labels = load_expert_labels(episode_dir)
    for t in range(0, 15, 3):
        assert labels["retained"][t], f"tick {t} should be periodically retained"


def test_gripper_disagreement_forces_retention(tmp_path, policy):
    """A tick with joint_l2_threshold set to infinity and sample_every huge
    should still retain any tick flagged gripper_disagreement=True."""
    episode_dir, metadata = _collect(
        tmp_path, policy, max_steps=15, sample_every=10_000, joint_l2_threshold=float("inf")
    )
    labels = load_expert_labels(episode_dir)
    for t in range(15):
        if labels["gripper_disagreement"][t]:
            assert labels["retained"][t]


def test_disagreement_values_match_recomputation_from_stored_actions(tmp_path, policy):
    """The stored joint_l2_disagreement/gripper_disagreement must be exactly
    what dagger.disagreement.compute_disagreement gives for the stored
    (model, expert) action pair -- an internal consistency check."""
    from control.action import RobotAction

    episode_dir, metadata = _collect(tmp_path, policy, max_steps=6, sample_every=2)
    episode = load_episode(episode_dir)
    labels = load_expert_labels(episode_dir)

    for t in range(episode.length):
        model_action = RobotAction(joint_targets=episode.joint_targets[t], gripper_target=float(episode.gripper_targets[t]))
        expert_action = RobotAction(
            joint_targets=labels["expert_joint_targets"][t], gripper_target=float(labels["expert_gripper_targets"][t])
        )
        recomputed = compute_disagreement(model_action, expert_action)
        assert np.isclose(recomputed["joint_l2"], labels["joint_l2_disagreement"][t])
        assert recomputed["gripper_disagreement"] == bool(labels["gripper_disagreement"][t])


def test_manifest_style_metadata_reports_counts_consistent_with_retained_array(tmp_path, policy):
    episode_dir, metadata = _collect(tmp_path, policy, max_steps=13, sample_every=4)
    labels = load_expert_labels(episode_dir)

    assert metadata["num_candidate_timesteps"] == 13
    assert metadata["num_retained_samples"] == int(labels["retained"].sum())
    assert np.isclose(metadata["retained_fraction"], labels["retained"].mean())


def test_episode_kept_even_when_collection_fails(tmp_path, policy):
    """Unlike Step 3 (expert-only dataset: failed episodes discarded by
    default), DAgger keeps every collected episode -- we want the
    off-expert/failure states specifically."""
    episode_dir, metadata = _collect(tmp_path, policy, max_steps=5, sample_every=2)
    # A random-weight tiny policy cannot plausibly solve the task in 5 ticks.
    assert metadata["collection_success"] is False
    assert episode_dir.exists()
