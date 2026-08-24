"""Generation smoke tests: real SimulationEnvironment, a handful of episodes.

Slower than the rest of the suite (real MuJoCo physics), but this is the
one place that exercises the actual end-to-end CLI path
(dataset.generate_dataset.main) rather than synthetic fixtures.
"""

import json

import numpy as np
import pytest

from dataset.generate_dataset import main
from dataset.loader import DemonstrationDataset
from dataset.validation import validate_dataset
from observations.robot_state import STATE_DIM


def test_generates_small_dataset_with_valid_layout(tmp_path):
    output_root = tmp_path / "demonstrations"
    main(["--episodes", "2", "--seed", "0", "--output", str(output_root)])

    successful_dirs = sorted((output_root / "successful").glob("episode_*"))
    assert len(successful_dirs) >= 1  # expert is reliable; at least one should succeed

    for episode_dir in successful_dirs:
        assert (episode_dir / "trajectory.npz").exists()
        assert (episode_dir / "metadata.json").exists()
        assert (episode_dir / "rgb").is_dir()
        assert len(list((episode_dir / "rgb").glob("*.png"))) > 0

    assert (output_root / "manifest.json").exists()
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["state_dim"] == STATE_DIM
    assert manifest["action_dim"] == 8
    assert manifest["num_successful_episodes"] == len(successful_dirs)

    assert (output_root / "splits.json").exists()


def test_generated_dataset_validates_cleanly(tmp_path):
    output_root = tmp_path / "demonstrations"
    main(["--episodes", "2", "--seed", "1", "--output", str(output_root)])

    report = validate_dataset(output_root)
    assert report["num_invalid"] == 0, report["problems"]


def test_generated_dataset_loads_and_has_no_privileged_state_leakage(tmp_path):
    output_root = tmp_path / "demonstrations"
    main(["--episodes", "2", "--seed", "2", "--output", str(output_root)])

    dataset = DemonstrationDataset(output_root)
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["state"].shape == (STATE_DIM,)  # no room for cube xyz
    assert sample["action"].shape == (8,)
    assert isinstance(sample["instruction"], str) and sample["instruction"]

    # Cube ground truth is metadata-only, never inside the trajectory state array.
    episode_dir = sorted((output_root / "successful").glob("episode_*"))[0]
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    assert "initial_cube_position" in metadata
    with np.load(episode_dir / "trajectory.npz") as npz:
        assert npz["states"].shape[1] == STATE_DIM


def test_refuses_to_overwrite_without_flag(tmp_path):
    output_root = tmp_path / "demonstrations"
    main(["--episodes", "1", "--seed", "0", "--output", str(output_root)])

    with pytest.raises(SystemExit):
        main(["--episodes", "1", "--seed", "0", "--output", str(output_root)])


def test_overwrite_flag_regenerates(tmp_path):
    output_root = tmp_path / "demonstrations"
    main(["--episodes", "1", "--seed", "0", "--output", str(output_root)])
    main(["--episodes", "1", "--seed", "0", "--output", str(output_root), "--overwrite"])
    assert (output_root / "manifest.json").exists()
