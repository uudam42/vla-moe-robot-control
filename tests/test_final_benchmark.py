"""Tests for evaluation/final_benchmark.py -- Step 10 benchmark aggregator."""

import csv
import json

from evaluation.final_benchmark import build_policy_comparison_rows, build_safety_summary


def test_build_policy_comparison_rows_pulls_real_stored_values():
    comparison = {
        "dense": {
            "offline_test": {"joint_mae": 0.0029, "gripper_accuracy": 0.9995},
            "total_parameters": 100, "trainable_parameters": 10,
            "closed_loop": {
                "xy_0.03": {"success_rate": 0.24, "gripper_stability": {"mean_switch_count": 25.1}},
                "xy_0.04": {"success_rate": 0.18},
                "xy_0.05": {"success_rate": 0.12},
            },
        },
        "moe": None,  # not measured -- must not be fabricated
    }
    rows = build_policy_comparison_rows(comparison)
    dense_row = next(r for r in rows if r["policy"] == "Dense")
    assert dense_row["measured"] is True
    assert dense_row["offline_joint_mae_rad"] == 0.0029
    assert dense_row["success_rate_id_0.03m"] == 0.24
    assert dense_row["mean_gripper_switches_id"] == 25.1

    moe_row = next(r for r in rows if r["policy"] == "Sparse MoE")
    assert moe_row["measured"] is False
    assert "offline_joint_mae_rad" not in moe_row  # never fabricated for an unmeasured system


def test_build_safety_summary_empty_when_no_episodes(tmp_path):
    summary = build_safety_summary(tmp_path / "nonexistent")
    assert summary["episodes_recorded"] == 0
    assert summary["total_safety_interventions"] == 0
    assert summary["intervention_rate_per_tick"] is None


def test_build_safety_summary_aggregates_real_recorded_episodes(tmp_path):
    for i, (steps, interventions) in enumerate([(10, 2), (5, 0)]):
        episode_dir = tmp_path / f"episode_{i}"
        episode_dir.mkdir()
        (episode_dir / "metadata.json").write_text(json.dumps({"policy_type": "temporal", "success": i == 0}))
        (episode_dir / "metrics.json").write_text(json.dumps({
            "episode_steps": steps, "safety_intervention_count": interventions,
            "stale_observation_count": 0, "watchdog_trigger_count": 0,
        }))

    summary = build_safety_summary(tmp_path)
    assert summary["episodes_recorded"] == 2
    assert summary["total_control_ticks"] == 15
    assert summary["total_safety_interventions"] == 2
    assert summary["intervention_rate_per_tick"] == 2 / 15


def test_final_benchmark_csv_has_no_fabricated_rows_for_unmeasured_systems(tmp_path):
    from evaluation.final_benchmark import write_policy_comparison_csv

    rows = [{"policy": "Dense", "measured": False}]
    path = tmp_path / "out.csv"
    write_policy_comparison_csv(rows, path)
    with open(path) as f:
        reader = list(csv.DictReader(f))
    assert reader[0]["measured"] == "False"
    assert reader[0]["offline_joint_mae_rad"] == ""
