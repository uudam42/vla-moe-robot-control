"""Dense vs Sparse MoE vs Temporal Dense VLA vs Temporal+DAgger comparison summary.

Usage:
    python -m evaluation.compare

Reads already-generated offline training metrics, held-out test metrics,
and closed-loop evaluation summaries for all four systems and writes one
machine-readable comparison file. Does not run any training or evaluation
itself -- run ``training.train`` / ``training.train_moe`` /
``training.train_temporal`` / ``training.train_dagger`` and the matching
``simulation.evaluate_*_closed_loop`` scripts first. Any system's columns
are simply omitted if that system hasn't been evaluated yet.
"""

import argparse
import json
from pathlib import Path

from evaluation.temporal_diagnostics import summarize_gripper_stability

DEFAULT_DENSE_TRAINING_DIR = "outputs/training/dense_vla_run_001"
DEFAULT_MOE_TRAINING_DIR = "outputs/training/moe_vla_run_001"
DEFAULT_TEMPORAL_TRAINING_DIR = "outputs/training/temporal_dense_vla_run_001"
DEFAULT_DAGGER_TRAINING_DIR = "outputs/training/temporal_dagger_run_001"
DEFAULT_DENSE_EVAL_DIR = "outputs/evaluation/dense_vla_closed_loop"
DEFAULT_MOE_EVAL_DIR = "outputs/evaluation/moe_vla_closed_loop"
DEFAULT_TEMPORAL_EVAL_DIR = "outputs/evaluation/temporal_vla_closed_loop"
DEFAULT_DAGGER_EVAL_DIR = "outputs/evaluation/temporal_dagger_vla_closed_loop"
DEFAULT_DENSE_TEST_METRICS = "outputs/evaluation/dense_test_metrics.json"
DEFAULT_MOE_TEST_METRICS = "outputs/evaluation/moe_test_metrics.json"
DEFAULT_TEMPORAL_TEST_METRICS = "outputs/evaluation/temporal_test_metrics.json"
DEFAULT_DAGGER_TEST_METRICS = "outputs/evaluation/dagger_test_metrics.json"
DEFAULT_OUTPUT = "outputs/evaluation/dense_vs_moe_vs_temporal_vs_dagger_summary.json"

XY_CONDITIONS = (0.03, 0.04, 0.05)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class _GripperOnlyResult:
    """Minimal adapter so summarize_gripper_stability (which reads
    ``.gripper_probabilities``) works on plain dicts loaded back from
    episodes.jsonl, for ANY system -- not just ones whose evaluation
    script happened to precompute gripper_stability.json."""

    def __init__(self, gripper_probabilities: list) -> None:
        self.gripper_probabilities = gripper_probabilities


def _gripper_stability_from_episodes_jsonl(episodes_path: Path) -> dict:
    if not episodes_path.exists():
        return None
    records = []
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "gripper_probabilities" in record:
                records.append(_GripperOnlyResult(record["gripper_probabilities"]))
    if not records:
        return None
    return summarize_gripper_stability(records)


def _closed_loop_by_condition(eval_dir: Path) -> dict:
    conditions = {}
    for xy in XY_CONDITIONS:
        run_dir = eval_dir / f"seed42_xy{xy}"
        summary = _load_json(run_dir / "summary.json")
        if summary is not None:
            conditions[f"xy_{xy}"] = summary["summary"]
            gripper_stability = summary.get("gripper_stability") or _gripper_stability_from_episodes_jsonl(
                run_dir / "episodes.jsonl"
            )
            if gripper_stability is not None:
                conditions[f"xy_{xy}"]["gripper_stability"] = gripper_stability
    return conditions


def _load_system(training_dir: str, eval_dir: str, test_metrics_path: str) -> dict:
    training = _load_json(Path(training_dir) / "metrics.json")
    if training is None:
        return None
    offline = _load_json(Path(test_metrics_path)) or training.get("final_val_metrics")
    return {
        "offline_test": offline,
        "total_parameters": training.get("total_parameters"),
        "trainable_parameters": training.get("trainable_parameters"),
        "active_parameters_per_token": (
            training.get("parameter_accounting", {}).get("active_parameters_per_token")
            if "parameter_accounting" in training
            else training.get("total_parameters")
        ),
        "parameter_accounting": training.get("parameter_accounting"),
        "closed_loop": _closed_loop_by_condition(Path(eval_dir)),
    }


def build_comparison(
    dense_training_dir: str = DEFAULT_DENSE_TRAINING_DIR,
    moe_training_dir: str = DEFAULT_MOE_TRAINING_DIR,
    temporal_training_dir: str = DEFAULT_TEMPORAL_TRAINING_DIR,
    dagger_training_dir: str = DEFAULT_DAGGER_TRAINING_DIR,
    dense_eval_dir: str = DEFAULT_DENSE_EVAL_DIR,
    moe_eval_dir: str = DEFAULT_MOE_EVAL_DIR,
    temporal_eval_dir: str = DEFAULT_TEMPORAL_EVAL_DIR,
    dagger_eval_dir: str = DEFAULT_DAGGER_EVAL_DIR,
    dense_test_metrics_path: str = DEFAULT_DENSE_TEST_METRICS,
    moe_test_metrics_path: str = DEFAULT_MOE_TEST_METRICS,
    temporal_test_metrics_path: str = DEFAULT_TEMPORAL_TEST_METRICS,
    dagger_test_metrics_path: str = DEFAULT_DAGGER_TEST_METRICS,
) -> dict:
    return {
        "dense": _load_system(dense_training_dir, dense_eval_dir, dense_test_metrics_path),
        "moe": _load_system(moe_training_dir, moe_eval_dir, moe_test_metrics_path),
        "temporal": _load_system(temporal_training_dir, temporal_eval_dir, temporal_test_metrics_path),
        "dagger": _load_system(dagger_training_dir, dagger_eval_dir, dagger_test_metrics_path),
    }


def print_comparison_table(comparison: dict, latency_benchmark: dict = None) -> None:
    labels = {"dense": "Dense", "moe": "MoE", "temporal": "Temporal", "dagger": "Temporal+DAgger"}
    systems = {name: comparison.get(name) for name in labels if comparison.get(name) is not None}
    columns = [name for name in ("dense", "moe", "temporal", "dagger") if name in systems]

    header = f"{'Metric':<26}" + "".join(f"{labels[c]:<20}" for c in columns)
    print(header)
    print("-" * len(header))

    def row(label, values_by_column):
        print(f"{label:<26}" + "".join(f"{str(values_by_column.get(c, '-')):<20}" for c in columns))

    if all(systems[c]["offline_test"] for c in columns):
        row("Joint MAE test (rad)", {c: f"{systems[c]['offline_test']['joint_mae']:.4f}" for c in columns})
        row("Gripper accuracy test", {c: f"{systems[c]['offline_test']['gripper_accuracy']:.4f}" for c in columns})
    row("Total params", {c: f"{systems[c]['total_parameters']:,}" if systems[c]["total_parameters"] else "-" for c in columns})
    row("Trainable params", {c: f"{systems[c]['trainable_parameters']:,}" if systems[c]["trainable_parameters"] else "-" for c in columns})
    row("Active params/token", {c: f"{systems[c]['active_parameters_per_token']:,}" if systems[c]["active_parameters_per_token"] else "-" for c in columns})
    if latency_benchmark:
        row("Batch-1 latency mean (ms)", {c: f"{latency_benchmark[c]['mean_ms']:.2f}" if c in latency_benchmark else "-" for c in columns})

    for xy in XY_CONDITIONS:
        key = f"xy_{xy}"
        values = {}
        for c in columns:
            cl = systems[c]["closed_loop"].get(key)
            values[c] = f"{cl['num_success']}/{cl['num_episodes']} ({cl['success_rate']:.1%})" if cl else "-"
        if any(v != "-" for v in values.values()):
            row(f"Success +/-{xy}m", values)

    id_key = f"xy_{XY_CONDITIONS[0]}"
    gripper_values = {}
    for c in columns:
        cl = systems[c]["closed_loop"].get(id_key)
        stability = cl.get("gripper_stability") if cl else None
        gripper_values[c] = f"{stability['mean_switch_count']:.1f}" if stability and stability.get("mean_switch_count") is not None else "-"
    if any(v != "-" for v in gripper_values.values()):
        row(f"Gripper switches +/-{XY_CONDITIONS[0]}m (mean)", gripper_values)


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-training-dir", default=DEFAULT_DENSE_TRAINING_DIR)
    parser.add_argument("--moe-training-dir", default=DEFAULT_MOE_TRAINING_DIR)
    parser.add_argument("--temporal-training-dir", default=DEFAULT_TEMPORAL_TRAINING_DIR)
    parser.add_argument("--dagger-training-dir", default=DEFAULT_DAGGER_TRAINING_DIR)
    parser.add_argument("--dense-eval-dir", default=DEFAULT_DENSE_EVAL_DIR)
    parser.add_argument("--moe-eval-dir", default=DEFAULT_MOE_EVAL_DIR)
    parser.add_argument("--temporal-eval-dir", default=DEFAULT_TEMPORAL_EVAL_DIR)
    parser.add_argument("--dagger-eval-dir", default=DEFAULT_DAGGER_EVAL_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--latency-benchmark", default="outputs/evaluation/latency_benchmark.json")
    args = parser.parse_args(argv)

    comparison = build_comparison(
        args.dense_training_dir, args.moe_training_dir, args.temporal_training_dir, args.dagger_training_dir,
        args.dense_eval_dir, args.moe_eval_dir, args.temporal_eval_dir, args.dagger_eval_dir,
    )
    latency_benchmark = _load_json(Path(args.latency_benchmark))
    if latency_benchmark:
        comparison["latency_benchmark_identical_conditions"] = latency_benchmark

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print_comparison_table(comparison, latency_benchmark)
    print()
    print(f"Saved: {output_path}")
    return comparison


if __name__ == "__main__":
    main()
