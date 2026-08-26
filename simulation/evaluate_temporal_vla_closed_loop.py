"""Step 7 closed-loop evaluation: N episodes under the learned Temporal Dense VLA policy.

Usage:
    python -m simulation.evaluate_temporal_vla_closed_loop \
        --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
        --episodes 50 --xy-randomization 0.03 --seed 42

Reuses the exact same closed-loop rollout and metrics/failure-taxonomy
code as Dense/MoE (``simulation/evaluate_vla_closed_loop.py`` /
``evaluate_moe_vla_closed_loop.py``) -- only the policy class differs.
Given the same ``--seed``/``--episodes``/``--xy-randomization``, the cube
offset sequence is identical to the Dense/MoE evaluations' (same RNG call
order), so all three models are evaluated on the same initial cube
positions. Adds gripper-switch (oscillation) statistics -- the central
Step 7 measurement -- to the standard summary.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.closed_loop import DEFAULT_MAX_STEPS, run_closed_loop_episode
from evaluation.metrics import summarize_by_instruction, summarize_results
from evaluation.temporal_diagnostics import summarize_gripper_stability
from models.temporal_policy import TemporalDenseVLAPolicy
from simulation.environment import SimulationEnvironment
from training.config import resolve_device

DEFAULT_OUTPUT = "outputs/evaluation/temporal_vla_closed_loop"
DEFAULT_INSTRUCTION = "Pick up the red cube."


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--all-instructions", action="store_true")
    parser.add_argument("--xy-randomization", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    print("Controller: Temporal Dense VLA")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}  xy-randomization: +/-{args.xy_randomization} m  seed: {args.seed}")

    policy = TemporalDenseVLAPolicy.from_checkpoint(args.checkpoint, device=device)
    print(f"History length: {policy.history_length}")
    rng = np.random.default_rng(args.seed)

    instructions = (
        ["Pick up the red cube.", "Grasp the red cube.", "Lift the red cube.", "Pick up the red block."]
        if args.all_instructions
        else None
    )

    results = []
    with SimulationEnvironment() as env:
        for episode_id in range(args.episodes):
            instruction = instructions[episode_id % len(instructions)] if instructions else args.instruction
            offset = rng.uniform(-args.xy_randomization, args.xy_randomization, size=2)

            result = run_closed_loop_episode(
                env, policy, instruction, cube_xy_offset=offset, max_steps=args.max_steps,
                record_trajectory=args.save_trajectories,
            )
            results.append(result)
            print(
                f"episode {episode_id:03d} [{instruction[:24]:<24}] "
                f"{'SUCCESS' if result.success else 'FAIL   '} "
                f"length={result.episode_length} lift_delta={result.cube_lift_delta:.4f}m"
            )

    summary = summarize_results(results)
    by_instruction = summarize_by_instruction(results) if instructions else None
    gripper_stability = summarize_gripper_stability(results)

    output_root = Path(args.output) / (args.run_name or f"seed{args.seed}_xy{args.xy_randomization}")
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / "episodes.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            record = result.to_summary_dict()
            record["gripper_switch_count"] = int(
                np.sum(np.diff((np.array(result.gripper_probabilities) >= 0.5).astype(int)) != 0)
            ) if len(result.gripper_probabilities) > 1 else 0
            if args.save_trajectories:
                record["steps"] = [s.__dict__ for s in result.steps]
            f.write(json.dumps(record) + "\n")

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "episodes": args.episodes,
                "xy_randomization": args.xy_randomization,
                "seed": args.seed,
                "history_length": policy.history_length,
                "instruction": args.instruction if not instructions else "all_variants",
                "summary": summary,
                "by_instruction": by_instruction,
                "gripper_stability": gripper_stability,
            },
            f,
            indent=2,
        )
    with open(output_root / "latency.json", "w", encoding="utf-8") as f:
        json.dump(summary["latency_ms"], f, indent=2)
    with open(output_root / "failure_counts.json", "w", encoding="utf-8") as f:
        json.dump(summary["failure_counts"], f, indent=2)
    with open(output_root / "gripper_stability.json", "w", encoding="utf-8") as f:
        json.dump(gripper_stability, f, indent=2)

    print()
    print("Temporal Dense VLA Closed-Loop Evaluation")
    print()
    print(f"Distribution: cube XY +/-{args.xy_randomization} m")
    print(f"Episodes: {summary['num_episodes']}")
    print()
    print(f"Success: {summary['num_success']} / {summary['num_episodes']}  ({summary['success_rate']:.1%})")
    if summary["mean_successful_episode_length"] is not None:
        print(f"Mean successful episode length: {summary['mean_successful_episode_length']:.1f} ticks")
    print()
    if summary["failure_counts"]:
        print("Failures:")
        for category, count in sorted(summary["failure_counts"].items(), key=lambda kv: -kv[1]):
            print(f"  {count:>2}  {category}")
        print()
    latency = summary["latency_ms"]
    print(f"Inference latency: mean {latency['mean_ms']:.2f}ms  p50 {latency['p50_ms']:.2f}ms  p95 {latency['p95_ms']:.2f}ms")
    if summary["effective_control_hz"]:
        print(f"Effective control rate: {summary['effective_control_hz']:.1f} Hz")
    print()
    print(f"Gripper switch count: mean {gripper_stability['mean_switch_count']:.1f}  median {gripper_stability['median_switch_count']:.1f}  max {gripper_stability['max_switch_count']}")
    if by_instruction:
        print()
        print("By instruction:")
        for instruction, sub_summary in by_instruction.items():
            print(f"  {instruction:<24} {sub_summary['num_success']}/{sub_summary['num_episodes']} ({sub_summary['success_rate']:.1%})")

    print()
    print(f"Saved: {output_root}/summary.json, episodes.jsonl, latency.json, failure_counts.json, gripper_stability.json")

    return summary


if __name__ == "__main__":
    main()
