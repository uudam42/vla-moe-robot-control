"""Step 5 closed-loop evaluation: N episodes under the learned Dense VLA policy.

Usage:
    python -m simulation.evaluate_vla_closed_loop \
        --checkpoint outputs/training/dense_vla_run_001/best.pt \
        --episodes 50 --xy-randomization 0.03 --seed 42

Every episode is a full closed-loop rollout (see evaluation/closed_loop.py):
Observation -> DenseVLAPolicy.predict() -> RobotAction -> env.step(), with
task success measured by the existing physical sustained-lift detector,
never by controller/model "done" state. Controller: Dense VLA Policy --
ScriptedController is not used to produce any action here.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.closed_loop import DEFAULT_MAX_STEPS, run_closed_loop_episode
from evaluation.metrics import summarize_by_instruction, summarize_results
from models.policy import DenseVLAPolicy
from simulation.environment import SimulationEnvironment
from training.config import resolve_device

DEFAULT_OUTPUT = "outputs/evaluation/dense_vla_closed_loop"
DEFAULT_INSTRUCTION = "Pick up the red cube."


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--all-instructions",
        action="store_true",
        help="Cycle through all 4 training instruction variants across episodes instead of a single --instruction.",
    )
    parser.add_argument("--xy-randomization", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-trajectories", action="store_true", help="Include per-step trajectories in episodes.jsonl (larger output).")
    parser.add_argument(
        "--smoothing-alpha", type=float, default=None,
        help="Optional EMA action smoothing weight in (0,1]; omit for the raw (unsmoothed) baseline.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    print("Controller: Dense VLA Policy")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}  xy-randomization: +/-{args.xy_randomization} m  seed: {args.seed}")

    policy = DenseVLAPolicy.from_checkpoint(args.checkpoint, device=device)
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
                env,
                policy,
                instruction,
                cube_xy_offset=offset,
                max_steps=args.max_steps,
                record_trajectory=args.save_trajectories,
                smoothing_alpha=args.smoothing_alpha,
            )
            results.append(result)
            print(
                f"episode {episode_id:03d} [{instruction[:24]:<24}] "
                f"{'SUCCESS' if result.success else 'FAIL   '} "
                f"length={result.episode_length} lift_delta={result.cube_lift_delta:.4f}m"
            )

    summary = summarize_results(results)
    by_instruction = summarize_by_instruction(results) if instructions else None

    output_root = Path(args.output) / (args.run_name or f"seed{args.seed}_xy{args.xy_randomization}")
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / "episodes.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            record = result.to_summary_dict()
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
                "smoothing_alpha": args.smoothing_alpha,
                "instruction": args.instruction if not instructions else "all_variants",
                "summary": summary,
                "by_instruction": by_instruction,
            },
            f,
            indent=2,
        )

    with open(output_root / "latency.json", "w", encoding="utf-8") as f:
        json.dump(summary["latency_ms"], f, indent=2)

    with open(output_root / "failure_counts.json", "w", encoding="utf-8") as f:
        json.dump(summary["failure_counts"], f, indent=2)

    print()
    print("Dense VLA Closed-Loop Evaluation")
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
    if by_instruction:
        print()
        print("By instruction:")
        for instruction, sub_summary in by_instruction.items():
            print(f"  {instruction:<24} {sub_summary['num_success']}/{sub_summary['num_episodes']} ({sub_summary['success_rate']:.1%})")

    print()
    print(f"Saved: {output_root}/summary.json, episodes.jsonl, latency.json, failure_counts.json")

    return summary


if __name__ == "__main__":
    main()
