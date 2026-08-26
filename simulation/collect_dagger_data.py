"""Step 8 DAgger data collection entry point.

    python -m simulation.collect_dagger_data \
        --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
        --episodes 50 --xy-randomization 0.03 --seed 123 --sample-every 3 \
        --output data/dagger/round_001

Runs the Temporal Dense VLA closed-loop under ``SimulationEnvironment``
(``dagger.collector.collect_episode``): every executed action comes from
the policy; the Step 2.5 scripted-expert-derived corrective labeler
(``dagger.corrective_expert``) only computes a supervised target for each
visited state, never an action that is stepped in the simulator. Uses a
seed DISTINCT from the official evaluation seed (42) by default -- see
README "DAgger collection distribution".
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dagger.collector import DEFAULT_JOINT_L2_DISAGREEMENT_THRESHOLD, DEFAULT_SAMPLE_EVERY, collect_episode
from dataset.splits import save_splits
from models.temporal_policy import TemporalDenseVLAPolicy
from simulation.environment import SimulationEnvironment
from training.config import resolve_device

DEFAULT_OUTPUT = "data/dagger/round_001"
DEFAULT_INSTRUCTION = "Pick up the red cube."
DEFAULT_MAX_STEPS = 350
DEFAULT_SEED = 123  # distinct from the official evaluation seed 42


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--xy-randomization", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY)
    parser.add_argument("--joint-l2-disagreement-threshold", type=float, default=DEFAULT_JOINT_L2_DISAGREEMENT_THRESHOLD)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--round-name", default=None)
    args = parser.parse_args(argv)

    if args.seed == 42:
        print(
            "warning: seed=42 is the official evaluation seed -- DAgger collection "
            "should use a distinct seed to avoid collecting on official test initial states.",
        )

    device = resolve_device(args.device)
    round_name = args.round_name or Path(args.output).name
    output_root = Path(args.output)
    episodes_root = output_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)

    print("DAgger Round Collection")
    print(f"Collection policy checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}  xy-randomization: +/-{args.xy_randomization} m  seed: {args.seed}")
    print(f"Sample every: {args.sample_every} ticks  joint-L2 disagreement threshold: {args.joint_l2_disagreement_threshold}")
    print(f"Output: {output_root}")
    print()

    policy = TemporalDenseVLAPolicy.from_checkpoint(args.checkpoint, device=device)
    rng = np.random.default_rng(args.seed)

    episode_summaries = []
    with SimulationEnvironment() as env:
        for episode_id in range(args.episodes):
            offset = rng.uniform(-args.xy_randomization, args.xy_randomization, size=2)
            episode_dir, metadata = collect_episode(
                env, policy, episode_id, args.instruction, episodes_root,
                cube_xy_offset=offset, max_steps=args.max_steps,
                sample_every=args.sample_every, joint_l2_threshold=args.joint_l2_disagreement_threshold,
                extra_metadata={
                    "round": round_name, "seed": args.seed, "source_checkpoint": str(args.checkpoint),
                    "xy_randomization": args.xy_randomization,
                },
            )
            episode_summaries.append({"episode": episode_dir.name, **metadata})
            print(
                f"episode {episode_id:03d}: {'SUCCESS' if metadata['collection_success'] else 'FAIL   '} "
                f"candidates={metadata['num_candidate_timesteps']:>3} retained={metadata['num_retained_samples']:>3} "
                f"mean_joint_l2={metadata['mean_joint_l2_disagreement']:.4f} "
                f"gripper_disagree_rate={metadata['gripper_disagreement_rate']:.3f}"
            )

    save_splits(
        {"train": sorted(s["episode"] for s in episode_summaries), "val": [], "test": []},
        output_root / "splits.json",
    )

    num_candidates = sum(s["num_candidate_timesteps"] for s in episode_summaries)
    num_retained = sum(s["num_retained_samples"] for s in episode_summaries)
    num_success = sum(1 for s in episode_summaries if s["collection_success"])
    mean_joint_l2 = float(np.mean([s["mean_joint_l2_disagreement"] for s in episode_summaries])) if episode_summaries else 0.0
    mean_gripper_disagreement = float(np.mean([s["gripper_disagreement_rate"] for s in episode_summaries])) if episode_summaries else 0.0

    manifest = {
        "round": round_name,
        "source_checkpoint": str(args.checkpoint),
        "num_episodes": args.episodes,
        "xy_randomization": args.xy_randomization,
        "seed": args.seed,
        "instruction": args.instruction,
        "sample_every": args.sample_every,
        "joint_l2_disagreement_threshold": args.joint_l2_disagreement_threshold,
        "num_candidate_timesteps": num_candidates,
        "num_retained_samples": num_retained,
        "retained_fraction": (num_retained / num_candidates) if num_candidates else 0.0,
        "collection_success_rate": (num_success / len(episode_summaries)) if episode_summaries else 0.0,
        "mean_joint_l2_disagreement": mean_joint_l2,
        "mean_gripper_disagreement_rate": mean_gripper_disagreement,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "episodes": episode_summaries,
    }
    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"Episodes collected: {len(episode_summaries)}  (collection success rate {manifest['collection_success_rate']:.1%})")
    print(f"Candidate timesteps: {num_candidates}")
    print(f"Retained DAgger samples: {num_retained}  ({manifest['retained_fraction']:.1%} of candidates)")
    print(f"Mean joint-L2 disagreement: {mean_joint_l2:.4f}")
    print(f"Mean gripper disagreement rate: {mean_gripper_disagreement:.3f}")
    print()
    print(f"Saved: {output_root}/manifest.json, splits.json, episodes/*")

    return manifest


if __name__ == "__main__":
    main()
