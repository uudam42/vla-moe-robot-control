"""README "ROS2 Benchmark" (partial): measures direct
(``SimulationEnvironment``) vs. ``RobotBackend``-mediated closed-loop
latency for the SAME checkpoint/seed, to isolate whatever overhead the
``RobotBackend`` abstraction itself adds -- separate from, and a
prerequisite building block for, the full direct-vs-ROS2 comparison
(README "Do not conflate inference and transport latency").

    python -m evaluation.backend_benchmark \
        --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt --episodes 5

This does NOT measure ROS2 message-transport latency -- no ROS2
distribution is installed in this environment (see README "Step 9"
limitations); that column of the direct-vs-ROS2 comparison is honestly
reported as not measured here, rather than fabricated.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.closed_loop import run_closed_loop_episode
from evaluation.metrics import latency_stats
from models.temporal_policy import TemporalDenseVLAPolicy
from robot_backend.backend_closed_loop import run_closed_loop_episode_via_backend
from robot_backend.mujoco_backend import MuJoCoBackend
from simulation.environment import SimulationEnvironment
from training.config import resolve_device

DEFAULT_OUTPUT = "outputs/evaluation/ros2/direct_vs_backend_latency.json"


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    offsets = [rng.uniform(-0.03, 0.03, size=2) for _ in range(args.episodes)]

    direct_policy = TemporalDenseVLAPolicy.from_checkpoint(args.checkpoint, device=device)
    direct_inference_ms, direct_step_ms = [], []
    with SimulationEnvironment() as env:
        for offset in offsets:
            result = run_closed_loop_episode(
                env, direct_policy, "Pick up the red cube.", cube_xy_offset=offset, max_steps=args.max_steps
            )
            direct_inference_ms.extend(result.inference_latencies_ms)
            direct_step_ms.extend(result.env_step_latencies_ms)

    backend_policy = TemporalDenseVLAPolicy.from_checkpoint(args.checkpoint, device=device)
    backend_inference_ms, backend_step_ms = [], []
    with MuJoCoBackend() as backend:
        for offset in offsets:
            result = run_closed_loop_episode_via_backend(
                backend, backend_policy, "Pick up the red cube.", cube_xy_offset=offset, max_steps=args.max_steps
            )
            backend_inference_ms.extend(result.inference_latencies_ms)
            backend_step_ms.extend(result.env_step_latencies_ms)

    summary = {
        "note": (
            "Direct (SimulationEnvironment) vs RobotBackend-mediated latency only. "
            "ROS2 message-transport latency was NOT measured -- no ROS2 distribution "
            "is installed in this environment (see README 'Step 9' limitations)."
        ),
        "episodes": args.episodes,
        "device": str(device),
        "direct": {
            "inference_latency_ms": latency_stats(direct_inference_ms),
            "backend_execute_latency_ms": latency_stats(direct_step_ms),
        },
        "via_robot_backend": {
            "inference_latency_ms": latency_stats(backend_inference_ms),
            "backend_execute_latency_ms": latency_stats(backend_step_ms),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print()
    print(f"Saved: {output_path}")
    return summary


if __name__ == "__main__":
    main()
