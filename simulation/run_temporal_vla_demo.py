"""Step 7 single-episode demo: the trained Temporal Dense VLA controls the robot in closed loop.

Usage:
    python -m simulation.run_temporal_vla_demo --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt

Controller: Temporal Dense VLA. Reuses the exact same
``evaluation.closed_loop.run_closed_loop_episode`` Dense/MoE demos use --
``run_closed_loop_episode`` calls ``policy.reset()`` automatically at the
start of the episode (README "Episode reset").
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.closed_loop import DEFAULT_MAX_STEPS, run_closed_loop_episode
from evaluation.temporal_diagnostics import gripper_switch_count
from models.temporal_policy import TemporalDenseVLAPolicy
from simulation.environment import SimulationEnvironment
from training.config import resolve_device

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DEFAULT_INSTRUCTION = "Pick up the red cube."
FRAME_INTERVAL = 20


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)


def main(argv: list = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--xy-randomization", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--save-frames", action="store_true", help=f"Also save a periodic frame every {FRAME_INTERVAL} ticks."
    )
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = OUTPUT_DIR / "temporal_vla_demo_frames"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print("Controller: Temporal Dense VLA")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Instruction: {args.instruction!r}")

    policy = TemporalDenseVLAPolicy.from_checkpoint(args.checkpoint, device=device)
    print(f"History length: {policy.history_length}")

    rng = np.random.default_rng(args.seed)
    cube_xy_offset = (
        rng.uniform(-args.xy_randomization, args.xy_randomization, size=2)
        if args.xy_randomization > 0
        else np.zeros(2)
    )

    def maybe_save_frame(tick: int, observation, action) -> None:
        if args.save_frames and tick % FRAME_INTERVAL == 0:
            save_rgb(observation.rgb, frames_dir / f"frame_{tick:04d}.png")

    with SimulationEnvironment() as env:
        env.reset()
        if args.xy_randomization > 0:
            cube_position = env.get_object_position("red_cube")
            cube_position[0] += cube_xy_offset[0]
            cube_position[1] += cube_xy_offset[1]
            env.set_object_position("red_cube", cube_position)

        initial_obs = env.get_observation()
        save_rgb(initial_obs.rgb, OUTPUT_DIR / "temporal_vla_demo_initial.png")
        print(f"Saved: {OUTPUT_DIR / 'temporal_vla_demo_initial.png'}")

        result = run_closed_loop_episode(
            env, policy, args.instruction, cube_xy_offset=cube_xy_offset,
            max_steps=args.max_steps, record_trajectory=True, on_step=maybe_save_frame,
        )

        final_obs = env.get_observation()
        save_rgb(final_obs.rgb, OUTPUT_DIR / "temporal_vla_demo_final.png")
        print(f"Saved: {OUTPUT_DIR / 'temporal_vla_demo_final.png'}")

    switches = gripper_switch_count(result.gripper_probabilities)

    print()
    print("Result")
    print(f"Termination reason: {result.termination_reason}")
    print(f"Episode length: {result.episode_length} ticks")
    print(f"Initial cube position: {np.round(result.initial_cube_position, 4)}")
    print(f"Final cube position:   {np.round(result.final_cube_position, 4)}")
    print(f"Cube lift delta: {result.cube_lift_delta:.4f} m")
    print(f"Task success (physical lift): {result.success}")
    print(f"Gripper open<->closed switches: {switches}")
    print(f"Mean inference latency: {np.mean(result.inference_latencies_ms):.2f} ms")
    if args.save_frames:
        print(f"Periodic frames: {frames_dir}/frame_*.png")
    print()
    print(
        "Note: this is a single closed-loop rollout under the learned Temporal Dense VLA "
        "policy. See README for the Dense-vs-Temporal gripper-oscillation comparison."
    )


if __name__ == "__main__":
    main()
