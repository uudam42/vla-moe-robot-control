"""Step 10 showcase entry point (README "Showcase Mode"): one command that
exercises the full production runtime path --

    RobotBackend -> policy.predict() -> SafetySupervisor -> RobotBackend.execute_action()

with telemetry recorded and, optionally, a video produced.

    python -m demo.run \
        --backend mujoco --policy temporal \
        --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
        --instruction "Pick up the red cube." \
        --record

``--viewer`` opens MuJoCo's interactive passive viewer if a local display
is available (README "Do not delay Step 10 excessively for graphical
polish. A reliable terminal + video workflow is acceptable." -- this repo
was built/run headless, so ``--viewer`` is best-effort and NOT verified
here; ``--record`` is the primary, verified visualization path).
"""

import argparse

import numpy as np

from robot_backend.mujoco_backend import MuJoCoBackend
from robot_backend.policy_factory import build_policy
from runtime.run_episode import run_episode
from safety.supervisor import SafetySupervisor
from telemetry.recorder import EpisodeConfig, EpisodeTelemetryRecorder
from training.config import resolve_device

DEFAULT_INSTRUCTION = "Pick up the red cube."
DEFAULT_OUTPUT_ROOT = "outputs/episodes"


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="mujoco", choices=["mujoco"], help="Only 'mujoco' is implemented this milestone.")
    parser.add_argument("--policy", required=True, choices=["dense", "moe", "temporal", "dagger"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--xy-randomization", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=350)
    parser.add_argument("--max-joint-delta", type=float, default=0.5)
    parser.add_argument("--stale-observation-sec", type=float, default=None)
    parser.add_argument("--command-timeout-sec", type=float, default=None)
    parser.add_argument("--record", action="store_true", help="Save per-tick frames + build video.mp4")
    parser.add_argument("--viewer", action="store_true", help="Best-effort interactive MuJoCo viewer; requires a local display.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    print(f"Backend: {args.backend}  Policy: {args.policy}  Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Instruction: {args.instruction!r}")

    policy = build_policy(args.policy, args.checkpoint, device=device)
    backend = MuJoCoBackend()

    viewer_handle = None
    if args.viewer:
        try:
            import mujoco.viewer

            viewer_handle = mujoco.viewer.launch_passive(backend.env.model, backend.env.data)
            print("Interactive viewer launched (best-effort; requires a local display).")
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(f"--viewer requested but could not start (headless environment?): {exc}")

    try:
        joint_range = backend.get_joint_range()
        supervisor = SafetySupervisor(
            joint_range=joint_range, max_joint_delta=args.max_joint_delta,
            stale_observation_sec=args.stale_observation_sec, command_timeout_sec=args.command_timeout_sec,
        )
        config = EpisodeConfig(
            policy_type=args.policy, checkpoint=str(args.checkpoint), instruction=args.instruction,
            seed=args.seed, cube_xy_randomization=args.xy_randomization, backend=args.backend,
            device=str(device), max_steps=args.max_steps, control_substeps=backend.env.control_substeps,
        )
        recorder = EpisodeTelemetryRecorder(config, output_root=args.output)

        rng = np.random.default_rng(args.seed)
        cube_xy_offset = rng.uniform(-args.xy_randomization, args.xy_randomization, size=2)

        metrics = run_episode(
            backend, policy, supervisor, recorder, args.instruction,
            cube_xy_offset=cube_xy_offset, max_steps=args.max_steps, save_frames=args.record,
        )
    finally:
        if viewer_handle is not None:
            viewer_handle.close()
        backend.close()

    print()
    print(f"Episode: {recorder.episode_dir}")
    print(f"Success: {metrics['success']}  Steps: {metrics['episode_steps']}  Duration: {metrics['episode_duration_sec']:.2f}s")
    if metrics.get("gripper_switch_count") is not None:
        print(f"Gripper switches: {metrics['gripper_switch_count']}")
    print(f"Safety interventions: {metrics['safety_intervention_count']}")
    if args.record:
        video_path = recorder.episode_dir / "video.mp4"
        print(f"Video: {video_path if video_path.exists() else '(imageio not available -- frames only)'}")
    print()
    print(f"Replay with: python -m tools.replay_episode {recorder.episode_dir}")

    return metrics


if __name__ == "__main__":
    main()
