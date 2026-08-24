"""Step 3 dataset generation entry point.

Usage:
    python -m dataset.generate_dataset --episodes 10 --seed 42 --output data/demonstrations

Runs the Step 2.5 scripted expert closed-loop under ``SimulationEnvironment``,
recording (RGB, 23D RobotState, instruction) -> RobotAction at every control
tick via ``EpisodeRecorder`` (see that module for the critical
observation-before-``env.step()`` alignment guarantee). Only episodes that
pass the existing physical success detector
(``control.success.sustained_lift_success`` -- NOT merely
``controller.done``) are kept as official training data, under
``<output>/successful/``. This dataset is generated entirely from a MuJoCo
scripted expert, not a physical robot.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from control.scripted_controller import ScriptedController
from control.success import cube_lift_delta, sustained_lift_success
from dataset import DATASET_VERSION, INSTRUCTION_VARIANTS
from dataset.recorder import EpisodeRecorder
from dataset.splits import make_splits, save_splits
from observations.robot_state import STATE_DIM
from simulation.environment import DEFAULT_CAMERA_NAME, DEFAULT_HEIGHT, DEFAULT_WIDTH, SimulationEnvironment

DEFAULT_MAX_STEPS = 400
DEFAULT_XY_RANDOMIZATION = 0.03
CUBE_BODY_NAME = "red_cube"
TASK_NAME = "pick_red_cube"


def generate_episode(
    env: SimulationEnvironment,
    episode_id: int,
    rng: np.random.Generator,
    xy_randomization: float,
    max_steps: int,
    staging_root: Path,
    seed: int,
) -> tuple:
    """Run one closed-loop expert rollout and record it. Returns (recorder, success, metadata)."""
    env.reset()
    base_cube_position = env.get_object_position(CUBE_BODY_NAME)
    xy_offset = rng.uniform(-xy_randomization, xy_randomization, size=2)
    perturbed_position = base_cube_position.copy()
    perturbed_position[0] += xy_offset[0]
    perturbed_position[1] += xy_offset[1]
    env.set_object_position(CUBE_BODY_NAME, perturbed_position)

    instruction = str(rng.choice(INSTRUCTION_VARIANTS))

    initial_state = env.get_robot_state()
    controller = ScriptedController(home_joint_positions=initial_state.joint_positions)

    initial_cube_z = float(env.get_object_position(CUBE_BODY_NAME)[2])
    cube_z_history = []

    recorder = EpisodeRecorder(staging_root, episode_id)

    step = 0
    try:
        while not controller.done and step < max_steps:
            observation = env.get_observation()
            robot_state = env.get_robot_state()
            cube_position = env.get_object_position(CUBE_BODY_NAME)
            jacobian = env.get_end_effector_jacobian()

            action = controller.compute_action(robot_state, jacobian, cube_position)

            # Record BEFORE env.step(): `observation` is exactly what the
            # expert used to compute `action` (state_t -> action_t). Do not
            # reorder this -- see dataset/recorder.py and
            # tests/test_dataset_recorder.py::test_alignment_*.
            recorder.record(observation, action, controller.stage.value)

            env.step(action)
            cube_z_history.append(float(env.get_object_position(CUBE_BODY_NAME)[2]))
            step += 1
    except Exception:
        recorder.abort()
        raise

    lift_delta = cube_lift_delta(cube_z_history, initial_cube_z)
    task_success = sustained_lift_success(cube_z_history, initial_cube_z)
    final_cube_position = env.get_object_position(CUBE_BODY_NAME)

    if task_success:
        termination_reason = "success"
    elif step >= max_steps:
        termination_reason = "timeout"
    elif controller.done:
        termination_reason = "controller_done_without_success"
    else:
        termination_reason = "controller_failure"

    metadata = {
        "instruction": instruction,
        "success": task_success,
        "task": TASK_NAME,
        "seed": seed,
        "scene": "simulation/scene.xml",
        "camera_name": DEFAULT_CAMERA_NAME,
        "control_substeps": env.control_substeps,
        "image_width": DEFAULT_WIDTH,
        "image_height": DEFAULT_HEIGHT,
        "cube_xy_randomization": xy_randomization,
        "cube_xy_offset": xy_offset.tolist(),
        "initial_cube_position": perturbed_position.tolist(),
        "final_cube_position": final_cube_position.tolist(),
        "cube_lift_delta": lift_delta,
        "termination_reason": termination_reason,
        "controller_completed": bool(controller.done),
        "final_controller_stage": controller.stage.value,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return recorder, task_success, metadata


def _fail_if_nonempty(output_root: Path, overwrite: bool) -> None:
    existing = [
        p
        for subdir in ("successful", "failed")
        for p in (output_root / subdir).glob("episode_*")
        if (output_root / subdir).exists()
    ]
    if existing and not overwrite:
        print(
            f"error: {output_root} already contains {len(existing)} episode(s). "
            "Pass --overwrite to regenerate, or choose a different --output.",
            file=sys.stderr,
        )
        sys.exit(1)
    if existing and overwrite:
        for subdir in ("successful", "failed"):
            shutil.rmtree(output_root / subdir, ignore_errors=True)
        for filename in ("manifest.json", "splits.json"):
            (output_root / filename).unlink(missing_ok=True)


def _print_statistics(results: list, successful_root: Path) -> None:
    num_requested = len(results)
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    num_success = len(successes)

    print(f"Requested episodes: {num_requested}")
    print(f"Successful episodes: {num_success}")
    print(f"Failed episodes: {len(failures)}")
    if num_requested:
        print(f"Success rate: {100.0 * num_success / num_requested:.1f}%")
    print()

    if successes:
        lengths = [r["episode_length"] for r in successes]
        total = sum(lengths)
        print(f"Total timesteps: {total:,}")
        print(f"Mean episode length: {np.mean(lengths):.1f}")
        print(f"Min episode length: {min(lengths)}")
        print(f"Max episode length: {max(lengths)}")
        print()

        print("Instruction variants:")
        counts = {}
        for r in successes:
            counts[r["instruction"]] = counts.get(r["instruction"], 0) + 1
        for instruction, count in sorted(counts.items()):
            print(f"  {instruction:<24} {count}")
        print()

    if failures:
        print("Failure reasons:")
        reason_counts = {}
        for r in failures:
            reason_counts[r["termination_reason"]] = reason_counts.get(r["termination_reason"], 0) + 1
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason:<28} {count}")
        print()

    print(f"State dimension: {STATE_DIM}")
    print("Action dimension: 8")
    print(f"RGB resolution: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
    if successful_root.exists():
        disk_bytes = sum(f.stat().st_size for f in successful_root.rglob("*") if f.is_file())
        print(f"Dataset disk size: {disk_bytes / 1e6:.1f} MB")


def main(argv: list = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to attempt.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/demonstrations"))
    parser.add_argument("--xy-randomization", type=float, default=DEFAULT_XY_RANDOMIZATION)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--save-failed", action="store_true", help="Keep failed episodes under <output>/failed/ instead of discarding them.")
    parser.add_argument("--overwrite", action="store_true", help="Delete any existing dataset under --output before generating.")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args(argv)

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    _fail_if_nonempty(output_root, args.overwrite)

    successful_root = output_root / "successful"
    failed_root = output_root / "failed"

    rng = np.random.default_rng(args.seed)
    results = []

    with SimulationEnvironment() as env:
        for episode_id in range(args.episodes):
            recorder, success, metadata = generate_episode(
                env,
                episode_id,
                rng,
                args.xy_randomization,
                args.max_steps,
                staging_root=output_root,
                seed=args.seed,
            )
            episode_length = recorder.num_steps

            if success:
                recorder.finalize(successful_root, metadata["instruction"], True, metadata)
            elif args.save_failed:
                recorder.finalize(failed_root, metadata["instruction"], False, metadata)
            else:
                recorder.abort()

            results.append({**metadata, "episode_length": episode_length})
            print(
                f"episode {episode_id:06d}: "
                f"{'SUCCESS' if success else 'FAIL (' + metadata['termination_reason'] + ')'} "
                f"length={episode_length} lift_delta={metadata['cube_lift_delta']:.4f}m"
            )

    print()
    _print_statistics(results, successful_root)

    successful_names = sorted(p.name for p in successful_root.glob("episode_*")) if successful_root.exists() else []
    if successful_names:
        manifest = {
            "dataset_version": DATASET_VERSION,
            "num_successful_episodes": len(successful_names),
            "num_samples": sum(r["episode_length"] for r in results if r["success"]),
            "state_dim": STATE_DIM,
            "action_dim": 8,
            "image_width": DEFAULT_WIDTH,
            "image_height": DEFAULT_HEIGHT,
            "task": TASK_NAME,
            "seed": args.seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        splits = make_splits(successful_names, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
        save_splits(splits, output_root / "splits.json")
        print()
        print(
            f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
        )
    else:
        print()
        print("No successful episodes -- skipping manifest.json/splits.json.")


if __name__ == "__main__":
    main()
