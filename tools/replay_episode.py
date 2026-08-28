"""Step 10 episode replay (README "Replay" / "Replay Integrity").

    python -m tools.replay_episode outputs/episodes/<episode>

Replays a RECORDED rollout from its ``telemetry.jsonl`` -- it answers
*"what actually happened during that recorded rollout?"*, never *"what
would the current model predict now?"*. **The VLA policy is never loaded
or called by this tool** (verified by
``tests/test_replay_episode.py::test_replay_never_imports_a_policy_class``);
everything printed comes from the archived telemetry/metadata/metrics
files written by ``telemetry.recorder.EpisodeTelemetryRecorder``.
"""

import argparse
import json
from pathlib import Path


def load_episode(episode_dir: Path) -> dict:
    episode_dir = Path(episode_dir)
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    metrics = json.loads((episode_dir / "metrics.json").read_text())
    steps = []
    with open(episode_dir / "telemetry.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return {"metadata": metadata, "metrics": metrics, "steps": steps, "episode_dir": episode_dir}


def _format_step(step: dict, frames_dir: Path) -> str:
    executed = step.get("executed_action")
    gripper = f"{step['gripper_command']:.3f}" if step.get("gripper_command") is not None else "-"
    latency_ms = step["inference_latency_sec"] * 1000.0
    safety = f"{step['safety_decision']}/{step['safety_reason']}"
    frame_path = frames_dir / f"{step['step_id']:06d}.png"
    frame_note = str(frame_path) if frame_path.exists() else "-"
    joint_str = "[" + ", ".join(f"{v:.3f}" for v in executed["joint_targets"]) + "]" if executed else "(none -- rejected)"
    return (
        f"step {step['step_id']:04d}  action={joint_str} gripper={gripper}  "
        f"latency={latency_ms:.2f}ms  safety={safety}  frame={frame_note}"
    )


def replay(episode_dir: Path, limit: int = None, as_json: bool = False) -> dict:
    episode = load_episode(episode_dir)
    frames_dir = episode["episode_dir"] / "frames"

    if as_json:
        print(json.dumps(episode, indent=2, default=str))
        return episode

    metadata, metrics = episode["metadata"], episode["metrics"]
    print(f"Episode: {episode_dir}")
    print(f"Policy: {metadata['policy_type']}  Checkpoint: {metadata['checkpoint']}")
    print(f"Instruction: {metadata['instruction']!r}")
    print(f"Backend: {metadata['backend']}  Device: {metadata['device']}  Seed: {metadata['seed']}")
    print(f"Started: {metadata['start_time']}  Ended: {metadata['end_time']}")
    print()

    steps = episode["steps"]
    shown = steps if limit is None else steps[:limit]
    for step in shown:
        print(_format_step(step, frames_dir))
    if limit is not None and len(steps) > limit:
        print(f"... ({len(steps) - limit} more steps not shown; use --limit to show more)")

    print()
    print(f"Success: {metadata['success']}  Failure reason: {metadata.get('failure_reason')}")
    print(f"Episode steps: {metrics['episode_steps']}  Duration: {metrics['episode_duration_sec']:.2f}s")
    if metrics.get("gripper_switch_count") is not None:
        print(f"Gripper switches: {metrics['gripper_switch_count']}")
    if metrics["inference_latency_ms"]["mean_ms"] is not None:
        lat = metrics["inference_latency_ms"]
        print(f"Inference latency: mean {lat['mean_ms']:.2f}ms  p50 {lat['p50_ms']:.2f}ms  p95 {lat['p95_ms']:.2f}ms")
    print(
        f"Safety interventions: {metrics['safety_intervention_count']}  "
        f"Stale observations: {metrics['stale_observation_count']}  "
        f"Watchdog triggers: {metrics['watchdog_trigger_count']}"
    )
    if (episode["episode_dir"] / "video.mp4").exists():
        print(f"Video: {episode['episode_dir'] / 'video.mp4'}")

    return episode


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode_dir", help="Path to an outputs/episodes/episode_<timestamp>/ directory")
    parser.add_argument("--limit", type=int, default=50, help="Max steps to print (default 50; use 0 for all)")
    parser.add_argument("--json", action="store_true", help="Dump the full episode as JSON instead of a pretty summary")
    args = parser.parse_args(argv)

    limit = None if args.limit == 0 else args.limit
    return replay(Path(args.episode_dir), limit=limit, as_json=args.json)


if __name__ == "__main__":
    main()
