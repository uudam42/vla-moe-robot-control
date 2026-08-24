"""Demo runner for the Milestone 1 MuJoCo observation pipeline.

Usage:
    python -m simulation.run_simulation

Initializes the simulation, saves an initial observation image, advances
the simulation state with a small joint-space nudge, saves a second
observation image, and prints information demonstrating that the state
changed.

Superseded by Step 2 (see simulation/run_robot_demo.py): Observation.state
now carries robot proprioception (not raw qpos/qvel), and step() now takes
a RobotAction rather than the old cube-nudging signature -- this script is
kept working against that new API rather than the original Milestone 1 one.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from control.action import RobotAction
from observations.observation import Observation
from simulation.environment import SimulationEnvironment

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    """Save an RGB uint8 array as a PNG image."""
    Image.fromarray(rgb, mode="RGB").save(path)


def describe(label: str, obs: Observation) -> None:
    """Print shape/dtype/state/timestamp info for an observation."""
    print(label)
    print(f"RGB: {obs.rgb.shape}, {obs.rgb.dtype}")
    print(f"State: {obs.state.shape}")
    print(f"Simulation time: {obs.timestamp:.3f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SimulationEnvironment() as env:
        obs_initial = env.get_observation()
        describe("Initial observation", obs_initial)
        initial_path = OUTPUT_DIR / "observation_initial.png"
        save_rgb(obs_initial.rgb, initial_path)
        print(f"Saved: {initial_path}")

        print()
        robot_state = env.get_robot_state()
        nudge = np.zeros(7)
        nudge[0] = 0.2  # small joint-1 move, enough to change state visibly
        env.step(RobotAction(joint_targets=robot_state.joint_positions + nudge, gripper_target=1.0))
        obs_after = env.get_observation()

        state_changed = not np.allclose(obs_initial.state, obs_after.state)

        print("After simulation change")
        print(f"RGB: {obs_after.rgb.shape}, {obs_after.rgb.dtype}")
        print(f"State changed: {state_changed}")
        print(f"Simulation time: {obs_after.timestamp:.3f}")

        after_path = OUTPUT_DIR / "observation_after_step.png"
        save_rgb(obs_after.rgb, after_path)
        print(f"Saved: {after_path}")


if __name__ == "__main__":
    main()
