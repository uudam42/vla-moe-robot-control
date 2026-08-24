"""Repeated-trial reliability evaluation for the pose-controlled scripted expert.

Usage:
    python -m simulation.evaluate_grasp_reliability

Runs 10 episodes with the deterministic default cube position, then 10 more
with a small random cube x/y offset (+/- 1-2 cm), and reports reach/
orientation/lift success rates for each. This is the evidence for whether
the scripted expert is reliable enough to trust as Step 3 demonstration
supervision -- see the "measured" numbers this script prints, not
aspirational ones.
"""

import numpy as np

from simulation.environment import SimulationEnvironment
from simulation.run_robot_demo import run_episode

NUM_EPISODES = 10
RANDOM_OFFSET_METERS = 0.02  # +/- 2 cm cube x/y perturbation
SEED = 0


def summarize(label: str, results: list) -> None:
    n = len(results)
    reach_rate = sum(r.reach_success for r in results) / n
    orientation_rate = sum(r.orientation_aligned for r in results) / n
    gripper_rate = sum(r.gripper_closed for r in results) / n
    completed_rate = sum(r.controller_completed for r in results) / n
    lift_rate = sum(r.task_success for r in results) / n
    mean_lift_delta = float(np.mean([r.cube_lift_delta_m for r in results]))

    print(f"--- {label} ({n} episodes) ---")
    print(f"reach success rate:       {reach_rate:.0%}")
    print(f"orientation aligned rate: {orientation_rate:.0%}")
    print(f"gripper closed rate:      {gripper_rate:.0%}")
    print(f"controller completed rate:{completed_rate:.0%}")
    print(f"lift (task) success rate: {lift_rate:.0%}")
    print(f"mean cube lift delta:     {mean_lift_delta:.4f} m")
    print()


def main() -> None:
    rng = np.random.default_rng(SEED)

    with SimulationEnvironment() as env:
        deterministic_results = [run_episode(env) for _ in range(NUM_EPISODES)]
        summarize("Deterministic (identical initial configuration)", deterministic_results)

        perturbed_results = []
        for _ in range(NUM_EPISODES):
            offset = rng.uniform(-RANDOM_OFFSET_METERS, RANDOM_OFFSET_METERS, size=2)
            perturbed_results.append(run_episode(env, cube_xy_offset=offset))
        summarize(f"Randomized (cube x/y +/- {RANDOM_OFFSET_METERS} m)", perturbed_results)


if __name__ == "__main__":
    main()
