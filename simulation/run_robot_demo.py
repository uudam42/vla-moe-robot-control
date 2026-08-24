"""Demo runner for the Step 2.5 pose-controlled scripted pick sequence.

Usage:
    python -m simulation.run_robot_demo

Resets the robot arm and cube, runs the closed-loop scripted controller
(HOME -> ABOVE_CUBE -> DESCEND -> CLOSE_GRIPPER -> LIFT -> DONE) with pose
(position + orientation) IK, logs each control tick, saves before/after
camera frames, and reports reach/orientation/grasp/lift results.

Controller completion (reaching Stage.DONE) is deliberately reported
separately from task success (control.success.sustained_lift_success) --
see the "Result summary" block at the end.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from control.scripted_controller import ORIENTATION_TOLERANCE, POSITION_TOLERANCE, ScriptedController, Stage
from control.success import cube_lift_delta, sustained_lift_success
from simulation.environment import SimulationEnvironment

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
MAX_CONTROL_STEPS = 400
LOG_EVERY = 10  # avoid printing every single tick


@dataclass
class EpisodeResult:
    controller_completed: bool
    reach_success: bool
    orientation_aligned: bool
    gripper_closed: bool
    cube_lift_delta_m: float
    task_success: bool
    final_stage: str
    steps_taken: int
    best_position_error_m: float
    best_orientation_error_deg: float
    cube_z_history: list = field(default_factory=list, repr=False)


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    """Save an RGB uint8 array as a PNG image."""
    Image.fromarray(rgb, mode="RGB").save(path)


def log_tick(step: int, controller: ScriptedController, robot_state, cube_position) -> None:
    eef = robot_state.end_effector_position
    target = controller.last_target_position
    target_str = (
        f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]" if target is not None else "-"
    )
    pos_err = controller.last_position_error
    pos_err_str = f"{pos_err:.3f}" if pos_err is not None else "-"
    orient_err = controller.last_orientation_error_deg
    orient_err_str = f"{orient_err:.1f}" if orient_err is not None else "-"
    gripper_label = (
        "CLOSED" if controller.stage in (Stage.CLOSE_GRIPPER, Stage.LIFT, Stage.DONE) else "OPEN"
    )
    print(
        f"step={step} stage={controller.stage.value} "
        f"eef=[{eef[0]:.3f}, {eef[1]:.3f}, {eef[2]:.3f}] target={target_str} "
        f"position_error={pos_err_str} orientation_error_deg={orient_err_str} "
        f"gripper={gripper_label} "
        f"cube=[{cube_position[0]:.3f}, {cube_position[1]:.3f}, {cube_position[2]:.3f}]"
    )


def run_episode(
    env: SimulationEnvironment,
    cube_xy_offset: np.ndarray = None,
    max_steps: int = MAX_CONTROL_STEPS,
    verbose: bool = False,
) -> EpisodeResult:
    """Reset, optionally perturb the cube, and run one full pick episode.

    Args:
        env: An open SimulationEnvironment.
        cube_xy_offset: Optional ``(2,)`` array added to the cube's default
            x/y position before the episode starts (for the reliability
            sweep's small randomization); ``None`` leaves the cube at its
            deterministic default position.
        max_steps: Episode timeout in control ticks.
        verbose: Print per-tick diagnostics.
    """
    env.reset()
    if cube_xy_offset is not None:
        cube_position = env.get_object_position("red_cube")
        cube_position[0] += cube_xy_offset[0]
        cube_position[1] += cube_xy_offset[1]
        env.set_object_position("red_cube", cube_position)

    initial_state = env.get_robot_state()
    controller = ScriptedController(home_joint_positions=initial_state.joint_positions)

    initial_cube_z = float(env.get_object_position("red_cube")[2])
    cube_z_history = []
    best_position_error = float("inf")
    best_orientation_error_deg = float("inf")
    reach_success = False
    orientation_aligned = False
    gripper_closed = False

    step = 0
    while not controller.done and step < max_steps:
        robot_state = env.get_robot_state()
        cube_position = env.get_object_position("red_cube")
        jacobian = env.get_end_effector_jacobian()

        action = controller.compute_action(robot_state, jacobian, cube_position)
        env.step(action)

        if controller.last_position_error is not None:
            best_position_error = min(best_position_error, controller.last_position_error)
            if controller.last_position_error < POSITION_TOLERANCE:
                reach_success = True
        if controller.last_orientation_error_deg is not None:
            orientation_error_rad = np.radians(controller.last_orientation_error_deg)
            best_orientation_error_deg = min(
                best_orientation_error_deg, controller.last_orientation_error_deg
            )
            if orientation_error_rad < ORIENTATION_TOLERANCE:
                orientation_aligned = True
        if controller.stage in (Stage.CLOSE_GRIPPER, Stage.LIFT, Stage.DONE):
            gripper_closed = True

        cube_z_history.append(float(env.get_object_position("red_cube")[2]))

        if verbose and step % LOG_EVERY == 0:
            log_tick(step, controller, robot_state, cube_position)
        step += 1

    if verbose:
        final_state = env.get_robot_state()
        log_tick(step, controller, final_state, env.get_object_position("red_cube"))

    lift_delta = cube_lift_delta(cube_z_history, initial_cube_z)
    task_success = sustained_lift_success(cube_z_history, initial_cube_z)

    return EpisodeResult(
        controller_completed=controller.done,
        reach_success=reach_success,
        orientation_aligned=orientation_aligned,
        gripper_closed=gripper_closed,
        cube_lift_delta_m=lift_delta,
        task_success=task_success,
        final_stage=controller.stage.value,
        steps_taken=step,
        best_position_error_m=best_position_error,
        best_orientation_error_deg=best_orientation_error_deg,
        cube_z_history=cube_z_history,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SimulationEnvironment() as env:
        env.reset()
        initial_obs = env.get_observation()
        initial_path = OUTPUT_DIR / "robot_observation_initial.png"
        save_rgb(initial_obs.rgb, initial_path)
        print("Initial robot state")
        print(f"joint_positions: {env.get_robot_state().joint_positions}")
        print(f"Saved: {initial_path}")
        print()

        result = run_episode(env, verbose=True)

        final_obs = env.get_observation()
        final_path = OUTPUT_DIR / "robot_observation_final.png"
        save_rgb(final_obs.rgb, final_path)
        print(f"\nSaved: {final_path}")

        print()
        print("Result summary")
        print(f"Controller completed: {result.controller_completed}")
        print(f"Reach success: {result.reach_success} (best position error: {result.best_position_error_m:.4f} m)")
        print(f"Orientation aligned: {result.orientation_aligned} (best orientation error: {result.best_orientation_error_deg:.1f} deg)")
        print(f"Gripper closed: {result.gripper_closed}")
        print(f"Cube lift delta: {result.cube_lift_delta_m:.4f} m")
        print(f"Task success: {result.task_success}")


if __name__ == "__main__":
    main()
