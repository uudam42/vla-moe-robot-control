"""DAgger data collection: the Temporal VLA drives a real closed-loop
rollout; the corrective expert only LABELS each visited state (README
"Critical runtime distinction" / "Policy controls the rollout").

Every ``env.step()`` call in this module is fed the MODEL's action. The
expert's corrective action is computed for diagnostics/storage only and is
never passed to ``env.step()`` -- see
``tests/test_dagger_expert_not_executed.py``.

On-disk format: reuses the Step 3 episode format verbatim
(``dataset.recorder.EpisodeRecorder`` / ``dataset.episode.load_episode``)
for the recorded stream -- ``states``/``joint_targets``/``gripper_targets``
in a DAgger episode are the MODEL's own observed states and issued actions,
NOT the expert's. A sibling ``expert_labels.npz`` holds the corrective
label and disagreement diagnostics for every timestep, plus which
timesteps were RETAINED as training samples (README "State sampling
frequency").
"""

from pathlib import Path

import numpy as np

from control.success import cube_lift_delta, sustained_lift_success
from dagger.corrective_expert import compute_corrective_action
from dagger.disagreement import compute_disagreement, should_retain
from dataset.recorder import EpisodeRecorder
from models.temporal_policy import TemporalDenseVLAPolicy
from simulation.environment import SimulationEnvironment

CUBE_BODY_NAME = "red_cube"
DEFAULT_MAX_STEPS = 400
DEFAULT_SAMPLE_EVERY = 3
DEFAULT_JOINT_L2_DISAGREEMENT_THRESHOLD = 0.15
EXPERT_LABELS_FILENAME = "expert_labels.npz"


def collect_episode(
    env: SimulationEnvironment,
    policy: TemporalDenseVLAPolicy,
    episode_id: int,
    instruction: str,
    episodes_root: Path,
    cube_xy_offset: np.ndarray = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    joint_l2_threshold: float = DEFAULT_JOINT_L2_DISAGREEMENT_THRESHOLD,
    extra_metadata: dict = None,
) -> tuple:
    """Run one DAgger collection episode. Returns ``(episode_dir, metadata)``.

    The rollout is driven entirely by ``policy.predict()`` (identical to
    ``evaluation.closed_loop.run_closed_loop_episode``); the corrective
    expert is consulted every tick purely to compute a label, exactly
    mirroring ``dataset/generate_dataset.py``'s "record BEFORE env.step()"
    alignment guarantee, but for the MODEL's action instead of the
    expert's.
    """
    env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    if cube_xy_offset is not None:
        cube_position = env.get_object_position(CUBE_BODY_NAME)
        cube_position[0] += cube_xy_offset[0]
        cube_position[1] += cube_xy_offset[1]
        env.set_object_position(CUBE_BODY_NAME, cube_position)

    initial_cube_position = env.get_object_position(CUBE_BODY_NAME).copy()
    initial_cube_z = float(initial_cube_position[2])
    cube_z_history = []

    recorder = EpisodeRecorder(episodes_root, episode_id)
    expert_joint_targets, expert_gripper_targets = [], []
    joint_l2_values, joint_mae_values, gripper_disagreement_flags, retained_flags = [], [], [], []
    cube_positions = []

    success = False
    step = 0
    try:
        while step < max_steps:
            observation = env.get_observation()

            # The ONLY action ever executed: the learned policy's own
            # prediction from (observation, instruction) alone.
            model_action = policy.predict(observation=observation, instruction=instruction)

            robot_state = env.get_robot_state()
            cube_position = env.get_object_position(CUBE_BODY_NAME)
            jacobian = env.get_end_effector_jacobian()

            # LABEL ONLY -- computed from privileged sim state, never
            # passed to env.step(). See module docstring.
            expert_action, phase = compute_corrective_action(robot_state, jacobian, cube_position, initial_cube_z)
            disagreement = compute_disagreement(model_action, expert_action)
            retained = should_retain(
                step, sample_every, disagreement["gripper_disagreement"], disagreement["joint_l2"], joint_l2_threshold
            )

            # Record BEFORE env.step(): `observation` is exactly what the
            # MODEL used to compute `model_action` (state_t -> action_t).
            recorder.record(observation, model_action, phase.value)
            expert_joint_targets.append(expert_action.joint_targets.copy())
            expert_gripper_targets.append(float(expert_action.gripper_target))
            joint_l2_values.append(disagreement["joint_l2"])
            joint_mae_values.append(disagreement["joint_mae"])
            gripper_disagreement_flags.append(disagreement["gripper_disagreement"])
            retained_flags.append(retained)
            cube_positions.append(cube_position.copy())

            env.step(model_action)  # <-- model action executed; expert_action is NEVER stepped
            cube_z_history.append(float(env.get_object_position(CUBE_BODY_NAME)[2]))
            step += 1

            if sustained_lift_success(cube_z_history, initial_cube_z):
                success = True
                break
    except Exception:
        recorder.abort()
        raise

    final_cube_position = env.get_object_position(CUBE_BODY_NAME).copy()
    lift_delta = cube_lift_delta(cube_z_history, initial_cube_z)
    num_retained = int(sum(retained_flags))

    metadata = {
        "collection_success": bool(success),
        "cube_xy_offset": (cube_xy_offset.tolist() if cube_xy_offset is not None else [0.0, 0.0]),
        "initial_cube_position": initial_cube_position.tolist(),
        "final_cube_position": final_cube_position.tolist(),
        "cube_lift_delta": lift_delta,
        "num_candidate_timesteps": step,
        "num_retained_samples": num_retained,
        "retained_fraction": (num_retained / step) if step else 0.0,
        "mean_joint_l2_disagreement": float(np.mean(joint_l2_values)) if joint_l2_values else 0.0,
        "gripper_disagreement_rate": float(np.mean(gripper_disagreement_flags)) if gripper_disagreement_flags else 0.0,
        **(extra_metadata or {}),
    }

    # Every DAgger episode is kept (success or not) -- unlike Step 3's
    # expert-only dataset, we WANT the off-expert / failure states.
    episode_dir = recorder.finalize(episodes_root, instruction, success, metadata)

    np.savez(
        episode_dir / EXPERT_LABELS_FILENAME,
        expert_joint_targets=np.asarray(expert_joint_targets, dtype=np.float64),
        expert_gripper_targets=np.asarray(expert_gripper_targets, dtype=np.float64),
        joint_l2_disagreement=np.asarray(joint_l2_values, dtype=np.float64),
        joint_mae_disagreement=np.asarray(joint_mae_values, dtype=np.float64),
        gripper_disagreement=np.asarray(gripper_disagreement_flags, dtype=bool),
        retained=np.asarray(retained_flags, dtype=bool),
        cube_position=np.asarray(cube_positions, dtype=np.float64),
    )

    return episode_dir, metadata


def load_expert_labels(episode_dir: Path) -> dict:
    """Load the ``expert_labels.npz`` sidecar for one DAgger episode directory."""
    with np.load(Path(episode_dir) / EXPERT_LABELS_FILENAME) as npz:
        return {name: npz[name] for name in npz.files}
