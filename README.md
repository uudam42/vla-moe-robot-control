# Vision-Language-Action Robot Control with Mixture-of-Experts

## Current milestone

**Step 3 — Demonstration Dataset Generation**

Step 2.5 made the scripted expert reliable (100% reach/orientation/lift
success across repeated trials, see below). Step 3 turns that reliability
into a dataset: it runs many closed-loop expert episodes and records, at
every control timestep, exactly what a future imitation-learning policy
would need to reproduce the expert's behavior.

```text
MuJoCo
   |
   v
Observation (RGB + 23D RobotState)
   |
   v
Scripted Expert (privileged: cube ground truth + Jacobian)
   |
   v
RobotAction
   |
   v
EpisodeRecorder  --records the observation BEFORE this action is applied--
   |
   v
Dataset (data/demonstrations/)
```

This dataset is generated entirely from a **MuJoCo scripted expert**, not
from a physical robot -- every image and action in it comes from
`control/scripted_controller.py` driving the simulation in
`simulation/environment.py`.

### Future step (not implemented yet)

Step 4 will consume this dataset:

```text
RGB + 23D RobotState + Language Instruction
                |
                v
          Dense VLA Policy
                |
                v
          8D RobotAction
```

### Dataset schema

```text
Observation (per timestep):
  RGB:   H x W x 3 uint8 PNG (lossless), current default 640x480
  state: 23 float64 values (RobotState.as_vector() -- robot
         proprioception only; no cube ground truth)

Instruction (per episode, shared by every timestep in it):
  UTF-8 string, one of 4 semantically-identical variants (see below)

Action (per timestep, = exactly what RobotAction the expert produced):
  joint_targets:   7 float64 (radians)
  gripper_target:  1 float64 in [0, 1]
  (RobotAction.as_vector() concatenates these into shape (8,))
```

The supervised task Step 4 will learn is `(RGB, state, instruction) ->
action`. Privileged expert-only information -- cube ground-truth XYZ,
the Jacobian, controller internal targets -- is stored as episode
*metadata* (`metadata.json`) or trajectory *diagnostics*
(`controller_stage`/`eef_positions`/`eef_orientations` in
`trajectory.npz`), never inside `state`. A regression test
(`tests/test_simulation.py::test_observation_state_unaffected_by_cube_position`,
plus a dataset-level check in `tests/test_dataset_generation.py`) enforces
this boundary.

### On-disk layout

```text
data/demonstrations/
├── successful/
│   └── episode_000000/
│       ├── rgb/000000.png, 000001.png, ...
│       ├── trajectory.npz   (states, joint_targets, gripper_targets,
│       │                     timestamps, + diagnostics: controller_stage,
│       │                     eef_positions, eef_orientations)
│       └── metadata.json    (instruction, success, cube positions,
│                              lift delta, seed, termination_reason, ...)
├── failed/                  (only with --save-failed)
├── manifest.json            (dataset-level: episode/sample counts, dims, seed)
└── splits.json              (episode-level train/val/test split)
```

Only episodes that pass the existing **physical** success detector
(`control.success.sustained_lift_success` -- sustained cube height gain,
not merely `controller.stage == DONE`) are written under `successful/`
and become official training data.

### Recording order (critical)

`EpisodeRecorder.record(observation, action, controller_stage)` is always
called **before** `env.step(action)` in `dataset/generate_dataset.py`, so
sample `t` is `(state_t, action_t)` -- the observation the expert actually
used to compute that action -- never `(state_t+1, action_t)`. This is
covered by a dedicated test
(`tests/test_dataset_recorder.py::test_end_to_end_alignment_with_real_environment`).

### Language instructions

```text
Pick up the red cube.
Grasp the red cube.
Lift the red cube.
Pick up the red block.
```

One instruction is chosen per episode (seeded random choice from the
generation RNG) and applies to every timestep in that episode -- it is
not re-parsed or repeated per-timestep in `trajectory.npz`, only stored
once in `metadata.json`.

### Randomization

Cube x/y position is randomized uniformly within `--xy-randomization`
meters (default `0.03`) of its nominal position each episode; height and
robot initial configuration are not randomized in Step 3. Given the same
`--seed`, generation is reproducible (one `numpy.random.default_rng(seed)`
drives both the cube offset and the instruction choice, in episode order).

### Measured results

Step 2.5 reliability (`python -m simulation.evaluate_grasp_reliability`,
10 deterministic + 10 episodes with cube x/y +/- 2cm):

```text
reach success rate:        100%
orientation aligned rate:  100%
lift (task) success rate:  100%
mean cube lift delta:      0.136 m / 0.137 m
```

Step 3 dataset generation (`python -m dataset.generate_dataset --episodes 100
--seed 42 --xy-randomization 0.03`):

```text
Requested episodes: 100      Successful: 100      Success rate: 100.0%
Total timesteps: 21,443      Mean length: 214.4 (min 205, max 226)
Train/val/test episodes: 80 / 10 / 10
Dataset disk size: ~1.56 GB
```

`python -m dataset.validation data/demonstrations` reports 100/100 valid.
These are measured values from an actual run, not targets -- re-run the
commands above to reproduce them (results will differ slightly per seed).

## What's here

- `simulation/scene.xml` -- ground plane, a table the robot is bolted to,
  a Franka Panda-style arm (`assets/franka_panda/`, vendored from
  [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
  Apache 2.0), a red cube (`red_cube`), a fixed camera (`main_camera`).
- `simulation/environment.py` -- `SimulationEnvironment`: the only module
  that knows MuJoCo internals. `get_observation()`, `get_robot_state()`,
  `get_end_effector_jacobian()`, `get_object_position()` /
  `set_object_position()` (privileged), `step(action)`, `reset()`, `close()`.
- `observations/` -- `Observation` (`rgb`, `state`, `timestamp`) and
  `RobotState` (+ `as_vector()`), simulator-agnostic.
- `control/` -- `RobotAction` (+ `as_vector()`), `control/kinematics.py`
  (pose IK + orientation math, pure NumPy), `control/success.py`
  (`sustained_lift_success`, `cube_lift_delta`), `control/scripted_controller.py`
  (`ScriptedController`, the pose-controlled pick state machine).
- `dataset/` -- Step 3:
  - `episode.py` -- on-disk episode format (`Episode`, `load_episode`).
  - `recorder.py` -- `EpisodeRecorder`: transactional `.tmp` ->
    `episode_NNNNNN/` writing; records only, never controls the robot.
  - `generate_dataset.py` -- CLI entry point (`python -m dataset.generate_dataset`).
  - `loader.py` -- `DemonstrationDataset` (flat, lazily-loaded, timestep-indexed).
  - `splits.py` -- deterministic episode-level train/val/test split.
  - `validation.py` -- per-episode and whole-dataset structural checks.
- `simulation/run_robot_demo.py` / `evaluate_grasp_reliability.py` -- Step
  2.5 single-episode demo and repeated-trial reliability sweep.
- `tests/` -- see `pytest` below; includes dataset serialization, the
  observation/action alignment test, loader/split/validation tests, and a
  small real-environment generation smoke test.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate the dataset

```bash
# Small check first (recommended before a large run)
python -m dataset.generate_dataset --episodes 5 --seed 42 --output data/demonstrations

# Full dataset (writes manifest.json + splits.json automatically)
python -m dataset.generate_dataset --episodes 100 --seed 42 \
    --output data/demonstrations --xy-randomization 0.03 --overwrite
```

Useful flags: `--max-steps` (per-episode timeout), `--save-failed` (keep
failed rollouts under `<output>/failed/` for debugging instead of
discarding them), `--overwrite` (required if `<output>` already has
episodes in it -- generation refuses to silently overwrite).

## Validate and load

```bash
python -m dataset.validation data/demonstrations
```

```python
from dataset.loader import DemonstrationDataset, load_episode

dataset = DemonstrationDataset("data/demonstrations", split="train")
sample = dataset[0]   # {"rgb", "state", "instruction", "action"}

episode = load_episode("data/demonstrations/successful/episode_000000")
```

## Run other demos

```bash
python -m simulation.run_simulation             # Step 1: camera + Observation only
python -m simulation.run_robot_demo              # Step 2.5: one full pose-controlled pick episode
python -m simulation.evaluate_grasp_reliability  # Step 2.5: 10 + 10 trial reliability sweep
```

## Run tests

```bash
pytest
```

## Platform notes

Tested on macOS with MuJoCo's default OpenGL backend (no `MUJOCO_GL`
override required). If rendering fails in a headless environment, set
`MUJOCO_GL=egl` (Linux) or `MUJOCO_GL=glfw`/`cgl` (macOS) before running.

Generated datasets under `data/demonstrations/` are not committed to git
(see `.gitignore`) -- regenerate with the command above.

## Future roadmap

```text
Step 4 — dense VLA behavior-cloning baseline
Step 5 — closed-loop VLA
Step 6 — sparse MoE VLA
Step 7 — benchmarking
Step 8 — ROS2/C++ integration
```
