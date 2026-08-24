"""Task-success metrics, kept separate from controller completion.

``ScriptedController`` reaching ``Stage.DONE`` means the state machine ran
to the end -- it says nothing about whether the cube was actually picked
up. These are pure functions over recorded trajectories (no MuJoCo, no
controller state) so they're trivial to unit test and to reuse across the
single-episode demo and the multi-episode reliability evaluation.
"""

import numpy as np

DEFAULT_LIFT_THRESHOLD = 0.045  # meters; cube must rise above this to count
DEFAULT_SUSTAIN_TICKS = 10  # consecutive samples the lift must hold for


def sustained_lift_success(
    cube_z_history: np.ndarray,
    initial_cube_z: float,
    lift_threshold: float = DEFAULT_LIFT_THRESHOLD,
    sustain_ticks: int = DEFAULT_SUSTAIN_TICKS,
) -> bool:
    """True iff the cube stayed elevated for `sustain_ticks` consecutive samples.

    Requiring a sustained run (rather than a single elevated sample) avoids
    counting a momentary bump from a finger/table collision as a successful
    lift.

    Args:
        cube_z_history: Cube height at each recorded control tick, shape ``(T,)``.
        initial_cube_z: Cube height before the episode started moving it.
        lift_threshold: Minimum height gain, in meters, to count as "lifted".
        sustain_ticks: Number of consecutive elevated samples required.

    Returns:
        Whether any window of `sustain_ticks` consecutive samples was all
        elevated above ``initial_cube_z + lift_threshold``.
    """
    history = np.asarray(cube_z_history, dtype=np.float64)
    if history.size < sustain_ticks:
        return False

    elevated = (history - initial_cube_z) > lift_threshold
    window = np.ones(sustain_ticks, dtype=int)
    run_lengths = np.convolve(elevated.astype(int), window, mode="valid")
    return bool(np.any(run_lengths == sustain_ticks))


def cube_lift_delta(cube_z_history: np.ndarray, initial_cube_z: float) -> float:
    """Peak cube height gain observed during an episode, in meters."""
    history = np.asarray(cube_z_history, dtype=np.float64)
    if history.size == 0:
        return 0.0
    return float(np.max(history) - initial_cube_z)
