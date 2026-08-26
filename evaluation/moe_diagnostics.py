"""Post-hoc MoE router analysis: ACTION_QUERY expert-selection trace,
expert-switching rate, and routing-vs-gripper correlation.

Everything here reads ``MoEVLAPolicy.predict_with_routing()``'s OUTPUT
diagnostics (never fed back as model input) and privileged simulator
state (cube position) purely to correlate routing choices with task
phase *after the fact* -- never fed to the policy itself. Per README
"Avoid causal overclaiming": findings here are reported as "expert E was
disproportionately selected during timesteps post-hoc associated with X",
never as "expert E learned X".
"""

import numpy as np

from control.success import sustained_lift_success
from models.moe_policy import MoEVLAPolicy
from simulation.environment import SimulationEnvironment

CUBE_BODY_NAME = "red_cube"
ACTION_QUERY_TOKEN_POSITION = 3  # fixed fusion token order: VISION, LANGUAGE, STATE, ACTION_QUERY


def run_routing_analysis_episode(
    env: SimulationEnvironment,
    policy: MoEVLAPolicy,
    instruction: str,
    cube_xy_offset: np.ndarray = None,
    max_steps: int = 350,
) -> dict:
    """Diagnostic-only rollout: identical control loop to
    ``evaluation.closed_loop.run_closed_loop_episode`` but calls
    ``predict_with_routing()`` instead of ``predict()`` so the ACTION_QUERY
    token's expert selection can be traced tick-by-tick. Not used for the
    official success-rate evaluation -- see ``simulation/evaluate_moe_vla_closed_loop.py``.
    """
    env.reset()
    if cube_xy_offset is not None:
        cube_position = env.get_object_position(CUBE_BODY_NAME)
        cube_position[0] += cube_xy_offset[0]
        cube_position[1] += cube_xy_offset[1]
        env.set_object_position(CUBE_BODY_NAME, cube_position)

    initial_cube_z = float(env.get_object_position(CUBE_BODY_NAME)[2])
    cube_z_history = []
    gripper_targets = []
    action_query_expert_by_layer = {layer: [] for layer in policy.model.config.moe_layers}

    step_count = 0
    success = False
    for tick in range(max_steps):
        observation = env.get_observation()
        action, layer_diagnostics = policy.predict_with_routing(observation, instruction)
        env.step(action)

        cube_position = env.get_object_position(CUBE_BODY_NAME)
        cube_z_history.append(float(cube_position[2]))
        gripper_targets.append(action.gripper_target)

        for layer_index, diagnostics in layer_diagnostics.items():
            expert_id = int(diagnostics["routing_indices"][0, ACTION_QUERY_TOKEN_POSITION, 0].item())
            action_query_expert_by_layer[layer_index].append(expert_id)

        step_count = tick + 1
        if sustained_lift_success(cube_z_history, initial_cube_z):
            success = True
            break

    return {
        "success": success,
        "episode_length": step_count,
        "cube_z_history": cube_z_history,
        "gripper_targets": gripper_targets,
        "action_query_expert_by_layer": action_query_expert_by_layer,
    }


def expert_switch_rate(expert_sequence: list) -> float:
    """Fraction of consecutive control ticks where the ACTION_QUERY token's
    selected expert changed (README "Expert switching rate")."""
    if len(expert_sequence) < 2:
        return 0.0
    switches = sum(1 for a, b in zip(expert_sequence, expert_sequence[1:]) if a != b)
    return switches / (len(expert_sequence) - 1)


def gripper_expert_correlation(expert_sequence: list, gripper_targets: list, num_experts: int) -> dict:
    """Per-expert mean predicted gripper_target (1=open, 0=closed) among
    ticks where that expert was selected for ACTION_QUERY -- descriptive
    only (README "Routing vs gripper behavior" / "Avoid causal overclaiming")."""
    expert_sequence = np.asarray(expert_sequence)
    gripper_targets = np.asarray(gripper_targets)
    result = {}
    for expert_id in range(num_experts):
        mask = expert_sequence == expert_id
        result[str(expert_id)] = {
            "count": int(mask.sum()),
            "mean_gripper_target": float(gripper_targets[mask].mean()) if mask.any() else None,
        }
    return result


def summarize_routing_analysis(episode_traces: list, num_experts: int) -> dict:
    """Aggregate ``run_routing_analysis_episode`` results across several episodes."""
    by_layer = {}
    for trace in episode_traces:
        for layer_index, sequence in trace["action_query_expert_by_layer"].items():
            by_layer.setdefault(layer_index, {"switch_rates": [], "sequences": [], "gripper": []})
            by_layer[layer_index]["switch_rates"].append(expert_switch_rate(sequence))
            by_layer[layer_index]["sequences"].extend(sequence)
            by_layer[layer_index]["gripper"].extend(trace["gripper_targets"])

    summary = {}
    for layer_index, data in by_layer.items():
        summary[str(layer_index)] = {
            "mean_expert_switch_rate": float(np.mean(data["switch_rates"])) if data["switch_rates"] else None,
            "gripper_expert_correlation": gripper_expert_correlation(data["sequences"], data["gripper"], num_experts),
        }
    return summary
