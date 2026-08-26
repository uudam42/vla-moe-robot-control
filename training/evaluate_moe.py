"""Offline evaluation for the MoE model -- same metrics as Dense
(``training/evaluate.py``) plus router/expert diagnostics.

    python -m training.evaluate_moe --checkpoint outputs/training/moe_vla_run_001/best.pt \
        --data data/demonstrations --split test

Same "no closed-loop claims" caveat as Dense evaluation applies.
"""

import argparse

import numpy as np
import torch

from dataset.loader import DemonstrationDataset
from dataset.torch_dataset import DemonstrationTorchDataset
from models.fusion import TOKEN_NAMES
from models.language_encoder import load_tokenizer
from models.moe_vla import MoEVLA, MoEVLAConfig, parameter_accounting
from models.vision_encoder import build_image_transform
from training.checkpoint import load_checkpoint
from training.config import resolve_device
from training.losses import compute_loss
from training.normalization import ActionNormalizer, StateNormalizer

NUM_JOINTS = 7


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_moe_model(
    model: MoEVLA,
    dataloader,
    device: torch.device,
    action_normalizer: ActionNormalizer,
    gripper_loss_weight: float = 1.0,
    router_aux_loss_weight: float = 0.01,
    collect_routing: bool = True,
) -> dict:
    """Same aggregate metrics as ``training.evaluate.evaluate_model`` plus
    ``router_aux_loss``, per-layer ``router_entropy``, and (if
    ``collect_routing``) expert-utilization counts overall and broken down
    by semantic token type (VISION/LANGUAGE/STATE/ACTION_QUERY).
    """
    model.eval()

    total_loss = total_joint_loss = total_gripper_loss = total_aux_loss = 0.0
    num_samples = 0
    joint_abs_error_sum = np.zeros(NUM_JOINTS)
    action_abs_error_sum = np.zeros(NUM_JOINTS + 1)
    gripper_correct = 0

    entropy_sum_by_layer = {}
    entropy_count_by_layer = {}
    # utilization_by_layer[layer][token_position][expert_id] = count
    utilization_by_layer = {}

    for batch in dataloader:
        batch = _move_batch(batch, device)
        prediction = model(
            batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"],
            collect_diagnostics=collect_routing,
        )
        _, metrics = compute_loss(prediction, batch, gripper_loss_weight)
        aux_loss_value = float(prediction["router_aux_loss"].item())

        batch_size = batch["state"].shape[0]
        total_loss += (metrics["loss"] + router_aux_loss_weight * aux_loss_value) * batch_size
        total_joint_loss += metrics["joint_loss"] * batch_size
        total_gripper_loss += metrics["gripper_loss"] * batch_size
        total_aux_loss += aux_loss_value * batch_size
        num_samples += batch_size

        joint_pred_physical = action_normalizer.denormalize_joints(
            prediction["joint_targets_normalized"].cpu().numpy()
        )
        joint_gt_physical = action_normalizer.denormalize_joints(
            batch["joint_targets_normalized"].cpu().numpy()
        )
        joint_abs_error = np.abs(joint_pred_physical - joint_gt_physical)
        joint_abs_error_sum += joint_abs_error.sum(axis=0)

        gripper_prob = torch.sigmoid(prediction["gripper_logit"].squeeze(-1)).cpu().numpy()
        gripper_gt = batch["gripper_target"].cpu().numpy()
        gripper_correct += int(((gripper_prob >= 0.5) == (gripper_gt >= 0.5)).sum())

        action_abs_error = np.concatenate(
            [joint_abs_error, np.abs(gripper_prob - gripper_gt)[:, None]], axis=1
        )
        action_abs_error_sum += action_abs_error.sum(axis=0)

        if collect_routing:
            for layer_index, diagnostics in prediction["layer_diagnostics"].items():
                entropy_sum_by_layer[layer_index] = entropy_sum_by_layer.get(layer_index, 0.0) + float(
                    diagnostics["entropy"].item()
                ) * batch_size
                entropy_count_by_layer[layer_index] = entropy_count_by_layer.get(layer_index, 0) + batch_size

                routing = diagnostics["routing_indices"][:, :, 0].cpu().numpy()  # (B, T), top-1
                num_experts = diagnostics["probs"].shape[-1]
                if layer_index not in utilization_by_layer:
                    utilization_by_layer[layer_index] = np.zeros((routing.shape[1], num_experts), dtype=np.int64)
                for token_position in range(routing.shape[1]):
                    counts = np.bincount(routing[:, token_position], minlength=num_experts)
                    utilization_by_layer[layer_index][token_position] += counts

    per_joint_mae = (joint_abs_error_sum / num_samples).tolist()

    result = {
        "num_samples": num_samples,
        "loss": total_loss / num_samples,
        "joint_loss": total_joint_loss / num_samples,
        "gripper_loss": total_gripper_loss / num_samples,
        "router_aux_loss": total_aux_loss / num_samples,
        "joint_mae": float(np.mean(per_joint_mae)),
        "per_joint_mae": per_joint_mae,
        "gripper_accuracy": gripper_correct / num_samples,
        "action_mae_8d": (action_abs_error_sum / num_samples).tolist(),
    }

    if collect_routing:
        result["router_entropy_by_layer"] = {
            str(layer): entropy_sum_by_layer[layer] / entropy_count_by_layer[layer]
            for layer in entropy_sum_by_layer
        }
        expert_utilization = {}
        for layer_index, counts in utilization_by_layer.items():
            total_per_token = counts.sum(axis=1, keepdims=True)
            fractions = counts / np.maximum(total_per_token, 1)
            expert_utilization[str(layer_index)] = {
                TOKEN_NAMES[position]: {
                    "counts": counts[position].tolist(),
                    "fractions": fractions[position].tolist(),
                }
                for position in range(counts.shape[0])
            }
        result["expert_utilization_by_layer"] = expert_utilization

    return result


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/demonstrations")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))

    config = MoEVLAConfig.from_dict(checkpoint["config"])
    model = MoEVLA(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    state_normalizer = StateNormalizer.from_dict(checkpoint["state_normalizer"])
    action_normalizer = ActionNormalizer.from_dict(checkpoint["action_normalizer"])

    tokenizer = load_tokenizer(config.language_backbone)
    image_transform = build_image_transform()

    base_dataset = DemonstrationDataset(args.data, split=args.split)
    torch_dataset = DemonstrationTorchDataset(
        base_dataset, image_transform, tokenizer, config.max_instruction_length, state_normalizer, action_normalizer
    )
    dataloader = torch.utils.data.DataLoader(
        torch_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    metrics = evaluate_moe_model(
        model, dataloader, device, action_normalizer, router_aux_loss_weight=config.router_aux_loss_weight
    )

    print(f"Split: {args.split}  ({metrics['num_samples']} samples)")
    print(f"Loss: {metrics['loss']:.4f}  (joint {metrics['joint_loss']:.4f}, gripper {metrics['gripper_loss']:.4f}, router_aux {metrics['router_aux_loss']:.4f})")
    print(f"Joint MAE (physical, rad): {metrics['joint_mae']:.4f}")
    print(f"Gripper accuracy: {metrics['gripper_accuracy']:.4f}")
    print(f"Router entropy by layer: {metrics.get('router_entropy_by_layer')}")
    print(f"Parameter accounting: {parameter_accounting(model)}")

    return metrics


if __name__ == "__main__":
    main()
