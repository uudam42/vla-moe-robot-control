"""Offline evaluation for the Temporal Dense VLA -- same metric set as
Dense/MoE (``training/evaluate.py`` / ``training/evaluate_moe.py``).

    python -m training.evaluate_temporal --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
        --data data/demonstrations --split test

Lower offline error is NOT assumed to predict better closed-loop success
(README "Offline evaluation") -- these numbers are reported alongside,
never instead of, the closed-loop results.
"""

import argparse

import numpy as np
import torch

from dataset.temporal_torch_dataset import TemporalDemonstrationDataset
from models.language_encoder import load_tokenizer
from models.temporal_vla import TemporalDenseVLA, TemporalDenseVLAConfig
from models.vision_encoder import build_image_transform
from training.checkpoint import load_checkpoint
from training.config import resolve_device
from training.losses import compute_loss
from training.normalization import ActionNormalizer, StateNormalizer

NUM_JOINTS = 7


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_temporal_model(
    model: TemporalDenseVLA, dataloader, device: torch.device, action_normalizer: ActionNormalizer,
    gripper_loss_weight: float = 1.0,
) -> dict:
    """Same aggregate metrics as ``training.evaluate.evaluate_model``."""
    model.eval()
    total_loss = total_joint_loss = total_gripper_loss = 0.0
    num_samples = 0
    joint_abs_error_sum = np.zeros(NUM_JOINTS)
    action_abs_error_sum = np.zeros(NUM_JOINTS + 1)
    gripper_correct = 0

    for batch in dataloader:
        batch = _move_batch(batch, device)
        prediction = model(
            batch["pixel_values"], batch["states"], batch["previous_actions"],
            batch["input_ids"], batch["attention_mask"],
        )
        _, metrics = compute_loss(prediction, batch, gripper_loss_weight)

        batch_size = batch["states"].shape[0]
        total_loss += metrics["loss"] * batch_size
        total_joint_loss += metrics["joint_loss"] * batch_size
        total_gripper_loss += metrics["gripper_loss"] * batch_size
        num_samples += batch_size

        joint_pred_physical = action_normalizer.denormalize_joints(prediction["joint_targets_normalized"].cpu().numpy())
        joint_gt_physical = action_normalizer.denormalize_joints(batch["joint_targets_normalized"].cpu().numpy())
        joint_abs_error = np.abs(joint_pred_physical - joint_gt_physical)
        joint_abs_error_sum += joint_abs_error.sum(axis=0)

        gripper_prob = torch.sigmoid(prediction["gripper_logit"].squeeze(-1)).cpu().numpy()
        gripper_gt = batch["gripper_target"].cpu().numpy()
        gripper_correct += int(((gripper_prob >= 0.5) == (gripper_gt >= 0.5)).sum())

        action_abs_error = np.concatenate([joint_abs_error, np.abs(gripper_prob - gripper_gt)[:, None]], axis=1)
        action_abs_error_sum += action_abs_error.sum(axis=0)

    per_joint_mae = (joint_abs_error_sum / num_samples).tolist()
    return {
        "num_samples": num_samples,
        "loss": total_loss / num_samples,
        "joint_loss": total_joint_loss / num_samples,
        "gripper_loss": total_gripper_loss / num_samples,
        "joint_mae": float(np.mean(per_joint_mae)),
        "per_joint_mae": per_joint_mae,
        "gripper_accuracy": gripper_correct / num_samples,
        "action_mae_8d": (action_abs_error_sum / num_samples).tolist(),
    }


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/demonstrations")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))

    config = TemporalDenseVLAConfig.from_dict(checkpoint["config"])
    model = TemporalDenseVLA(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    state_normalizer = StateNormalizer.from_dict(checkpoint["state_normalizer"])
    action_normalizer = ActionNormalizer.from_dict(checkpoint["action_normalizer"])

    tokenizer = load_tokenizer(config.language_backbone)
    image_transform = build_image_transform()

    torch_dataset = TemporalDemonstrationDataset(
        args.data, args.split, image_transform, tokenizer, config.max_instruction_length,
        state_normalizer, action_normalizer, history_length=config.history_length,
    )
    dataloader = torch.utils.data.DataLoader(
        torch_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    metrics = evaluate_temporal_model(model, dataloader, device, action_normalizer)

    print(f"Split: {args.split}  ({metrics['num_samples']} samples)")
    print(f"Loss: {metrics['loss']:.4f}  (joint {metrics['joint_loss']:.4f}, gripper {metrics['gripper_loss']:.4f})")
    print(f"Joint MAE (physical, rad): {metrics['joint_mae']:.4f}")
    print(f"Per-joint MAE: {[round(v, 4) for v in metrics['per_joint_mae']]}")
    print(f"Gripper accuracy: {metrics['gripper_accuracy']:.4f}")

    return metrics


if __name__ == "__main__":
    main()
