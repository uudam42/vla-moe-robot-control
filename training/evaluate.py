"""Offline evaluation metrics, shared by the per-epoch validation pass in
``training/train.py`` and the standalone held-out evaluation CLI:

    python -m training.evaluate --checkpoint outputs/training/<run>/best.pt \
        --data data/demonstrations --split test

Reports offline action-prediction accuracy only. Per README "No
closed-loop claims yet": these numbers say nothing about closed-loop
robot-control success, which is Step 5.
"""

import argparse
import json

import numpy as np
import torch

from dataset.loader import DemonstrationDataset
from dataset.torch_dataset import DemonstrationTorchDataset
from models.dense_vla import DenseVLA, DenseVLAConfig
from models.language_encoder import load_tokenizer
from models.vision_encoder import build_image_transform
from training.checkpoint import load_checkpoint
from training.config import resolve_device
from training.losses import compute_loss
from training.normalization import ActionNormalizer, StateNormalizer

NUM_JOINTS = 7


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_model(
    model: DenseVLA,
    dataloader,
    device: torch.device,
    action_normalizer: ActionNormalizer,
    gripper_loss_weight: float = 1.0,
) -> dict:
    """Runs one full pass over ``dataloader`` and returns aggregate metrics.

    All physical-scale metrics (``per_joint_mae``, ``action_mae_8d``) are
    computed after denormalizing joint predictions back to radians -- see
    README "Offline evaluation metrics".
    """
    model.eval()

    total_loss = total_joint_loss = total_gripper_loss = 0.0
    num_samples = 0
    joint_abs_error_sum = np.zeros(NUM_JOINTS)
    action_abs_error_sum = np.zeros(NUM_JOINTS + 1)
    gripper_correct = 0

    for batch in dataloader:
        batch = _move_batch(batch, device)
        prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
        _, metrics = compute_loss(prediction, batch, gripper_loss_weight)

        batch_size = batch["state"].shape[0]
        total_loss += metrics["loss"] * batch_size
        total_joint_loss += metrics["joint_loss"] * batch_size
        total_gripper_loss += metrics["gripper_loss"] * batch_size
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


def print_prediction_examples(
    torch_dataset: DemonstrationTorchDataset,
    model: DenseVLA,
    device: torch.device,
    action_normalizer: ActionNormalizer,
    num_examples: int = 3,
    indices: list = None,
) -> None:
    """Prints GT vs predicted action for a few samples -- catches a collapsed
    model (e.g. constant output) that aggregate metrics alone can hide."""
    model.eval()
    if indices is None:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(torch_dataset), size=min(num_examples, len(torch_dataset)), replace=False)

    with torch.no_grad():
        for index in indices:
            sample = torch_dataset[int(index)]
            batch = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v]) for k, v in sample.items()}
            prediction = model(
                batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"]
            )

            gt_joints = action_normalizer.denormalize_joints(sample["joint_targets_normalized"].numpy())
            pred_joints = action_normalizer.denormalize_joints(
                prediction["joint_targets_normalized"][0].cpu().numpy()
            )
            gt_gripper = float(sample["gripper_target"])
            pred_gripper = float(torch.sigmoid(prediction["gripper_logit"][0, 0]).cpu())

            print(f"  instruction: {sample['instruction']!r}")
            print(f"  GT joints:   {np.round(gt_joints, 3)}")
            print(f"  Pred joints: {np.round(pred_joints, 3)}")
            print(f"  GT gripper: {gt_gripper:.2f}  Pred gripper: {pred_gripper:.3f}")
            print()


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/demonstrations")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))

    config = DenseVLAConfig.from_dict(checkpoint["config"])
    model = DenseVLA(config).to(device)
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

    metrics = evaluate_model(model, dataloader, device, action_normalizer)

    print(f"Split: {args.split}  ({metrics['num_samples']} samples)")
    print(f"Loss: {metrics['loss']:.4f}  (joint {metrics['joint_loss']:.4f}, gripper {metrics['gripper_loss']:.4f})")
    print(f"Joint MAE (physical, rad): {metrics['joint_mae']:.4f}")
    print(f"Per-joint MAE: {[round(v, 4) for v in metrics['per_joint_mae']]}")
    print(f"Gripper accuracy: {metrics['gripper_accuracy']:.4f}")
    print(f"8D action MAE: {[round(v, 4) for v in metrics['action_mae_8d']]}")
    print()
    print(f"Example predictions ({args.examples}):")
    print_prediction_examples(torch_dataset, model, device, action_normalizer, num_examples=args.examples)

    print(
        "Note: these are OFFLINE action-prediction metrics only -- they do not "
        "measure closed-loop robot-control success (see README)."
    )

    return metrics


if __name__ == "__main__":
    main()
