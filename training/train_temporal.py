"""Step 7 Temporal Dense VLA training entry point.

    python -m training.train_temporal --data data/demonstrations \
        --dense-checkpoint outputs/training/dense_vla_run_001/best.pt \
        --epochs 30 --batch-size 16 --learning-rate 1e-4 --history-length 4 --seed 42

Tiny-subset overfit sanity check (mandatory before full training):

    python -m training.train_temporal --dense-checkpoint outputs/training/dense_vla_run_001/best.pt \
        --overfit-samples 64 --epochs 300

Initializes a ``TemporalDenseVLA`` by converting the trained Dense Step 4
checkpoint (``models.temporal_vla.convert_dense_to_temporal`` -- shared
encoders/projections/attention/FFN/action-head copied verbatim; only the
new temporal-specific components -- action-history encoder, action
projection, temporal position embeddings -- are randomly initialized),
then trains on the SAME Step 3 dataset/split/loss as Dense. No MoE is
used anywhere in this milestone (README "Do NOT use MoE").
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.temporal_torch_dataset import DEFAULT_HISTORY_LENGTH, TemporalDemonstrationDataset
from models.dense_vla import DenseVLA, DenseVLAConfig, count_parameters
from models.language_encoder import load_tokenizer
from models.temporal_vla import TemporalDenseVLA, TemporalDenseVLAConfig, convert_dense_to_temporal
from models.vision_encoder import build_image_transform
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import TrainingConfig, resolve_device, set_seed
from training.evaluate_temporal import evaluate_temporal_model
from training.losses import compute_loss
from training.normalization import ActionNormalizer, StateNormalizer, fit_normalizers_from_split

EXAMPLES_EVERY_N_EPOCHS = 10


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _next_run_dir(output_root: Path, run_name: str = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_name:
        return output_root / run_name
    existing = sorted(output_root.glob("temporal_dense_vla_run_*"))
    next_index = 1
    if existing:
        last = existing[-1].name.rsplit("_", 1)[-1]
        next_index = int(last) + 1 if last.isdigit() else len(existing) + 1
    return output_root / f"temporal_dense_vla_run_{next_index:03d}"


def train_one_epoch(model, dataloader, optimizer, device, gripper_loss_weight) -> dict:
    model.train()
    total_loss = total_joint_loss = total_gripper_loss = 0.0
    num_samples = 0

    for batch in dataloader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()

        prediction = model(
            batch["pixel_values"], batch["states"], batch["previous_actions"],
            batch["input_ids"], batch["attention_mask"],
        )
        loss, metrics = compute_loss(prediction, batch, gripper_loss_weight)
        loss.backward()
        optimizer.step()

        batch_size = batch["states"].shape[0]
        total_loss += metrics["loss"] * batch_size
        total_joint_loss += metrics["joint_loss"] * batch_size
        total_gripper_loss += metrics["gripper_loss"] * batch_size
        num_samples += batch_size

    return {
        "loss": total_loss / num_samples,
        "joint_loss": total_joint_loss / num_samples,
        "gripper_loss": total_gripper_loss / num_samples,
    }


def build_datasets(config: TrainingConfig, model_config: TemporalDenseVLAConfig, state_normalizer, action_normalizer):
    tokenizer = load_tokenizer(model_config.language_backbone)
    image_transform = build_image_transform()

    train_dataset = TemporalDemonstrationDataset(
        config.data, "train", image_transform, tokenizer, model_config.max_instruction_length,
        state_normalizer, action_normalizer, history_length=model_config.history_length,
    )
    val_dataset = TemporalDemonstrationDataset(
        config.data, "val", image_transform, tokenizer, model_config.max_instruction_length,
        state_normalizer, action_normalizer, history_length=model_config.history_length,
    )

    if config.overfit_samples > 0:
        indices = list(range(min(config.overfit_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)
        val_dataset = train_dataset

    return train_dataset, val_dataset


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=TrainingConfig.data)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--output", default="outputs/training")
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--history-length", type=int, default=DEFAULT_HISTORY_LENGTH)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    config = TrainingConfig(
        data=args.data, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, seed=args.seed, device=args.device, num_workers=args.num_workers,
        output=args.output, overfit_samples=args.overfit_samples, resume=args.resume,
        gripper_loss_weight=args.gripper_loss_weight, run_name=args.run_name,
    )

    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Device: {device}")

    run_dir = _next_run_dir(Path(config.output), config.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    if config.resume:
        checkpoint = load_checkpoint(config.resume, map_location=str(device))
        model_config = TemporalDenseVLAConfig.from_dict(checkpoint["config"])
        state_normalizer = StateNormalizer.from_dict(checkpoint["state_normalizer"])
        action_normalizer = ActionNormalizer.from_dict(checkpoint["action_normalizer"])
        model = TemporalDenseVLA(model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {config.resume} at epoch {start_epoch}")
    else:
        dense_checkpoint = load_checkpoint(args.dense_checkpoint, map_location=str(device))
        dense_config = DenseVLAConfig.from_dict(dense_checkpoint["config"])
        dense_model = DenseVLA(dense_config).to(device)
        dense_model.load_state_dict(dense_checkpoint["model_state_dict"])
        dense_total, dense_trainable = count_parameters(dense_model)
        print(f"Loaded Dense checkpoint: {args.dense_checkpoint} (total={dense_total:,} trainable={dense_trainable:,})")

        model_config = TemporalDenseVLAConfig(
            state_dim=dense_config.state_dim, hidden_dim=dense_config.hidden_dim,
            num_layers=dense_config.num_layers, num_heads=dense_config.num_heads,
            ffn_dim=dense_config.ffn_dim, dropout=dense_config.dropout,
            max_instruction_length=dense_config.max_instruction_length,
            vision_backbone=dense_config.vision_backbone, language_backbone=dense_config.language_backbone,
            train_vision_encoder=dense_config.train_vision_encoder,
            train_language_encoder=dense_config.train_language_encoder,
            history_length=args.history_length,
        )
        model = convert_dense_to_temporal(dense_model, model_config).to(device)
        print(f"Converted Dense -> Temporal (history_length={args.history_length})")

        recomputed_state, recomputed_action = fit_normalizers_from_split(config.data, split="train")
        dense_state_normalizer = StateNormalizer.from_dict(dense_checkpoint["state_normalizer"])
        dense_action_normalizer = ActionNormalizer.from_dict(dense_checkpoint["action_normalizer"])
        state_match = np.allclose(recomputed_state.mean, dense_state_normalizer.mean)
        action_match = np.allclose(recomputed_action.joint_mean, dense_action_normalizer.joint_mean)
        print(f"Normalization stats match Dense checkpoint: state={state_match} action={action_match}")
        state_normalizer, action_normalizer = dense_state_normalizer, dense_action_normalizer

        start_epoch = 0
        del dense_model

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"training": config.to_dict(), "model": model_config.to_dict()}, f, indent=2)

    train_dataset, val_dataset = build_datasets(config, model_config, state_normalizer, action_normalizer)
    print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")

    persistent_workers = config.num_workers > 0
    train_batch_size = min(config.batch_size, len(train_dataset))
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=config.num_workers, persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=min(config.batch_size, len(val_dataset)), shuffle=False,
        num_workers=config.num_workers, persistent_workers=persistent_workers,
    )

    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    history = []
    best_joint_mae = float("inf")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config.gripper_loss_weight)
        val_metrics = evaluate_temporal_model(model, val_loader, device, action_normalizer, config.gripper_loss_weight)

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_joint_loss": train_metrics["joint_loss"],
            "val_loss": val_metrics["loss"],
            "val_joint_mae": val_metrics["joint_mae"],
            "val_gripper_accuracy": val_metrics["gripper_accuracy"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_joint_mae={val_metrics['joint_mae']:.4f} val_gripper_acc={val_metrics['gripper_accuracy']:.3f} "
            f"({record['epoch_seconds']:.1f}s)"
        )

        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, model_config, state_normalizer, action_normalizer, val_metrics)
        if val_metrics["joint_mae"] < best_joint_mae:
            best_joint_mae = val_metrics["joint_mae"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, model_config, state_normalizer, action_normalizer, val_metrics)

        with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    final_val_metrics = evaluate_temporal_model(model, val_loader, device, action_normalizer, config.gripper_loss_weight)
    metrics_summary = {
        "final_val_metrics": final_val_metrics,
        "best_val_joint_mae": best_joint_mae,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Checkpoints: {run_dir / 'best.pt'}, {run_dir / 'last.pt'}")
    return metrics_summary


if __name__ == "__main__":
    main()
