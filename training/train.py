"""Step 4 training entry point.

    python -m training.train --data data/demonstrations --epochs 30 \
        --batch-size 32 --learning-rate 1e-4 --seed 42

Tiny-subset overfit sanity check (run this BEFORE full training -- see
README "Tiny overfit is a hard gate"):

    python -m training.train --overfit-samples 64 --epochs 300

Trains a dense multimodal behavior-cloning policy on the Step 3
demonstration dataset. Read-only with respect to ``data/demonstrations``
(see README "Data integrity") -- all outputs go under ``--output``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.loader import DemonstrationDataset
from dataset.torch_dataset import DemonstrationTorchDataset
from models.dense_vla import DenseVLA, DenseVLAConfig, count_parameters
from models.language_encoder import load_tokenizer
from models.vision_encoder import build_image_transform
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import TrainingConfig, resolve_device, set_seed
from training.evaluate import evaluate_model, print_prediction_examples
from training.losses import compute_loss
from training.normalization import ActionNormalizer, StateNormalizer, fit_normalizers_from_split

EXAMPLES_EVERY_N_EPOCHS = 10


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()
    }


def _next_run_dir(output_root: Path, run_name: str = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_name:
        return output_root / run_name
    existing = sorted(output_root.glob("dense_vla_run_*"))
    next_index = 1
    if existing:
        last = existing[-1].name.rsplit("_", 1)[-1]
        next_index = int(last) + 1 if last.isdigit() else len(existing) + 1
    return output_root / f"dense_vla_run_{next_index:03d}"


def train_one_epoch(model, dataloader, optimizer, device, gripper_loss_weight, use_amp, scaler) -> dict:
    model.train()
    total_loss = total_joint_loss = total_gripper_loss = 0.0
    num_samples = 0

    for batch in dataloader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=use_amp):
            prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
            loss, metrics = compute_loss(prediction, batch, gripper_loss_weight)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = batch["state"].shape[0]
        total_loss += metrics["loss"] * batch_size
        total_joint_loss += metrics["joint_loss"] * batch_size
        total_gripper_loss += metrics["gripper_loss"] * batch_size
        num_samples += batch_size

    return {
        "loss": total_loss / num_samples,
        "joint_loss": total_joint_loss / num_samples,
        "gripper_loss": total_gripper_loss / num_samples,
    }


@torch.no_grad()
def benchmark_latency(model, sample_batch, device, num_iters: int = 50, num_warmup: int = 10) -> dict:
    """Batch-1 forward-pass latency, for the Dense-vs-future-MoE comparison."""
    model.eval()
    batch = {k: (v[:1].to(device) if isinstance(v, torch.Tensor) else v[:1]) for k, v in sample_batch.items()}

    for _ in range(num_warmup):
        model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    latencies_ms = []
    for _ in range(num_iters):
        start = time.perf_counter()
        model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    return {
        "device": str(device),
        "num_iters": num_iters,
        "mean_ms": float(latencies_ms.mean()),
        "p50_ms": float(np.median(latencies_ms)),
    }


def build_datasets(config: TrainingConfig, model_config: DenseVLAConfig, state_normalizer, action_normalizer):
    tokenizer = load_tokenizer(model_config.language_backbone)
    image_transform = build_image_transform()

    train_base = DemonstrationDataset(config.data, split="train")
    val_base = DemonstrationDataset(config.data, split="val")

    train_dataset = DemonstrationTorchDataset(
        train_base, image_transform, tokenizer, model_config.max_instruction_length, state_normalizer, action_normalizer
    )
    val_dataset = DemonstrationTorchDataset(
        val_base, image_transform, tokenizer, model_config.max_instruction_length, state_normalizer, action_normalizer
    )

    if config.overfit_samples > 0:
        n = min(config.overfit_samples, len(train_dataset))
        indices = list(range(n))
        train_dataset = Subset(train_dataset, indices)
        val_dataset = train_dataset  # overfit mode: monitor the same tiny subset

    return train_dataset, val_dataset


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=TrainingConfig.data)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--output", default=TrainingConfig.output)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gripper-loss-weight", type=float, default=TrainingConfig.gripper_loss_weight)
    parser.add_argument("--train-vision-encoder", action="store_true")
    parser.add_argument("--train-language-encoder", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    config = TrainingConfig(
        data=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        output=args.output,
        overfit_samples=args.overfit_samples,
        resume=args.resume,
        gripper_loss_weight=args.gripper_loss_weight,
        train_vision_encoder=args.train_vision_encoder,
        train_language_encoder=args.train_language_encoder,
        run_name=args.run_name,
    )

    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Device: {device}")

    run_dir = _next_run_dir(Path(config.output), config.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    if config.resume:
        checkpoint = load_checkpoint(config.resume, map_location=str(device))
        model_config = DenseVLAConfig.from_dict(checkpoint["config"])
        state_normalizer = StateNormalizer.from_dict(checkpoint["state_normalizer"])
        action_normalizer = ActionNormalizer.from_dict(checkpoint["action_normalizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {config.resume} at epoch {start_epoch}")
    else:
        model_config = DenseVLAConfig(
            train_vision_encoder=config.train_vision_encoder,
            train_language_encoder=config.train_language_encoder,
        )
        state_normalizer, action_normalizer = fit_normalizers_from_split(config.data, split="train")
        start_epoch = 0
        checkpoint = None

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"training": config.to_dict(), "model": model_config.to_dict()}, f, indent=2)

    train_dataset, val_dataset = build_datasets(config, model_config, state_normalizer, action_normalizer)
    print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")

    persistent_workers = config.num_workers > 0
    train_batch_size = min(config.batch_size, len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config.batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=persistent_workers,
    )

    model = DenseVLA(model_config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")
    print(f"Vision encoder trainable: {model_config.train_vision_encoder}")
    print(f"Language encoder trainable: {model_config.train_language_encoder}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history = []
    best_joint_mae = float("inf")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, config.gripper_loss_weight, use_amp, scaler
        )
        val_metrics = evaluate_model(model, val_loader, device, action_normalizer, config.gripper_loss_weight)

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_joint_loss": train_metrics["joint_loss"],
            "train_gripper_loss": train_metrics["gripper_loss"],
            "val_loss": val_metrics["loss"],
            "val_joint_mae": val_metrics["joint_mae"],
            "val_gripper_accuracy": val_metrics["gripper_accuracy"],
            "learning_rate": config.learning_rate,
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_joint_mae={val_metrics['joint_mae']:.4f} "
            f"val_gripper_acc={val_metrics['gripper_accuracy']:.3f} "
            f"({record['epoch_seconds']:.1f}s)"
        )

        save_checkpoint(
            run_dir / "last.pt", model, optimizer, epoch, model_config, state_normalizer, action_normalizer, val_metrics
        )
        if val_metrics["joint_mae"] < best_joint_mae:
            best_joint_mae = val_metrics["joint_mae"]
            save_checkpoint(
                run_dir / "best.pt", model, optimizer, epoch, model_config, state_normalizer, action_normalizer, val_metrics
            )

        if (epoch + 1) % EXAMPLES_EVERY_N_EPOCHS == 0 or epoch == config.epochs - 1:
            print("Example predictions:")
            print_prediction_examples(val_dataset, model, device, action_normalizer, num_examples=2)

        with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    final_val_metrics = evaluate_model(model, val_loader, device, action_normalizer, config.gripper_loss_weight)
    sample_batch = next(iter(val_loader))
    sample_batch = _move_batch(sample_batch, device)
    latency = benchmark_latency(model, sample_batch, device)
    print(f"Batch-1 inference latency: mean={latency['mean_ms']:.2f}ms p50={latency['p50_ms']:.2f}ms on {latency['device']}")

    metrics_summary = {
        "final_val_metrics": final_val_metrics,
        "best_val_joint_mae": best_joint_mae,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "latency": latency,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Checkpoints: {run_dir / 'best.pt'}, {run_dir / 'last.pt'}")
    return metrics_summary


if __name__ == "__main__":
    main()
