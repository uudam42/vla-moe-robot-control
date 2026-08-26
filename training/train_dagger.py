"""Step 8 DAgger fine-tuning entry point.

    python -m training.train_dagger \
        --base-data data/demonstrations \
        --dagger-data data/dagger/round_001 \
        --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
        --epochs 15 --learning-rate 5e-5 --output outputs/training/temporal_dagger_run_001

Fine-tunes (not restarts) the trained Temporal Dense VLA on an AGGREGATED
dataset -- the original Step 3 expert train split PLUS the retained DAgger
corrective samples from one collection round, mixed at an explicit
batch-sampling ratio (README "Dataset weighting", default 50/50).
Architecture, normalization (reused verbatim from the Temporal checkpoint,
never recomputed -- README "Normalization"), and loss are all UNCHANGED
from Step 7; the only experimental variable is the training data.

Model selection (``best.pt``) still uses the ORIGINAL expert val split's
joint MAE, exactly like ``training/train_temporal.py`` -- so Temporal and
Temporal+DAgger remain apples-to-apples comparable. The DAgger round's own
held-out corrective-state metric (README "Corrective-state offline metric")
and the original expert TEST split are both evaluated before and after
fine-tuning to check for improvement / catastrophic forgetting (README
"Avoid catastrophic forgetting"), but neither drives checkpoint selection.

Tiny corrective-overfit sanity check (mandatory before trusting a full
fine-tune -- README "DAgger tiny sanity test"):

    python -m training.train_dagger --dagger-data data/dagger/round_001 \
        --checkpoint outputs/training/temporal_dense_vla_run_001/best.pt \
        --corrective-overfit-samples 32 --epochs 300
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dagger.aggregation import build_aggregated_dataloader
from dagger.dataset import TemporalDaggerCorrectiveDataset
from dataset.splits import load_splits, make_splits
from dataset.temporal_torch_dataset import TemporalDemonstrationDataset
from models.dense_vla import count_parameters
from models.language_encoder import load_tokenizer
from models.temporal_vla import TemporalDenseVLA, TemporalDenseVLAConfig
from models.vision_encoder import build_image_transform
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import TrainingConfig, resolve_device, set_seed
from training.evaluate_temporal import evaluate_temporal_model
from training.losses import compute_loss
from training.normalization import ActionNormalizer, StateNormalizer

DEFAULT_EPOCHS = 15
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_DAGGER_VAL_FRACTION = 0.1


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _resolve_run_dir(output: Path) -> Path:
    """``--output`` names the run directory directly (README "DAgger
    training CLI" example: ``--output outputs/training/temporal_dagger_run_001``),
    unlike Dense/MoE/Temporal's root-plus-auto-numbered-subdir convention."""
    output.mkdir(parents=True, exist_ok=True)
    return output


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


def _split_dagger_episodes(dagger_data: Path, val_fraction: float, seed: int) -> tuple:
    """Deterministic episode-level train/corrective-val split of one DAgger
    round -- reuses ``dataset.splits.make_splits`` (episode-level, never
    per-timestep, for the same reason Step 3's official split is
    episode-level: adjacent timesteps are near-duplicates)."""
    episode_dirs = sorted((Path(dagger_data) / "episodes").glob("episode_*"))
    episode_names = [d.name for d in episode_dirs]
    splits = make_splits(episode_names, seed=seed, train_frac=1.0 - val_fraction, val_frac=val_fraction)
    return splits["train"], splits["val"]


def build_expert_datasets(data_root, model_config, state_normalizer, action_normalizer):
    tokenizer = load_tokenizer(model_config.language_backbone)
    image_transform = build_image_transform()
    train_dataset = TemporalDemonstrationDataset(
        data_root, "train", image_transform, tokenizer, model_config.max_instruction_length,
        state_normalizer, action_normalizer, history_length=model_config.history_length,
    )
    val_dataset = TemporalDemonstrationDataset(
        data_root, "val", image_transform, tokenizer, model_config.max_instruction_length,
        state_normalizer, action_normalizer, history_length=model_config.history_length,
    )
    test_dataset = TemporalDemonstrationDataset(
        data_root, "test", image_transform, tokenizer, model_config.max_instruction_length,
        state_normalizer, action_normalizer, history_length=model_config.history_length,
    )
    return train_dataset, val_dataset, test_dataset, tokenizer, image_transform


def build_dagger_datasets(
    dagger_data, model_config, state_normalizer, action_normalizer, tokenizer, image_transform,
    val_fraction: float, seed: int,
) -> tuple:
    train_names, val_names = _split_dagger_episodes(Path(dagger_data), val_fraction, seed)
    train_dataset = TemporalDaggerCorrectiveDataset(
        dagger_data, image_transform, tokenizer, model_config.max_instruction_length,
        state_normalizer, action_normalizer, history_length=model_config.history_length,
        episode_names=train_names, retained_only=True,
    )
    val_dataset = TemporalDaggerCorrectiveDataset(
        dagger_data, image_transform, tokenizer, model_config.max_instruction_length,
        state_normalizer, action_normalizer, history_length=model_config.history_length,
        episode_names=val_names, retained_only=True,
    ) if val_names else None
    return train_dataset, val_dataset, train_names, val_names


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-data", default=TrainingConfig.data)
    parser.add_argument("--dagger-data", required=True)
    parser.add_argument("--checkpoint", required=True, help="Temporal Dense VLA checkpoint to fine-tune from.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--output", default="outputs/training/temporal_dagger_run_001", help="The run directory itself (not a root) -- see module docstring.")
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--expert-weight", type=float, default=0.5)
    parser.add_argument("--dagger-weight", type=float, default=0.5)
    parser.add_argument("--dagger-val-fraction", type=float, default=DEFAULT_DAGGER_VAL_FRACTION)
    parser.add_argument("--corrective-overfit-samples", type=int, default=0, help="DAgger tiny sanity check: overfit only on this many retained DAgger samples, no expert data, no aggregation.")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    run_dir = _resolve_run_dir(Path(args.output))
    print(f"Run directory: {run_dir}")

    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    model_config = TemporalDenseVLAConfig.from_dict(checkpoint["config"])
    model = TemporalDenseVLA(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    total_params, trainable_params = count_parameters(model)
    print(f"Loaded Temporal checkpoint: {args.checkpoint} (total={total_params:,} trainable={trainable_params:,})")

    # Normalization reused verbatim from the Temporal checkpoint -- never
    # recomputed from DAgger data (README "Normalization").
    state_normalizer = StateNormalizer.from_dict(checkpoint["state_normalizer"])
    action_normalizer = ActionNormalizer.from_dict(checkpoint["action_normalizer"])

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_config.to_dict(),
                "base_checkpoint": str(args.checkpoint),
                "base_data": args.base_data,
                "dagger_data": args.dagger_data,
                "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay, "seed": args.seed,
                "expert_weight": args.expert_weight, "dagger_weight": args.dagger_weight,
                "dagger_val_fraction": args.dagger_val_fraction,
                "corrective_overfit_samples": args.corrective_overfit_samples,
            },
            f, indent=2,
        )

    expert_train, expert_val, expert_test, tokenizer, image_transform = build_expert_datasets(
        args.base_data, model_config, state_normalizer, action_normalizer
    )
    dagger_train, dagger_val, dagger_train_names, dagger_val_names = build_dagger_datasets(
        args.dagger_data, model_config, state_normalizer, action_normalizer, tokenizer, image_transform,
        args.dagger_val_fraction, args.seed,
    )
    print(f"Expert train/val/test samples: {len(expert_train)}/{len(expert_val)}/{len(expert_test)}")
    print(f"DAgger episodes: train={len(dagger_train_names)} corrective-val={len(dagger_val_names)}")
    print(f"DAgger train/corrective-val samples: {len(dagger_train)}/{len(dagger_val) if dagger_val else 0}")

    expert_val_loader = DataLoader(expert_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    expert_test_loader = DataLoader(expert_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dagger_val_loader = (
        DataLoader(dagger_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if dagger_val else None
    )

    print()
    print("Before fine-tuning:")
    before_expert_test = evaluate_temporal_model(model, expert_test_loader, device, action_normalizer, args.gripper_loss_weight)
    print(f"  Expert test joint MAE: {before_expert_test['joint_mae']:.4f}  gripper acc: {before_expert_test['gripper_accuracy']:.4f}")
    before_dagger_val = None
    if dagger_val_loader is not None:
        before_dagger_val = evaluate_temporal_model(model, dagger_val_loader, device, action_normalizer, args.gripper_loss_weight)
        print(f"  DAgger corrective-val joint MAE: {before_dagger_val['joint_mae']:.4f}  gripper acc: {before_dagger_val['gripper_accuracy']:.4f}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.corrective_overfit_samples > 0:
        # DAgger tiny sanity check (README "DAgger tiny sanity test"):
        # prioritize the highest model/expert disagreement samples so the
        # overfit set is meaningfully corrective, not just periodic
        # near-agreement states.
        def _disagreement(i: int) -> float:
            episode_index, t = dagger_train._index[i]
            return float(dagger_train._labels[episode_index]["joint_l2_disagreement"][t])

        disagreement_order = sorted(range(len(dagger_train)), key=_disagreement, reverse=True)
        indices = disagreement_order[: min(args.corrective_overfit_samples, len(dagger_train))]
        overfit_dataset = Subset(dagger_train, indices)
        train_loader = DataLoader(overfit_dataset, batch_size=min(args.batch_size, len(overfit_dataset)), shuffle=True)
        eval_loader = train_loader
        aggregation_stats = None
    else:
        train_loader, aggregation_stats = build_aggregated_dataloader(
            expert_train, dagger_train, batch_size=args.batch_size,
            expert_weight=args.expert_weight, dagger_weight=args.dagger_weight, num_workers=args.num_workers,
        )
        print()
        print("Aggregated training data:", json.dumps(aggregation_stats.to_dict(), indent=2))
        eval_loader = expert_val_loader

    history = []
    best_joint_mae = float("inf")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.gripper_loss_weight)
        val_metrics = evaluate_temporal_model(model, eval_loader, device, action_normalizer, args.gripper_loss_weight)

        record = {
            "epoch": epoch, "train_loss": train_metrics["loss"], "train_joint_loss": train_metrics["joint_loss"],
            "val_loss": val_metrics["loss"], "val_joint_mae": val_metrics["joint_mae"],
            "val_gripper_accuracy": val_metrics["gripper_accuracy"], "epoch_seconds": time.time() - epoch_start,
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

    print()
    print("After fine-tuning:")
    after_expert_test = evaluate_temporal_model(model, expert_test_loader, device, action_normalizer, args.gripper_loss_weight)
    print(f"  Expert test joint MAE: {after_expert_test['joint_mae']:.4f}  gripper acc: {after_expert_test['gripper_accuracy']:.4f}")
    after_dagger_val = None
    if dagger_val_loader is not None:
        after_dagger_val = evaluate_temporal_model(model, dagger_val_loader, device, action_normalizer, args.gripper_loss_weight)
        print(f"  DAgger corrective-val joint MAE: {after_dagger_val['joint_mae']:.4f}  gripper acc: {after_dagger_val['gripper_accuracy']:.4f}")

    before_after = {
        "expert_test": {"before": before_expert_test, "after": after_expert_test},
        "dagger_corrective_val": {"before": before_dagger_val, "after": after_dagger_val},
    }
    with open(run_dir / "before_after_metrics.json", "w", encoding="utf-8") as f:
        json.dump(before_after, f, indent=2)

    metrics_summary = {
        "final_val_metrics": val_metrics,
        "best_val_joint_mae": best_joint_mae,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "aggregation": aggregation_stats.to_dict() if aggregation_stats else None,
        "dagger_train_episodes": len(dagger_train_names),
        "dagger_corrective_val_episodes": len(dagger_val_names),
        "corrective_overfit_samples": args.corrective_overfit_samples or None,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print()
    print(f"Checkpoints: {run_dir / 'best.pt'}, {run_dir / 'last.pt'}")
    return metrics_summary


if __name__ == "__main__":
    main()
