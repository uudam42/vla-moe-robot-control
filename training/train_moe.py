"""Step 6 MoE training entry point.

    python -m training.train_moe --data data/demonstrations \
        --dense-checkpoint outputs/training/dense_vla_run_001/best.pt \
        --epochs 30 --batch-size 32 --learning-rate 1e-4 \
        --num-experts 4 --top-k 1 --moe-layers 1 3 --router-aux-weight 0.01 --seed 42

Tiny-subset overfit sanity check (mandatory before full training, same gate
as Dense -- see README "Tiny-subset overfit gate"):

    python -m training.train_moe --dense-checkpoint outputs/training/dense_vla_run_001/best.pt \
        --overfit-samples 64 --epochs 300

Initializes a ``MoEVLA`` by converting the trained Dense Step 4 checkpoint
(``models.moe_vla.convert_dense_to_moe`` -- every expert starts as an exact
copy of the Dense FFN; only the router is freshly initialized), then
fine-tunes router + experts (+ everything else that's already trainable in
Dense) on the identical Step 3 dataset/split Dense used. Everything except
the FFN-vs-MoE architectural difference and the router auxiliary loss term
is kept as close to ``training/train.py`` as possible -- see README
"Experimental fairness".
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
from models.moe_vla import DEFAULT_MOE_LAYERS, MoEVLA, MoEVLAConfig, convert_dense_to_moe, parameter_accounting
from models.vision_encoder import build_image_transform
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import TrainingConfig, resolve_device, set_seed
from training.evaluate_moe import evaluate_moe_model
from training.losses import compute_loss
from training.normalization import ActionNormalizer, StateNormalizer, fit_normalizers_from_split

EXAMPLES_EVERY_N_EPOCHS = 10


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _next_run_dir(output_root: Path, run_name: str = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_name:
        return output_root / run_name
    existing = sorted(output_root.glob("moe_vla_run_*"))
    next_index = 1
    if existing:
        last = existing[-1].name.rsplit("_", 1)[-1]
        next_index = int(last) + 1 if last.isdigit() else len(existing) + 1
    return output_root / f"moe_vla_run_{next_index:03d}"


def compute_moe_loss(prediction: dict, batch: dict, gripper_loss_weight: float, router_aux_loss_weight: float) -> tuple:
    """Dense's ``compute_loss`` (joint MSE + gripper BCE) plus the router
    load-balancing auxiliary term (README "Loss")."""
    base_loss, metrics = compute_loss(prediction, batch, gripper_loss_weight)
    aux_loss = prediction["router_aux_loss"]
    total_loss = base_loss + router_aux_loss_weight * aux_loss
    metrics = dict(metrics)
    metrics["router_aux_loss"] = float(aux_loss.item())
    metrics["loss"] = float(total_loss.item())
    return total_loss, metrics


def verify_initial_functional_similarity(dense_model: DenseVLA, moe_model: MoEVLA, tokenizer, device) -> dict:
    """README "Verify initial functional similarity": compare Dense vs
    freshly-converted MoE predictions on a small fixed batch, BEFORE any
    MoE-specific training. Returns the measured max abs differences."""
    dense_model.eval()
    moe_model.eval()

    tokenized = tokenizer(
        ["Pick up the red cube."] * 4, return_tensors="pt", padding="max_length", truncation=True,
        max_length=dense_model.config.max_instruction_length,
    )
    generator = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(4, 3, 224, 224, generator=generator).to(device)
    state = torch.randn(4, dense_model.config.state_dim, generator=generator).to(device)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        dense_out = dense_model(pixel_values, input_ids, attention_mask, state)
        moe_out = moe_model(pixel_values, input_ids, attention_mask, state)

    joint_diff = (dense_out["joint_targets_normalized"] - moe_out["joint_targets_normalized"]).abs().max().item()
    gripper_diff = (dense_out["gripper_logit"] - moe_out["gripper_logit"]).abs().max().item()
    return {"max_joint_diff": joint_diff, "max_gripper_diff": gripper_diff}


def build_datasets(config: TrainingConfig, model_config: MoEVLAConfig, state_normalizer, action_normalizer):
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
        indices = list(range(min(config.overfit_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)
        val_dataset = train_dataset

    return train_dataset, val_dataset


def train_one_epoch(model, dataloader, optimizer, device, gripper_loss_weight, router_aux_loss_weight) -> dict:
    model.train()
    total_loss = total_joint_loss = total_gripper_loss = total_aux_loss = 0.0
    num_samples = 0

    for batch in dataloader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()

        prediction = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["state"])
        loss, metrics = compute_moe_loss(prediction, batch, gripper_loss_weight, router_aux_loss_weight)
        loss.backward()
        optimizer.step()

        batch_size = batch["state"].shape[0]
        total_loss += metrics["loss"] * batch_size
        total_joint_loss += metrics["joint_loss"] * batch_size
        total_gripper_loss += metrics["gripper_loss"] * batch_size
        total_aux_loss += metrics["router_aux_loss"] * batch_size
        num_samples += batch_size

    return {
        "loss": total_loss / num_samples,
        "joint_loss": total_joint_loss / num_samples,
        "gripper_loss": total_gripper_loss / num_samples,
        "router_aux_loss": total_aux_loss / num_samples,
    }


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=TrainingConfig.data)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--output", default="outputs/training")
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--moe-layers", type=int, nargs="+", default=list(DEFAULT_MOE_LAYERS))
    parser.add_argument("--router-aux-weight", type=float, default=0.01)
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

    router_aux_loss_weight = args.router_aux_weight

    if config.resume:
        checkpoint = load_checkpoint(config.resume, map_location=str(device))
        model_config = MoEVLAConfig.from_dict(checkpoint["config"])
        state_normalizer = StateNormalizer.from_dict(checkpoint["state_normalizer"])
        action_normalizer = ActionNormalizer.from_dict(checkpoint["action_normalizer"])
        model = MoEVLA(model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        router_aux_loss_weight = model_config.router_aux_loss_weight
        print(f"Resumed from {config.resume} at epoch {start_epoch}")
    else:
        dense_checkpoint = load_checkpoint(args.dense_checkpoint, map_location=str(device))
        dense_config = DenseVLAConfig.from_dict(dense_checkpoint["config"])
        dense_model = DenseVLA(dense_config).to(device)
        dense_model.load_state_dict(dense_checkpoint["model_state_dict"])
        dense_total, dense_trainable = count_parameters(dense_model)
        print(f"Loaded Dense checkpoint: {args.dense_checkpoint} (total={dense_total:,} trainable={dense_trainable:,})")

        model_config = MoEVLAConfig(
            state_dim=dense_config.state_dim,
            hidden_dim=dense_config.hidden_dim,
            num_layers=dense_config.num_layers,
            num_heads=dense_config.num_heads,
            ffn_dim=dense_config.ffn_dim,
            dropout=dense_config.dropout,
            max_instruction_length=dense_config.max_instruction_length,
            vision_backbone=dense_config.vision_backbone,
            language_backbone=dense_config.language_backbone,
            train_vision_encoder=dense_config.train_vision_encoder,
            train_language_encoder=dense_config.train_language_encoder,
            num_experts=args.num_experts,
            top_k=args.top_k,
            moe_layers=tuple(args.moe_layers),
            router_aux_loss_weight=router_aux_loss_weight,
        )
        model = convert_dense_to_moe(dense_model, model_config).to(device)
        print(f"Converted Dense -> MoE (experts={args.num_experts}, top_k={args.top_k}, moe_layers={args.moe_layers})")

        tokenizer_for_check = load_tokenizer(model_config.language_backbone)
        similarity = verify_initial_functional_similarity(dense_model, model, tokenizer_for_check, device)
        print(f"Initial Dense vs MoE output difference: {similarity}")
        with open(run_dir / "dense_moe_initial_similarity.json", "w", encoding="utf-8") as f:
            json.dump(similarity, f, indent=2)

        # Verify normalization stats match Dense exactly (README "Normalization")
        # -- reuse Dense's checkpointed stats as the source of truth, but also
        # recompute independently from the same train split as a cross-check.
        recomputed_state, recomputed_action = fit_normalizers_from_split(config.data, split="train")
        dense_state_normalizer = StateNormalizer.from_dict(dense_checkpoint["state_normalizer"])
        dense_action_normalizer = ActionNormalizer.from_dict(dense_checkpoint["action_normalizer"])
        state_stats_match = np.allclose(recomputed_state.mean, dense_state_normalizer.mean) and np.allclose(
            recomputed_state.std, dense_state_normalizer.std
        )
        action_stats_match = np.allclose(recomputed_action.joint_mean, dense_action_normalizer.joint_mean) and np.allclose(
            recomputed_action.joint_std, dense_action_normalizer.joint_std
        )
        print(f"Normalization stats match Dense checkpoint: state={state_stats_match} action={action_stats_match}")
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
    accounting = parameter_accounting(model)
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,} active/token={accounting['active_parameters_per_token']:,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    history = []
    router_history = []
    best_joint_mae = float("inf")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config.gripper_loss_weight, router_aux_loss_weight)
        val_metrics = evaluate_moe_model(
            model, val_loader, device, action_normalizer, config.gripper_loss_weight, router_aux_loss_weight,
            collect_routing=True,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_joint_loss": train_metrics["joint_loss"],
            "train_router_aux_loss": train_metrics["router_aux_loss"],
            "val_loss": val_metrics["loss"],
            "val_joint_mae": val_metrics["joint_mae"],
            "val_gripper_accuracy": val_metrics["gripper_accuracy"],
            "val_router_aux_loss": val_metrics["router_aux_loss"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(record)
        router_history.append({
            "epoch": epoch,
            "router_entropy_by_layer": val_metrics.get("router_entropy_by_layer"),
            "expert_utilization_by_layer": val_metrics.get("expert_utilization_by_layer"),
        })

        print(
            f"epoch {epoch:03d} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_joint_mae={val_metrics['joint_mae']:.4f} val_gripper_acc={val_metrics['gripper_accuracy']:.3f} "
            f"router_aux={val_metrics['router_aux_loss']:.4f} ({record['epoch_seconds']:.1f}s)"
        )

        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, model_config, state_normalizer, action_normalizer, val_metrics)
        if val_metrics["joint_mae"] < best_joint_mae:
            best_joint_mae = val_metrics["joint_mae"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, model_config, state_normalizer, action_normalizer, val_metrics)

        with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(run_dir / "router_history.json", "w", encoding="utf-8") as f:
            json.dump(router_history, f, indent=2)

    final_val_metrics = evaluate_moe_model(
        model, val_loader, device, action_normalizer, config.gripper_loss_weight, router_aux_loss_weight, collect_routing=True
    )
    if router_history:
        with open(run_dir / "expert_utilization.json", "w", encoding="utf-8") as f:
            json.dump(final_val_metrics.get("expert_utilization_by_layer"), f, indent=2)

    metrics_summary = {
        "final_val_metrics": final_val_metrics,
        "best_val_joint_mae": best_joint_mae,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_accounting": accounting,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Checkpoints: {run_dir / 'best.pt'}, {run_dir / 'last.pt'}")
    return metrics_summary


if __name__ == "__main__":
    main()
