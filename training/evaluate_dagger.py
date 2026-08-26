"""Offline evaluation for the Temporal Dense VLA + DAgger checkpoint --
mirrors ``training/evaluate_temporal.py`` for the original held-out expert
test split, and additionally reports the "Corrective State Action Error"
(README "Corrective-state offline metric") on a DAgger round's held-out
corrective-val subset when ``--dagger-data`` is given.

    python -m training.evaluate_dagger --checkpoint outputs/training/temporal_dagger_run_001/best.pt \
        --data data/demonstrations --split test \
        --dagger-data data/dagger/round_001 --dagger-val-fraction 0.1

Reuses ``training.evaluate_temporal.evaluate_temporal_model`` unmodified --
both ``TemporalDemonstrationDataset`` and
``dagger.dataset.TemporalDaggerCorrectiveDataset`` yield the same batch
dict shape, so the same metric function applies to either.
"""

import argparse
import json

import torch

from dagger.dataset import TemporalDaggerCorrectiveDataset
from dataset.temporal_torch_dataset import TemporalDemonstrationDataset
from models.language_encoder import load_tokenizer
from models.temporal_vla import TemporalDenseVLA, TemporalDenseVLAConfig
from models.vision_encoder import build_image_transform
from training.checkpoint import load_checkpoint
from training.config import resolve_device
from training.evaluate_temporal import evaluate_temporal_model
from training.normalization import ActionNormalizer, StateNormalizer
from training.train_dagger import DEFAULT_DAGGER_VAL_FRACTION, _split_dagger_episodes


def main(argv: list = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/demonstrations")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dagger-data", default=None, help="If given, also report the corrective-state metric on this round's held-out corrective-val subset.")
    parser.add_argument("--dagger-val-fraction", type=float, default=DEFAULT_DAGGER_VAL_FRACTION)
    parser.add_argument("--dagger-split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Optional path to save metrics as JSON.")
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

    expert_dataset = TemporalDemonstrationDataset(
        args.data, args.split, image_transform, tokenizer, config.max_instruction_length,
        state_normalizer, action_normalizer, history_length=config.history_length,
    )
    expert_loader = torch.utils.data.DataLoader(expert_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    expert_metrics = evaluate_temporal_model(model, expert_loader, device, action_normalizer)

    print(f"Expert {args.split} split: ({expert_metrics['num_samples']} samples)")
    print(f"  Joint MAE (physical, rad): {expert_metrics['joint_mae']:.4f}")
    print(f"  Gripper accuracy: {expert_metrics['gripper_accuracy']:.4f}")

    result = {"expert_split": args.split, "expert_metrics": expert_metrics}

    if args.dagger_data:
        _, dagger_val_names = _split_dagger_episodes(args.dagger_data, args.dagger_val_fraction, args.dagger_split_seed)
        if dagger_val_names:
            dagger_val_dataset = TemporalDaggerCorrectiveDataset(
                args.dagger_data, image_transform, tokenizer, config.max_instruction_length,
                state_normalizer, action_normalizer, history_length=config.history_length,
                episode_names=dagger_val_names, retained_only=True,
            )
            dagger_val_loader = torch.utils.data.DataLoader(dagger_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            dagger_metrics = evaluate_temporal_model(model, dagger_val_loader, device, action_normalizer)
            print(f"DAgger corrective-val ({dagger_metrics['num_samples']} samples):")
            print(f"  Corrective-state joint MAE: {dagger_metrics['joint_mae']:.4f}")
            print(f"  Corrective-state gripper accuracy: {dagger_metrics['gripper_accuracy']:.4f}")
            result["dagger_corrective_val_metrics"] = dagger_metrics
        else:
            print("DAgger round has no held-out corrective-val episodes at this fraction/seed.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result["expert_metrics"], f, indent=2)
        print(f"Saved: {args.output}")

    return result


if __name__ == "__main__":
    main()
