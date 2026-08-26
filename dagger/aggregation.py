"""Combines the original expert demonstrations with DAgger corrective
samples for fine-tuning (README "Aggregated dataset" / "Dataset weighting"):

    D_train = D_expert UNION D_dagger

Implemented as a plain ``ConcatDataset`` plus a ``WeightedRandomSampler``
so the two sources can be mixed at an explicit, reported ratio at the
batch-sampling level (default 50/50, README "Dataset weighting") rather
than whatever ratio their raw sizes happen to produce.
"""

from dataclasses import dataclass

from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

DEFAULT_EXPERT_WEIGHT = 0.5
DEFAULT_DAGGER_WEIGHT = 0.5


@dataclass
class AggregationStats:
    """Exact sample accounting for one aggregated dataset (README "Dataset
    statistics": "Do not merely say 'more data was added.'")."""

    num_expert_samples: int
    num_dagger_samples: int
    expert_weight: float
    dagger_weight: float
    per_expert_sample_weight: float
    per_dagger_sample_weight: float

    def to_dict(self) -> dict:
        total = self.num_expert_samples + self.num_dagger_samples
        return {
            "num_expert_samples": self.num_expert_samples,
            "num_dagger_samples": self.num_dagger_samples,
            "dagger_to_expert_ratio": (self.num_dagger_samples / self.num_expert_samples) if self.num_expert_samples else None,
            "dagger_fraction_of_raw_total": (self.num_dagger_samples / total) if total else None,
            "requested_expert_sampling_weight": self.expert_weight,
            "requested_dagger_sampling_weight": self.dagger_weight,
        }


def build_aggregated_dataloader(
    expert_dataset,
    dagger_dataset,
    batch_size: int,
    expert_weight: float = DEFAULT_EXPERT_WEIGHT,
    dagger_weight: float = DEFAULT_DAGGER_WEIGHT,
    num_workers: int = 0,
    num_samples_per_epoch: int = None,
) -> tuple:
    """Returns ``(DataLoader, AggregationStats)``.

    Every batch is drawn (with replacement) from the concatenation of
    ``expert_dataset`` and ``dagger_dataset`` such that, in expectation,
    ``expert_weight`` of the mass comes from ``expert_dataset`` and
    ``dagger_weight`` from ``dagger_dataset`` -- regardless of their raw
    relative sizes (README "Do not let one source dominate accidentally
    without measurement").
    """
    num_expert = len(expert_dataset)
    num_dagger = len(dagger_dataset)
    if num_expert == 0:
        raise ValueError("expert_dataset is empty")
    if num_dagger == 0:
        raise ValueError("dagger_dataset is empty -- nothing retained to fine-tune on")
    if not abs((expert_weight + dagger_weight) - 1.0) < 1e-9:
        raise ValueError(f"expert_weight + dagger_weight must sum to 1.0, got {expert_weight} + {dagger_weight}")

    per_expert_weight = expert_weight / num_expert
    per_dagger_weight = dagger_weight / num_dagger
    sample_weights = [per_expert_weight] * num_expert + [per_dagger_weight] * num_dagger

    concatenated = ConcatDataset([expert_dataset, dagger_dataset])
    num_samples = num_samples_per_epoch or len(concatenated)
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

    loader = DataLoader(concatenated, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    stats = AggregationStats(
        num_expert_samples=num_expert, num_dagger_samples=num_dagger,
        expert_weight=expert_weight, dagger_weight=dagger_weight,
        per_expert_sample_weight=per_expert_weight, per_dagger_sample_weight=per_dagger_weight,
    )
    return loader, stats
