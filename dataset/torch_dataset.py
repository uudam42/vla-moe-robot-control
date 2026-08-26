"""PyTorch Dataset wrapper around the Step 3 ``DemonstrationDataset`` loader.

Read-only with respect to ``data/demonstrations``: this module only reads
episode arrays and PNG frames already on disk (via ``dataset.loader``) and
never writes transformed images or cached tensors back into episode
directories (see README "Data integrity").

Official model inputs, enforced by construction here: RGB, 23D state,
instruction. Nothing privileged (cube position, Jacobian, controller
stage) is ever read from ``dataset.loader`` samples in the first place, so
there is nothing here to accidentally leak.
"""

from PIL import Image
from torch.utils.data import Dataset

import torch

from dataset.loader import DemonstrationDataset
from training.normalization import ActionNormalizer, StateNormalizer

NUM_JOINTS = 7


class DemonstrationTorchDataset(Dataset):
    """Wraps a ``DemonstrationDataset`` split, returning training-ready tensors.

    Args:
        base_dataset: A ``DemonstrationDataset`` (already split-restricted).
        image_transform: Callable, PIL Image -> ``Tensor[3,H,W]`` (see
            ``models.vision_encoder.build_image_transform``).
        tokenizer: A HuggingFace tokenizer (see ``models.language_encoder.load_tokenizer``).
        max_length: Tokenizer padding/truncation length.
        state_normalizer: Fit on the TRAIN split only (see
            ``training.normalization.fit_normalizers_from_split``); reused
            unchanged for val/test wrapping.
        action_normalizer: Same train-only requirement, for joint targets.
    """

    def __init__(
        self,
        base_dataset: DemonstrationDataset,
        image_transform,
        tokenizer,
        max_length: int,
        state_normalizer: StateNormalizer,
        action_normalizer: ActionNormalizer,
    ) -> None:
        self.base_dataset = base_dataset
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.base_dataset[index]

        pixel_values = self.image_transform(Image.fromarray(sample["rgb"]))

        state_normalized = self.state_normalizer.normalize(sample["state"])

        action = sample["action"]  # (8,) = [7 joint targets, gripper target]
        joint_targets_normalized = self.action_normalizer.normalize_joints(action[:NUM_JOINTS])
        gripper_target = action[NUM_JOINTS]

        tokenized = self.tokenizer(
            sample["instruction"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "pixel_values": pixel_values,
            "state": torch.from_numpy(state_normalized).float(),
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "joint_targets_normalized": torch.from_numpy(joint_targets_normalized).float(),
            "gripper_target": torch.tensor(gripper_target, dtype=torch.float32),
            "instruction": sample["instruction"],
        }
