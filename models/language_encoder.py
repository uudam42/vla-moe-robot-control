"""Pretrained language backbone producing one pooled instruction embedding.

Frozen by default, same rationale as ``vision_encoder.py``: the Step 3
dataset only has 4 instruction variants, so there is no signal to
meaningfully fine-tune a language model on and a real risk of it degrading
into a degenerate embedding for those 4 strings.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_LANGUAGE_BACKBONE = "distilbert-base-uncased"


def load_tokenizer(model_name: str = DEFAULT_LANGUAGE_BACKBONE):
    """Single source of truth for tokenization -- used by the dataset loader
    (``dataset/torch_dataset.py``) and inference (``models/policy.py``), so
    training and inference always tokenize identically."""
    return AutoTokenizer.from_pretrained(model_name)


class LanguageEncoder(nn.Module):
    """Pretrained DistilBERT encoder, mean-pooled over valid (non-pad) tokens.

    Args:
        model_name: HuggingFace model id.
        trainable: If False (default), backbone parameters are frozen and
            kept in eval mode (disables dropout) regardless of the outer
            model's train/eval mode.
    """

    def __init__(self, model_name: str = DEFAULT_LANGUAGE_BACKBONE, trainable: bool = False) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size
        self.trainable = trainable
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``input_ids``/``attention_mask``: ``(B, L)`` -> pooled embedding ``(B, hidden_size)``."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def train(self, mode: bool = True) -> "LanguageEncoder":
        super().train(mode)
        if not self.trainable:
            self.model.eval()
        return self
