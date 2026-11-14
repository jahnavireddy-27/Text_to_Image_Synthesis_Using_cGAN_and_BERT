
"""BERT-based caption encoder with an independently trainable projection."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """Encode captions as a global vector and optional contextual word vectors."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 32,
        output_dim: int = 256,
        freeze_text_encoder: bool = True,
        word_level_features: bool = False,
        *,
        tokenizer: Any | None = None,
        bert: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.bert = bert or AutoModel.from_pretrained(model_name)

        hidden_size = int(self.bert.config.hidden_size)

        self.projection = nn.Linear(hidden_size, output_dim)
        self.token_projection = (
            nn.Linear(hidden_size, output_dim)
            if word_level_features
            else None
        )

        self.max_length = max_length
        self.freeze_text_encoder = freeze_text_encoder

        if self.freeze_text_encoder:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)
            self.bert.eval()

    def train(self, mode: bool = True) -> TextEncoder:
        """Keep frozen BERT in evaluation mode during training."""
        super().train(mode)

        if self.freeze_text_encoder:
            self.bert.eval()

        return self

    def forward(
        self,
        captions: Sequence[str],
        *,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not captions:
            raise ValueError("TextEncoder requires at least one caption.")

        tokenizer_args = {
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }

        if return_tokens:
            tokenizer_args["return_special_tokens_mask"] = True

        try:
            encoded = self.tokenizer(list(captions), **tokenizer_args)
        except TypeError:
            tokenizer_args.pop("return_special_tokens_mask", None)
            encoded = self.tokenizer(list(captions), **tokenizer_args)

        special_tokens_mask = encoded.pop("special_tokens_mask", None)

        device = self.projection.weight.device

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        if special_tokens_mask is not None:
            special_tokens_mask = special_tokens_mask.to(device)

        # Avoid building an unnecessary computation graph through frozen BERT.
        context = torch.no_grad() if self.freeze_text_encoder else nullcontext()

        with context:
            outputs = self.bert(**encoded)

        pooled = getattr(outputs, "pooler_output", None)

        if pooled is None:
            raise ValueError(
                "The configured BERT model must provide pooler_output."
            )

        # Projection remains trainable even when BERT is frozen.
        sentence = self.projection(pooled)

        if not return_tokens:
            return sentence

        if self.token_projection is None:
            raise ValueError(
                "TextEncoder was created without word-level features."
            )

        contextual = getattr(outputs, "last_hidden_state", None)

        if contextual is None:
            raise ValueError(
                "The configured BERT model must provide last_hidden_state."
            )

        words = self.token_projection(contextual)

        word_mask = encoded["attention_mask"].bool()

        if special_tokens_mask is not None:
            word_mask &= ~special_tokens_mask.bool()

        if torch.any(word_mask.sum(dim=1) == 0):
            raise ValueError(
                "Every caption must contain at least one non-special token."
            )

        words = words * word_mask.unsqueeze(-1).to(words.dtype)

        return sentence, words, word_mask
