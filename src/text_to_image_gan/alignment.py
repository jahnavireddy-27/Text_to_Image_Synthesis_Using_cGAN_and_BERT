"""Frozen CLIP guidance for image-caption alignment during GAN training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, CLIPModel


def extract_clip_features(output: Any) -> torch.Tensor:
    """Support both tensor-returning and ModelOutput-returning Transformers APIs."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("CLIP feature extraction did not return a feature tensor.")


class CLIPAlignmentLoss(nn.Module):
    """Symmetric in-batch contrastive loss using frozen pretrained CLIP encoders.

    CLIP weights stay frozen, but gradients flow through the image encoder to
    generated pixels and back into the GAN generator.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        temperature: float = 0.07,
        *,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("CLIP alignment temperature must be positive.")
        self.temperature = float(temperature)
        self.model_name = model_name
        self.model = model or CLIPModel.from_pretrained(model_name)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.register_buffer(
            "image_mean",
            torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True) -> CLIPAlignmentLoss:
        """Keep the frozen teacher in evaluation mode."""
        super().train(mode)
        self.model.eval()
        return self

    @staticmethod
    def _get_image_features(output: Any) -> torch.Tensor:
        return extract_clip_features(output)

    @staticmethod
    def _get_text_features(output: Any) -> torch.Tensor:
        return extract_clip_features(output)

    def forward(self, generated_images: torch.Tensor, captions: Sequence[str]) -> torch.Tensor:
        """Return symmetric InfoNCE loss for generated images and their captions."""
        if generated_images.ndim != 4 or generated_images.shape[1] != 3:
            raise ValueError("generated_images must have shape (batch, 3, height, width).")
        if generated_images.shape[0] != len(captions):
            raise ValueError("The number of generated images and captions must match.")
        if not captions or any(not caption.strip() for caption in captions):
            raise ValueError("CLIP alignment requires non-empty captions.")
        if len(captions) < 2:
            return generated_images.sum() * 0.0

        images = (generated_images.float() + 1.0) / 2.0
        images = F.interpolate(
            images, size=(224, 224), mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        images = (images - self.image_mean.to(images)) / self.image_std.to(images)
        image_output = self.model.get_image_features(pixel_values=images)
        image_features = F.normalize(self._get_image_features(image_output), dim=-1)

        text_inputs = self.tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(generated_images.device) for key, value in text_inputs.items()}
        with torch.no_grad():
            text_output = self.model.get_text_features(**text_inputs)
            text_features = F.normalize(self._get_text_features(text_output), dim=-1)

        similarities = image_features @ text_features.transpose(0, 1)
        similarities = similarities / self.temperature
        targets = torch.arange(len(captions), device=generated_images.device)
        return 0.5 * (
            F.cross_entropy(similarities, targets)
            + F.cross_entropy(similarities.transpose(0, 1), targets)
        )
