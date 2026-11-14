"""CLIP cosine-similarity evaluation and training-loss plotting."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPModel, CLIPProcessor

from text_to_image_gan.alignment import extract_clip_features


def cosine_similarities(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor
) -> torch.Tensor:
    """Return one cosine similarity per aligned image/text embedding pair."""
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("Image and text embedding tensors must have matching shapes.")
    if image_embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape (batch, embedding_dim).")
    return F.cosine_similarity(image_embeddings, text_embeddings, dim=1)


def to_clip_image(image: torch.Tensor | Image.Image) -> Image.Image:
    """Convert a model output in [-1, 1] or a PIL image to RGB for CLIP."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Image tensors must have shape (3, height, width).")
    normalized = image.detach().cpu().float().clamp(-1, 1).add(1).div(2)
    return to_pil_image(normalized).convert("RGB")


class CLIPScorer:
    """Score aligned image-caption pairs using pretrained CLIP embeddings."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 32,
        device: torch.device | str = "cpu",
        *,
        model: Any | None = None,
        processor: Any | None = None,
        text_padding: bool = True,
        text_truncation: bool = True,
        text_max_length: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model_name = model_name
        self.text_padding = text_padding
        self.text_truncation = text_truncation
        self.text_max_length = text_max_length
        self.model = model or CLIPModel.from_pretrained(model_name)
        self.processor = processor or CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def metadata(self) -> dict[str, Any]:
        """Describe model and effective image/text preprocessing for evaluation logs."""
        image_processor = self.processor.image_processor
        tokenizer = self.processor.tokenizer
        image_attributes = (
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
        )
        return {
            "model_id": self.model_name,
            "model_config_name_or_path": getattr(
                self.model.config, "_name_or_path", None
            ),
            "model_type": getattr(self.model.config, "model_type", None),
            "processor_class": type(self.processor).__name__,
            "image_processor_class": type(image_processor).__name__,
            "image_preprocessing": {
                "input_tensor_range": [-1.0, 1.0],
                "input_tensor_to_pil_mapping": "clamp to [-1, 1], then (x + 1) / 2",
                **{
                    name: getattr(image_processor, name, None)
                    for name in image_attributes
                },
            },
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
            "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
            "text_preprocessing": {
                "padding": self.text_padding,
                "truncation": self.text_truncation,
                "max_length": (
                    self.text_max_length
                    if self.text_max_length is not None
                    else getattr(tokenizer, "model_max_length", None)
                ),
            },
        }

    @torch.inference_mode()
    def score(
        self,
        images: Sequence[torch.Tensor | Image.Image],
        captions: Sequence[str],
    ) -> list[float]:
        """Return pairwise image-text cosine similarities, one score per pair."""
        if len(images) != len(captions):
            raise ValueError("The number of images and captions must match.")
        scores: list[float] = []
        for start in range(0, len(images), self.batch_size):
            image_batch = [
                to_clip_image(image) for image in images[start : start + self.batch_size]
            ]
            caption_batch = list(captions[start : start + self.batch_size])
            processor_kwargs: dict[str, Any] = {
                "images": image_batch,
                "text": caption_batch,
                "padding": self.text_padding,
                "truncation": self.text_truncation,
                "return_tensors": "pt",
            }
            if self.text_max_length is not None:
                processor_kwargs["max_length"] = self.text_max_length
            inputs = self.processor(
                **processor_kwargs,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            image_embeddings = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            image_embeddings = extract_clip_features(image_embeddings)
            text_embeddings = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            text_embeddings = extract_clip_features(text_embeddings)
            similarities = cosine_similarities(
                F.normalize(image_embeddings, dim=1),
                F.normalize(text_embeddings, dim=1),
            )
            scores.extend(float(score) for score in similarities.cpu())
        return scores


def plot_losses(losses: dict[str, list[float]], output_path: str | Path) -> None:
    """Save generator and discriminator losses as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, values in losses.items():
        if values:
            axis.plot(range(1, len(values) + 1), values, label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training losses")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)
