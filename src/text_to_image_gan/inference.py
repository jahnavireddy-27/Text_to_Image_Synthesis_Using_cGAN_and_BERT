"""Caption-to-image generation using a saved training checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image

from text_to_image_gan.training import (
    _encode_caption_batch,
    _generate_with_caption_features,
    build_models,
)
from text_to_image_gan.utils import resolve_device, validate_config


def load_trained_models(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[dict[str, Any], torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Load model weights and their exact saved configuration."""
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain its saved configuration.")
    validate_config(config, require_dataset=False)
    text_encoder, generator, discriminator = build_models(config)
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    text_encoder.to(device).eval()
    generator.to(device).eval()
    discriminator.to(device).eval()
    return config, text_encoder, generator, discriminator


@torch.inference_mode()
def generate_images(
    checkpoint_path: str | Path,
    captions: list[str],
    output_dir: str | Path,
    *,
    images_per_caption: int = 1,
    device_name: str | None = None,
) -> list[Path]:
    """Generate and save one or more images for every supplied caption."""
    if not captions or any(not caption.strip() for caption in captions):
        raise ValueError("Provide at least one non-empty caption.")
    if images_per_caption <= 0:
        raise ValueError("images_per_caption must be greater than zero.")
    device = resolve_device(device_name)
    config, text_encoder, generator, _ = load_trained_models(checkpoint_path, device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    noise_dim = int(config["generator"]["noise_dim"])
    for caption_index, caption in enumerate(captions):
        repeated_captions = [caption] * images_per_caption
        text_embeddings, word_features, word_mask = _encode_caption_batch(
            text_encoder, generator, repeated_captions
        )
        noise = torch.randn(images_per_caption, noise_dim, device=device)
        images = _generate_with_caption_features(
            generator, text_embeddings, noise, word_features, word_mask
        )
        for image_index, image in enumerate(images):
            image_path = destination / f"caption_{caption_index:03d}_{image_index:03d}.png"
            save_image(image, image_path, normalize=True, value_range=(-1, 1))
            generated_paths.append(image_path)
    return generated_paths
