"""Generate from the held-out test captions and compute CLIP cosine scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from text_to_image_gan.data import load_train_test_datasets, make_dataloader
from text_to_image_gan.evaluation import CLIPScorer
from text_to_image_gan.inference import load_trained_models
from text_to_image_gan.utils import resolve_device


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: str | Path, device_name: str | None = None
) -> dict[str, Any]:
    """Generate test-set images and report their CLIP image-text cosine score."""
    device = resolve_device(device_name)
    config, text_encoder, generator, _ = load_trained_models(checkpoint_path, device)
    _, test_dataset = load_train_test_datasets(config)
    test_loader = make_dataloader(test_dataset, config, train=False)
    evaluation_config = config["evaluation"]
    scorer = CLIPScorer(
        model_name=evaluation_config["clip_model_name"],
        batch_size=int(evaluation_config["batch_size"]),
        device=device,
        text_padding=bool(evaluation_config["text_padding"]),
        text_truncation=bool(evaluation_config["text_truncation"]),
        text_max_length=evaluation_config["text_max_length"],
    )
    evaluation_seed = int(evaluation_config["seed"])
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(evaluation_seed)
    similarities: list[float] = []
    for batch in test_loader:
        captions = batch["caption"]
        embeddings = text_encoder(captions)
        noise = torch.randn(
            len(captions),
            int(config["generator"]["noise_dim"]),
            device=device,
            generator=noise_generator,
        )
        generated = generator(embeddings, noise)
        similarities.extend(scorer.score(list(generated), list(captions)))
    if not similarities:
        raise RuntimeError("The test subset produced no image-caption pairs.")
    result = {
        "generated_clip_cosine_similarity": sum(similarities) / len(similarities),
        "evaluated_pairs": float(len(similarities)),
        "historical_reported_similarity_reference": float(
            evaluation_config["historical_reported_similarity"]
        ),
        "evaluation_seed": evaluation_seed,
        "evaluation_metadata": {
            **scorer.metadata(),
            "dataset_id": config["data"]["dataset_id"],
            "test_split": config["data"]["test_split"],
            "test_subset_size": int(config["data"]["test_subset_size"]),
            "checkpoint_path": str(checkpoint_path),
            "text_encoder_model_id": config["text_encoder"]["model_name"],
            "text_encoder_frozen": bool(
                config["text_encoder"]["freeze_text_encoder"]
            ),
            "similarity_metric": "pairwise cosine similarity of L2-normalized embeddings",
        },
    }
    output_path = Path(config["paths"]["output_dir"]) / "clip_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated images with CLIP.")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional torch device.")
    arguments = parser.parse_args()
    result = evaluate_checkpoint(arguments.checkpoint, arguments.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
