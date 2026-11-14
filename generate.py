"""Command-line image-generation entry point."""

from __future__ import annotations

import argparse

from text_to_image_gan.inference import generate_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from captions.")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint path.")
    parser.add_argument(
        "--caption", action="append", required=True, help="Caption text; repeat for more."
    )
    parser.add_argument("--output-dir", default="outputs/generated")
    parser.add_argument("--images-per-caption", type=int, default=1)
    parser.add_argument("--device", default=None, help="Optional torch device.")
    arguments = parser.parse_args()
    paths = generate_images(
        arguments.checkpoint,
        arguments.caption,
        arguments.output_dir,
        images_per_caption=arguments.images_per_caption,
        device_name=arguments.device,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
