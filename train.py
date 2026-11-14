"""Command-line training entry point."""

from __future__ import annotations

import argparse

from text_to_image_gan.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the text-to-image GAN.")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path.")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda.")
    arguments = parser.parse_args()
    train(arguments.config, arguments.device)


if __name__ == "__main__":
    main()
