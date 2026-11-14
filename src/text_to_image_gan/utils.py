"""Configuration, validation, and reproducibility helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


_RESUME_IGNORED_CONFIG_PATHS = {
    ("training", "resume_from"),
}
_RESUME_ALLOWED_PATH_PREFIXES = {
    ("paths",),
}
_RESUME_ALLOWED_INCREASES = {
    ("training", "epochs"),
}


def load_config(path: str | Path, *, require_dataset: bool = True) -> dict[str, Any]:
    """Load YAML configuration and reject incompatible settings."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {config_path} must contain a YAML mapping.")
    validate_config(config, require_dataset=require_dataset)
    return config


def validate_config(
    config: dict[str, Any], *, require_dataset: bool = True
) -> None:
    """Validate dimensions, schedules, split separation, and required dataset ID."""
    required_sections = (
        "data",
        "text_encoder",
        "generator",
        "discriminator",
        "training",
        "evaluation",
        "paths",
        "reproducibility",
    )
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    data = config["data"]
    if require_dataset and not (
        isinstance(data.get("dataset_id"), str) and data["dataset_id"].strip()
    ):
        raise ValueError(
            "data.dataset_id must be set to a Hugging Face dataset repository ID."
        )
    if data.get("train_split") == data.get("test_split"):
        raise ValueError("Training and test splits must be different to prevent overlap.")
    if int(data.get("train_subset_size", 0)) <= 0:
        raise ValueError("data.train_subset_size must be greater than zero.")
    if int(data.get("test_subset_size", 0)) <= 0:
        raise ValueError("data.test_subset_size must be greater than zero.")
    if data.get("dataset_format") == "webdataset":
        shard_count = int(data.get("shard_count", 0))
        max_buffer_shards = int(data.get("streaming_max_buffer_input_shards", 0))
        if shard_count <= 0:
            raise ValueError("data.shard_count must be positive for WebDataset.")
        if max_buffer_shards != shard_count:
            raise ValueError(
                "data.streaming_max_buffer_input_shards must equal "
                "data.shard_count for WebDataset sampling."
            )
        if int(data.get("streaming_shuffle_buffer_size", 0)) <= 0:
            raise ValueError("data.streaming_shuffle_buffer_size must be positive.")
        if int(data.get("streaming_duplicate_skip_limit", -1)) < 0:
            raise ValueError(
                "data.streaming_duplicate_skip_limit cannot be negative."
            )
        if not isinstance(data.get("cache_dir"), str) or not data["cache_dir"].strip():
            raise ValueError("data.cache_dir must be a non-empty path.")
        if "partition_seed" not in data:
            raise ValueError("data.partition_seed must be configured.")

    text_dim = int(config["text_encoder"]["output_dim"])
    generator = config["generator"]
    discriminator = config["discriminator"]
    data_size = int(data["image_size"])
    image_channels = int(data["image_channels"])
    if image_channels != 3:
        raise ValueError("This RGB reconstruction requires data.image_channels=3.")

    if int(generator["text_dim"]) != text_dim:
        raise ValueError("generator.text_dim must match text_encoder.output_dim.")
    if int(discriminator["text_dim"]) != text_dim:
        raise ValueError("discriminator.text_dim must match text_encoder.output_dim.")
    if int(generator["initial_channels"]) != int(generator["channel_schedule"][0]):
        raise ValueError(
            "generator.initial_channels must match the first channel_schedule value."
        )
    if int(generator["initial_size"]) <= 0:
        raise ValueError("generator.initial_size must be greater than zero.")
    generator_schedule = generator["channel_schedule"]
    if len(generator_schedule) != 5:
        raise ValueError(
            "generator.channel_schedule must define the initial block and four "
            "upsampling stages."
        )
    if any(int(channel) <= 0 for channel in generator_schedule):
        raise ValueError("All generator channel_schedule values must be positive.")
    if int(generator_schedule[0]) != int(generator["initial_channels"]):
        raise ValueError(
            "generator.channel_schedule[0] must match initial_channels."
        )
    if int(generator_schedule[-1]) != image_channels:
        raise ValueError(
            "The generator's final channel count must match data.image_channels."
        )
    generated_size = int(generator["initial_size"]) * (2 ** (len(generator_schedule) - 1))
    if generated_size != data_size:
        raise ValueError(
            "Generator upsampling schedule yields "
            f"{generated_size}x{generated_size}, not data.image_size={data_size}."
        )

    image_schedule = discriminator["image_channel_schedule"]
    if len(image_schedule) != 5:
        raise ValueError(
            "discriminator.image_channel_schedule must define four image blocks."
        )
    if int(image_schedule[0]) != image_channels:
        raise ValueError(
            "The discriminator's first channel count must match data.image_channels."
        )
    if int(discriminator["image_channels"]) != image_channels:
        raise ValueError("discriminator.image_channels must match data.image_channels.")
    if int(discriminator["image_size"]) != data_size:
        raise ValueError("discriminator.image_size must match data.image_size.")
    if any(int(channel) <= 0 for channel in image_schedule):
        raise ValueError(
            "All discriminator image_channel_schedule values must be positive."
        )
    actual_feature_size = data_size
    image_kernel = int(discriminator["image_kernel_size"])
    image_stride = int(discriminator["image_stride"])
    image_padding = int(discriminator["image_padding"])
    if image_kernel <= 0 or image_stride <= 0 or image_padding < 0:
        raise ValueError("Discriminator image convolution parameters are invalid.")
    for _ in range(4):
        actual_feature_size = (
            actual_feature_size + 2 * image_padding - image_kernel
        ) // image_stride + 1
        if actual_feature_size <= 0:
            raise ValueError("Discriminator image blocks collapse the spatial dimensions.")
    if int(discriminator["feature_size"]) != actual_feature_size:
        raise ValueError(
            "discriminator.feature_size must match the image branch output size "
            f"({actual_feature_size})."
        )
    if int(image_schedule[-1]) <= 0:
        raise ValueError("The final discriminator image channel count must be positive.")
    if len(
        {
            len(discriminator["joint_channels"]),
            len(discriminator["joint_kernel_sizes"]),
            len(discriminator["joint_strides"]),
            len(discriminator["joint_paddings"]),
        }
    ) != 1:
        raise ValueError("All discriminator joint-head schedules must have equal lengths.")
    if not discriminator["joint_channels"]:
        raise ValueError("The discriminator joint head must contain convolution layers.")
    if int(discriminator["joint_channels"][-1]) != 1:
        raise ValueError("The final discriminator joint-head channel count must be 1.")
    final_size = int(discriminator["feature_size"])
    for kernel, stride, padding in zip(
        discriminator["joint_kernel_sizes"],
        discriminator["joint_strides"],
        discriminator["joint_paddings"],
    ):
        kernel_value = int(kernel)
        stride_value = int(stride)
        padding_value = int(padding)
        if kernel_value <= 0 or stride_value <= 0 or padding_value < 0:
            raise ValueError("Discriminator joint convolution parameters are invalid.")
        final_size = (
            final_size + 2 * padding_value - kernel_value
        ) // stride_value + 1
        if final_size <= 0:
            raise ValueError("Discriminator joint head collapses before producing a logit.")
    if final_size != 1:
        raise ValueError(
            "Discriminator joint-head dimensions must reduce the final output to 1x1."
        )
    if any(int(channel) <= 0 for channel in discriminator["joint_channels"]):
        raise ValueError("Discriminator joint-head channels must be positive.")
    if int(config["text_encoder"]["max_length"]) <= 0:
        raise ValueError("text_encoder.max_length must be greater than zero.")
    if int(generator["noise_dim"]) <= 0:
        raise ValueError("generator.noise_dim must be greater than zero.")
    if int(config["training"]["generator_updates_per_discriminator_update"]) <= 0:
        raise ValueError("Generator updates per discriminator update must be positive.")
    training = config["training"]
    required_training = (
        "drop_last",
        "generator_target_label",
        "noise_on_generator_updates",
        "checkpoint_interval_epochs",
        "scheduler_step_size",
    )
    missing_training = [key for key in required_training if key not in training]
    if missing_training:
        raise ValueError(
            "Missing training configuration values: " + ", ".join(missing_training)
        )
    if int(training["checkpoint_interval_epochs"]) <= 0:
        raise ValueError("training.checkpoint_interval_epochs must be positive.")
    if int(training["scheduler_step_size"]) <= 0:
        raise ValueError("training.scheduler_step_size must be positive.")
    if float(training.get("clip_alignment_weight", 0.0)) < 0:
        raise ValueError("training.clip_alignment_weight cannot be negative.")
    if float(training.get("clip_alignment_temperature", 0.07)) <= 0:
        raise ValueError("training.clip_alignment_temperature must be positive.")
    for label_key in ("real_label", "fake_label", "generator_target_label"):
        if label_key in training and not 0.0 <= float(training[label_key]) <= 1.0:
            raise ValueError(f"training.{label_key} must be between 0 and 1.")


def _flatten_config(config: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    flattened: dict[tuple[str, ...], Any] = {}
    for key, value in config.items():
        path = (*prefix, str(key))
        if isinstance(value, dict):
            flattened.update(_flatten_config(value, path))
        else:
            flattened[path] = value
    return flattened


def validate_resume_config(
    saved_config: dict[str, Any],
    active_config: dict[str, Any],
    *,
    checkpoint_epoch: int,
) -> list[str]:
    """Reject changed run settings except explicit epoch extension/output paths."""
    saved_values = _flatten_config(saved_config)
    active_values = _flatten_config(active_config)
    changed = {
        path
        for path in saved_values.keys() | active_values.keys()
        if saved_values.get(path) != active_values.get(path)
    }
    changed -= _RESUME_IGNORED_CONFIG_PATHS

    rejected: list[tuple[str, ...]] = []
    approved: list[str] = []
    for path in sorted(changed):
        if any(path[: len(prefix)] == prefix for prefix in _RESUME_ALLOWED_PATH_PREFIXES):
            approved.append(
                f"{'.'.join(path)} changed from {saved_values.get(path)!r} "
                f"to {active_values.get(path)!r}"
            )
        elif path in _RESUME_ALLOWED_INCREASES:
            old_value = int(saved_values.get(path, 0))
            new_value = int(active_values.get(path, 0))
            if new_value > old_value and new_value >= checkpoint_epoch:
                approved.append(
                    f"{'.'.join(path)} increased from {old_value} to {new_value}"
                )
            else:
                rejected.append(path)
        else:
            rejected.append(path)

    if rejected:
        names = ", ".join(".".join(path) for path in rejected)
        raise ValueError(
            "Resume configuration is incompatible with the checkpoint. "
            f"Changed settings: {names}. Only increasing training.epochs and "
            "changing paths.* are permitted."
        )
    return approved


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str | None = None) -> torch.device:
    """Resolve an explicitly requested device or choose the available default."""
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_config(config: dict[str, Any], path: str | Path) -> None:
    """Persist the complete configuration as YAML."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
