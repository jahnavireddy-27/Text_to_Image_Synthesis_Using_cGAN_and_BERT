"""Reproducible Hugging Face image-caption dataset loading."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import random
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms


class ImageCaptionDataset(Dataset[dict[str, Any]]):
    """Map-style image-caption pairs with images normalized to [-1, 1]."""

    def __init__(
        self,
        dataset: Any,
        image_column: str,
        caption_column: str,
        image_size: int = 64,
        caption_selection: str = "first",
    ) -> None:
        self.dataset = dataset
        self.image_column = image_column
        self.caption_column = caption_column
        self.caption_selection = caption_selection
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.dataset[index]
        image = _as_rgb_image(record[self.image_column])
        caption = _select_caption(
            _get_field(record, self.caption_column), self.caption_selection
        )
        return {"image": self.image_transform(image), "caption": caption}


class StreamingImageCaptionDataset(IterableDataset[dict[str, Any]]):
    """Streaming counterpart to ImageCaptionDataset."""

    def __init__(
        self,
        dataset: Any,
        image_column: str,
        caption_column: str,
        image_size: int = 64,
        caption_selection: str = "first",
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.image_column = image_column
        self.caption_column = caption_column
        self.caption_selection = caption_selection
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for record in self.dataset:
            image = _as_rgb_image(record[self.image_column])
            caption = _select_caption(
                _get_field(record, self.caption_column), self.caption_selection
            )
            yield {"image": self.image_transform(image), "caption": caption}


def _get_field(record: Mapping[str, Any], field_path: str) -> Any:
    value: Any = record
    for field in field_path.split("."):
        if not isinstance(value, Mapping) or field not in value:
            raise ValueError(f"Dataset field {field_path!r} is missing.")
        value = value[field]
    return value


def _as_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as image:
            return image.convert("RGB")
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image:
                return image.convert("RGB")
        if value.get("path"):
            with Image.open(value["path"]) as image:
                return image.convert("RGB")
    try:
        return Image.fromarray(value).convert("RGB")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported image value of type {type(value).__name__}.") from error


def _as_caption(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (list, tuple)) and value:
        first_caption = value[0]
        if isinstance(first_caption, str) and first_caption.strip():
            return first_caption
    raise ValueError("Each image-caption record must contain a non-empty caption string.")


def _select_caption(value: Any, strategy: str = "first") -> str:
    """Choose one caption, optionally sampling a detailed caption each access."""
    if strategy == "first":
        return _as_caption(value)
    if strategy == "random_detailed":
        if isinstance(value, (list, tuple)) and value:
            captions = [item for item in value if isinstance(item, str) and item.strip()]
            if not captions:
                raise ValueError(
                    "Each image-caption record must contain a non-empty caption string."
                )
            detailed = captions[len(captions) // 2 :]
            return random.choice(detailed or captions)
        return _as_caption(value)
    raise ValueError(f"Unsupported caption selection strategy: {strategy!r}.")


def _available_columns(dataset: Any) -> set[str]:
    columns = getattr(dataset, "column_names", None)
    if isinstance(columns, dict):
        return set(columns)
    if columns is not None:
        return set(columns)
    features = getattr(dataset, "features", None)
    if features is not None:
        return set(features)
    return set()


def _check_columns(
    dataset: Any,
    image_column: str,
    caption_column: str,
    id_column: str | None = None,
) -> None:
    columns = _available_columns(dataset)
    expected_columns = {image_column, caption_column.split(".", 1)[0]}
    if id_column:
        expected_columns.add(id_column)
    if columns and not expected_columns.issubset(columns):
        missing = expected_columns - columns
        raise ValueError(
            "Configured dataset columns are unavailable: "
            + ", ".join(sorted(missing))
            + f". Available columns: {', '.join(sorted(columns))}."
        )


def _select_subset(
    dataset: Any,
    *,
    size: int,
    seed: int,
    streaming: bool,
    id_column: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Select a seeded subset and return provenance for the selected records."""
    if streaming:
        rng = random.Random(seed)
        reservoir: list[tuple[int, Any]] = []
        for index, record in enumerate(dataset):
            if index < size:
                reservoir.append((index, record))
            else:
                replacement = rng.randrange(index + 1)
                if replacement < size:
                    reservoir[replacement] = (index, record)
        if len(reservoir) < size:
            raise ValueError(
                f"Requested subset size {size} exceeds streaming split size "
                f"{len(reservoir)}."
            )
        reservoir.sort(key=lambda item: item[0])
        selected = [record for _, record in reservoir]
        selected_indices = [index for index, _ in reservoir]
        method = "deterministic_reservoir_sampling"
    else:
        available = len(dataset)
        if available < size:
            raise ValueError(
                f"Requested subset size {size} exceeds split size {available}."
            )
        selected_indices = random.Random(seed).sample(range(available), size)
        if hasattr(dataset, "select"):
            selected = dataset.select(selected_indices)
        else:
            selected = [dataset[index] for index in selected_indices]
        method = "seeded_random_sample_without_replacement"

    provenance: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "indices": selected_indices,
    }
    if id_column is not None:
        provenance["ids"] = [str(record[id_column]) for record in selected]
    return selected, provenance


def _image_fingerprint(record: Mapping[str, Any], image_column: str) -> str:
    image = _as_rgb_image(record[image_column])
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB\0".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def assert_subsets_disjoint(
    train_dataset: Any,
    test_dataset: Any,
    image_column: str,
    *,
    expected_train_size: int | None = None,
    expected_test_size: int | None = None,
    allow_overlap_count: int = 0,
) -> None:
    """Raise if an identical image appears in both selected subsets."""
    train_fingerprints: set[str] = set()
    test_fingerprints: set[str] = set()
    train_count = 0
    test_count = 0
    for record in train_dataset:
        train_fingerprints.add(_image_fingerprint(record, image_column))
        train_count += 1
    for record in test_dataset:
        test_fingerprints.add(_image_fingerprint(record, image_column))
        test_count += 1
    if expected_train_size is not None and train_count != expected_train_size:
        raise ValueError(
            f"Selected train subset has {train_count} examples; "
            f"expected {expected_train_size}."
        )
    if expected_test_size is not None and test_count != expected_test_size:
        raise ValueError(
            f"Selected test subset has {test_count} examples; "
            f"expected {expected_test_size}."
        )
    overlap = train_fingerprints & test_fingerprints
    if len(overlap) > allow_overlap_count:
        raise ValueError(
            f"Training and test subsets contain {len(overlap)} identical "
            "image(s); choose non-overlapping splits."
        )
    if overlap:
        print(
            f"Warning: allowing {len(overlap)} identical train/test image(s) "
            "as configured."
        )


def load_train_test_datasets(
    config: dict[str, Any],
) -> tuple[Dataset[dict[str, Any]] | IterableDataset[dict[str, Any]],
           Dataset[dict[str, Any]] | IterableDataset[dict[str, Any]]]:
    """Load independently selected and audited train and test image-caption splits."""
    data_config = config["data"]
    dataset_id = data_config.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(
            "data.dataset_id must be set to a Hugging Face dataset repository ID."
        )
    from datasets import load_dataset

    streaming = bool(data_config["streaming"])
    train_split = data_config["train_split"]
    test_split = data_config["test_split"]
    image_column = data_config["image_column"]
    caption_column = data_config["caption_column"]
    id_column = data_config.get("id_column")
    seed = int(config["reproducibility"]["seed"])
    train_size = int(data_config["train_subset_size"])
    test_size = int(data_config["test_subset_size"])

    if data_config.get("dataset_format") == "webdataset":
        if not streaming:
            raise ValueError(
                "The configured WebDataset shards must be loaded with streaming=true."
            )
        if train_split == test_split:
            raise ValueError("Generated train and test subset names must differ.")
        train_records, test_records, _ = _load_webdataset_selection(
            config, load_dataset
        )
        train_selected = ImageCaptionDataset(
            train_records, "image", "caption", int(data_config["image_size"])
        )
        test_selected = ImageCaptionDataset(
            test_records, "image", "caption", int(data_config["image_size"])
        )
        assert_subsets_disjoint(
            train_records,
            test_records,
            "image",
            expected_train_size=train_size,
            expected_test_size=test_size,
            allow_overlap_count=int(data_config.get("allow_identical_images", 0)),
        )
        return train_selected, test_selected
    else:
        if train_split == test_split:
            raise ValueError("Training and test splits must be different.")
        dataset_kwargs: dict[str, Any] = {"streaming": streaming}
        if data_config.get("dataset_revision"):
            dataset_kwargs["revision"] = data_config["dataset_revision"]
        if data_config.get("cache_dir"):
            dataset_kwargs["cache_dir"] = data_config["cache_dir"]
        train_raw = load_dataset(dataset_id, split=train_split, **dataset_kwargs)
        test_raw = load_dataset(dataset_id, split=test_split, **dataset_kwargs)
        _check_columns(train_raw, image_column, caption_column, id_column)
        _check_columns(test_raw, image_column, caption_column, id_column)
        train_selected, train_selection = _select_subset(
            train_raw,
            size=train_size,
            seed=seed,
            streaming=streaming,
            id_column=id_column,
        )
        test_selected, test_selection = _select_subset(
            test_raw,
            size=test_size,
            seed=seed + 1,
            streaming=streaming,
            id_column=id_column,
        )
    assert_subsets_disjoint(
        train_selected,
        test_selected,
        image_column,
        expected_train_size=train_size,
        expected_test_size=test_size,
        allow_overlap_count=int(data_config.get("allow_identical_images", 0)),
    )
    selection_path = Path(config["paths"]["output_dir"]) / "dataset_selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "dataset_format": data_config.get("dataset_format", "huggingface"),
                "dataset_revision": data_config.get("dataset_revision"),
                "streaming": streaming,
                "train_split": train_split,
                "test_split": test_split,
                "train": train_selection,
                "test": test_selection,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_class = StreamingImageCaptionDataset if streaming else ImageCaptionDataset
    common_args = {
        "image_column": image_column,
        "caption_column": caption_column,
        "image_size": int(data_config["image_size"]),
        "caption_selection": data_config.get("caption_selection", "first"),
    }
    return dataset_class(train_selected, **common_args), dataset_class(
        test_selected, **common_args
    )


def _webdataset_source_shard(source_url: Any) -> str:
    if not isinstance(source_url, str) or not source_url:
        raise ValueError("A WebDataset record is missing its __url__ source shard.")
    shard = Path(urlsplit(source_url).path).name
    if not shard:
        raise ValueError(f"Could not determine source shard from {source_url!r}.")
    return shard


def _webdataset_cache_key(data_config: Mapping[str, Any], seed: int) -> str:
    settings = {
        "dataset_id": data_config["dataset_id"],
        "dataset_revision": data_config.get("dataset_revision"),
        "source_split": data_config["source_split"],
        "shard_url_template": data_config["shard_url_template"],
        "shard_count": int(data_config["shard_count"]),
        "image_column": data_config["image_column"],
        "caption_column": data_config["caption_column"],
        "id_column": data_config.get("id_column"),
        "sample_identity": "source_shard_and_id_column",
        "train_subset_size": int(data_config["train_subset_size"]),
        "test_subset_size": int(data_config["test_subset_size"]),
        "image_size": int(data_config["image_size"]),
        "seed": seed,
        "partition_seed": int(data_config["partition_seed"]),
        "duplicate_skip_limit": int(data_config["streaming_duplicate_skip_limit"]),
        "shuffle_buffer_size": int(data_config["streaming_shuffle_buffer_size"]),
        "max_buffer_input_shards": int(
            data_config["streaming_max_buffer_input_shards"]
        ),
    }
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _rebalance_image_overlap(
    examples: list[dict[str, Any]],
    image_root: Path,
    *,
    train_size: int,
    test_size: int,
) -> int:
    """Keep identical decoded images on one side of the train/test boundary."""
    fingerprints = [
        _image_fingerprint(
            {"image": {"path": str(image_root / example["image_file"])}},
            "image",
        )
        for example in examples
    ]
    train_indices = [
        index for index, example in enumerate(examples) if example["split"] == "train"
    ]
    test_indices = [
        index for index, example in enumerate(examples) if example["split"] == "test"
    ]
    train_fingerprints = {fingerprints[index] for index in train_indices}
    test_fingerprints = {fingerprints[index] for index in test_indices}
    overlap_test_indices = [
        index for index in test_indices if fingerprints[index] in train_fingerprints
    ]
    if not overlap_test_indices:
        return 0

    train_fingerprint_counts = Counter(
        fingerprints[index] for index in train_indices
    )
    replacement_indices = [
        index
        for index in train_indices
        if fingerprints[index] not in test_fingerprints
        and train_fingerprint_counts[fingerprints[index]] == 1
    ]
    if len(replacement_indices) < len(overlap_test_indices):
        raise ValueError(
            "Could not rebalance the train/test split without identical images: "
            f"need {len(overlap_test_indices)} unique train replacements, found "
            f"{len(replacement_indices)}."
        )

    replacements = replacement_indices[: len(overlap_test_indices)]
    for index in overlap_test_indices:
        examples[index]["split"] = "train"
    for index in replacements:
        examples[index]["split"] = "test"

    final_train = [
        fingerprints[index]
        for index, example in enumerate(examples)
        if example["split"] == "train"
    ]
    final_test = [
        fingerprints[index]
        for index, example in enumerate(examples)
        if example["split"] == "test"
    ]
    if len(final_train) != train_size or len(final_test) != test_size:
        raise ValueError("Image-overlap rebalancing changed the configured split sizes.")
    if set(final_train) & set(final_test):
        raise ValueError("Image-overlap rebalancing did not remove all exact duplicates.")
    return len(overlap_test_indices)


def _read_cached_webdataset(
    cache_path: Path,
    *,
    expected_count: int,
    expected_train_count: int,
    expected_test_count: int,
    expected_shards: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_path = cache_path / "sampling_report.json"
    if not report_path.is_file():
        raise ValueError(
            f"Selected-data cache {cache_path} is incomplete: "
            "sampling_report.json is missing."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    examples = report.get("examples")
    if not isinstance(examples, list) or len(examples) != expected_count:
        raise ValueError(
            f"Selected-data cache {cache_path} has an unexpected example count."
        )

    ids = [str(example.get("id", "")) for example in examples]
    if any(not sample_id.strip() for sample_id in ids):
        raise ValueError("Selected-data cache contains an empty sample ID.")
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "Selected-data cache contains duplicate sample IDs: "
            + ", ".join(duplicate_ids[:10])
        )
    actual_shards = {str(example.get("source_shard", "")) for example in examples}
    if actual_shards != expected_shards:
        missing = sorted(expected_shards - actual_shards)
        unexpected = sorted(actual_shards - expected_shards)
        raise ValueError(
            "Selected-data cache shard coverage does not match configuration. "
            f"Missing: {missing}; unexpected: {unexpected}."
        )
    train_examples = [
        example for example in examples if example.get("split") == "train"
    ]
    test_examples = [
        example for example in examples if example.get("split") == "test"
    ]
    if len(train_examples) != expected_train_count or len(test_examples) != expected_test_count:
        raise ValueError(
            f"Selected-data cache split sizes are {len(train_examples)} train and "
            f"{len(test_examples)} test; expected {expected_train_count} and "
            f"{expected_test_count}."
        )
    if {example["id"] for example in train_examples} & {
        example["id"] for example in test_examples
    }:
        raise ValueError("Selected-data cache has sample IDs in both splits.")

    rebalanced_count = _rebalance_image_overlap(
        examples,
        cache_path,
        train_size=expected_train_count,
        test_size=expected_test_count,
    )
    if rebalanced_count:
        report["image_overlap_rebalanced_examples"] = (
            int(report.get("image_overlap_rebalanced_examples", 0))
            + rebalanced_count
        )
        temporary_report = report_path.with_suffix(".json.tmp")
        temporary_report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_report, report_path)

    cached_examples: list[dict[str, Any]] = []
    for example in examples:
        image_path = cache_path / str(example["image_file"])
        if not image_path.is_file():
            raise ValueError(
                f"Selected-data cache image is missing: {image_path}."
            )
        cached_examples.append(
            {
                **example,
                "image": {"path": str(image_path)},
                "caption": str(example["caption"]),
            }
        )
    report["cache_reused"] = True
    return cached_examples, report


def _write_webdataset_cache(
    selected_stream: Any,
    *,
    cache_path: Path,
    shard_names: list[str],
    dataset_id: str,
    dataset_revision: str | None,
    seed: int,
    partition_seed: int,
    buffer_size: int,
    max_buffer_input_shards: int,
    duplicate_skip_limit: int,
    train_size: int,
    test_size: int,
    image_column: str,
    caption_column: str,
    id_column: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    total_size = train_size + test_size
    selected: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    duplicate_ids: list[dict[str, str]] = []
    key_sources: dict[str, set[str]] = {}
    key_collision_examples: list[dict[str, Any]] = []
    key_collision_count = 0
    shard_counts: Counter[str] = Counter()

    with tempfile.TemporaryDirectory(
        prefix=f".{cache_path.name}.tmp-", dir=cache_path.parent
    ) as temporary_directory:
        staging_path = Path(temporary_directory)
        images_path = staging_path / "images"
        images_path.mkdir()
        scanned_records = 0
        for record in selected_stream:
            if scanned_records >= total_size + duplicate_skip_limit:
                break
            scanned_records += 1
            raw_key = _get_field(record, id_column)
            if raw_key is None or not str(raw_key).strip():
                raise ValueError("A WebDataset record has an empty sample ID.")
            raw_key = str(raw_key)
            source_shard = _webdataset_source_shard(record.get("__url__"))
            if source_shard not in shard_names:
                raise ValueError(
                    f"WebDataset record references unconfigured shard "
                    f"{source_shard!r}."
                )
            sample_id = f"{source_shard}::{raw_key}"
            prior_sources = key_sources.setdefault(raw_key, set())
            if prior_sources and source_shard not in prior_sources:
                key_collision_count += 1
                if len(key_collision_examples) < 100:
                    key_collision_examples.append(
                        {
                            "key": raw_key,
                            "source_shards": sorted(prior_sources | {source_shard}),
                        }
                    )
            prior_sources.add(source_shard)
            if sample_id in seen_ids:
                duplicate_ids.append(
                    {
                        "id": sample_id,
                        "first_source_shard": seen_ids[sample_id],
                        "duplicate_source_shard": source_shard,
                    }
                )
                continue
            seen_ids[sample_id] = source_shard
            shard_counts[source_shard] += 1
            caption = _as_caption(_get_field(record, caption_column))
            image_value = record[image_column]
            image_filename = f"images/{len(selected):08d}.jpg"
            image_path = staging_path / image_filename
            encoded_image = image_value.get("bytes") if isinstance(image_value, Mapping) else None
            if isinstance(encoded_image, bytes):
                image_path.write_bytes(encoded_image)
            else:
                _as_rgb_image(image_value).save(image_path, format="JPEG")
            selected.append(
                {
                    "id": sample_id,
                    "__key__": raw_key,
                    "source_shard": source_shard,
                    "image_file": image_filename,
                    "caption": caption,
                }
            )
            if len(selected) == total_size:
                break

        if len(selected) != total_size:
            raise ValueError(
                f"Streaming selection found only {len(selected)} unique examples "
                f"after scanning {scanned_records} candidates and skipping "
                f"{len(duplicate_ids)} duplicate IDs; expected {total_size}. "
                f"Duplicate IDs: {[item['id'] for item in duplicate_ids[:10]]}"
            )

        encountered_shards = set(shard_counts)
        missing_shards = sorted(set(shard_names) - encountered_shards)
        if missing_shards:
            raise ValueError(
                "Selected sample does not cover every configured shard; "
                f"{len(encountered_shards)}/{len(shard_names)} contributed. "
                f"Missing shards: {', '.join(missing_shards)}. "
                "No selected-data cache was committed."
            )

        random.Random(partition_seed).shuffle(selected)
        for index, example in enumerate(selected):
            example["split"] = "train" if index < train_size else "test"
        image_overlap_rebalanced = _rebalance_image_overlap(
            selected,
            staging_path,
            train_size=train_size,
            test_size=test_size,
        )
        train_ids = {
            example["id"] for example in selected if example["split"] == "train"
        }
        test_ids = {
            example["id"] for example in selected if example["split"] == "test"
        }
        if train_ids & test_ids:
            raise ValueError("Selected train and test splits contain duplicate IDs.")

        report: dict[str, Any] = {
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "sampling_method": "deterministic_shuffled_streaming_across_shards",
            "selection_seed": seed,
            "train_test_partition_seed": partition_seed,
            "streaming_shuffle_buffer_size": buffer_size,
            "max_buffer_input_shards": max_buffer_input_shards,
            "duplicate_skip_limit": duplicate_skip_limit,
            "shards_configured": len(shard_names),
            "shards_encountered": len(encountered_shards),
            "configured_shards": shard_names,
            "selected_examples_per_shard": {
                shard: shard_counts[shard] for shard in shard_names
            },
            "training_examples": train_size,
            "testing_examples": test_size,
            "duplicate_ids": duplicate_ids,
            "duplicate_key_collisions_across_shards": key_collision_count,
            "duplicate_key_collision_examples": key_collision_examples,
            "image_overlap_rebalanced_examples": image_overlap_rebalanced,
            "total_selected_images": len(selected),
            "all_shards_contributed": encountered_shards == set(shard_names),
            "examples": selected,
            "cache_reused": False,
        }
        (staging_path / "sampling_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        if cache_path.exists():
            raise FileExistsError(
                f"Selected-data cache appeared during sampling: {cache_path}."
            )
        os.replace(staging_path, cache_path)

    cached_examples = [
        {
            **example,
            "image": {"path": str(cache_path / example["image_file"])},
            "caption": example["caption"],
        }
        for example in selected
    ]
    return cached_examples, report


def _load_webdataset_selection(
    config: dict[str, Any],
    load_dataset: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data_config = config["data"]
    shard_count = int(data_config["shard_count"])
    if shard_count <= 0:
        raise ValueError("data.shard_count must be positive.")
    shard_urls = [
        data_config["shard_url_template"].format(i=index)
        for index in range(shard_count)
    ]
    shard_names = [Path(urlsplit(url).path).name for url in shard_urls]
    if len(set(shard_names)) != len(shard_names):
        raise ValueError("Configured WebDataset shard URLs must be unique.")

    train_size = int(data_config["train_subset_size"])
    test_size = int(data_config["test_subset_size"])
    total_size = train_size + test_size
    seed = int(config["reproducibility"]["seed"])
    partition_seed = int(data_config["partition_seed"])
    buffer_size = int(data_config["streaming_shuffle_buffer_size"])
    max_buffer_input_shards = int(data_config["streaming_max_buffer_input_shards"])
    duplicate_skip_limit = int(data_config["streaming_duplicate_skip_limit"])
    if buffer_size <= 0:
        raise ValueError("data.streaming_shuffle_buffer_size must be positive.")
    if max_buffer_input_shards != shard_count:
        raise ValueError(
            "data.streaming_max_buffer_input_shards must equal data.shard_count "
            "to shuffle across every configured shard."
        )
    if duplicate_skip_limit < 0:
        raise ValueError("data.streaming_duplicate_skip_limit cannot be negative.")

    cache_root = Path(data_config["cache_dir"]).expanduser()
    cache_path = cache_root / f"webdataset-{_webdataset_cache_key(data_config, seed)}"
    if cache_path.exists():
        cached_examples, report = _read_cached_webdataset(
            cache_path,
            expected_count=total_size,
            expected_train_count=train_size,
            expected_test_count=test_size,
            expected_shards=set(shard_names),
        )
    else:
        raw_stream = load_dataset(
            "webdataset",
            data_files={data_config["source_split"]: shard_urls},
            split=data_config["source_split"],
            streaming=True,
            revision=data_config.get("dataset_revision"),
        )
        _check_columns(
            raw_stream,
            data_config["image_column"],
            data_config["caption_column"],
            data_config.get("id_column"),
        )
        shuffle_parameters = inspect.signature(raw_stream.shuffle).parameters
        if "max_buffer_input_shards" not in shuffle_parameters:
            raise RuntimeError(
                "The installed Hugging Face Datasets version does not support "
                "max_buffer_input_shards; upgrade datasets before sampling."
            )
        selected_stream = raw_stream.shuffle(
            seed=seed,
            buffer_size=buffer_size,
            max_buffer_input_shards=max_buffer_input_shards,
        ).take(total_size + duplicate_skip_limit)
        cached_examples, report = _write_webdataset_cache(
            selected_stream,
            cache_path=cache_path,
            shard_names=shard_names,
            dataset_id=str(data_config["dataset_id"]),
            dataset_revision=data_config.get("dataset_revision"),
            seed=seed,
            partition_seed=partition_seed,
            buffer_size=buffer_size,
            max_buffer_input_shards=max_buffer_input_shards,
            duplicate_skip_limit=duplicate_skip_limit,
            train_size=train_size,
            test_size=test_size,
            image_column=data_config["image_column"],
            caption_column=data_config["caption_column"],
            id_column=data_config["id_column"],
        )

    train_records = [
        example for example in cached_examples if example["split"] == "train"
    ]
    test_records = [
        example for example in cached_examples if example["split"] == "test"
    ]
    if {example["id"] for example in train_records} & {
        example["id"] for example in test_records
    }:
        raise ValueError("Selected train and test splits contain duplicate IDs.")
    output_path = Path(config["paths"]["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    output_report = dict(report)
    output_report["cache_dir"] = str(cache_path)
    (output_path / "dataset_selection.json").write_text(
        json.dumps(output_report, indent=2) + "\n", encoding="utf-8"
    )
    report["cache_dir"] = str(cache_path)
    return train_records, test_records, report


def make_dataloader(
    dataset: Dataset[dict[str, Any]] | IterableDataset[dict[str, Any]],
    config: dict[str, Any],
    *,
    train: bool,
    epoch: int = 0,
) -> DataLoader[dict[str, Any]]:
    """Create a seeded loader; streaming stays single-worker to prevent duplication."""
    streaming = isinstance(dataset, IterableDataset)
    workers = 0 if streaming else int(config["data"]["num_workers"])
    generator = torch.Generator()
    generator.manual_seed(
        int(config["reproducibility"]["seed"]) + epoch * 2 + int(not train)
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=train and not streaming,
        num_workers=workers,
        generator=generator,
        drop_last=bool(config["training"]["drop_last"]),
    )
