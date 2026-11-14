"""Synthetic forward-pass and configuration tests; no pretrained downloads."""

from __future__ import annotations

import json
import io
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

import evaluate as evaluation_cli
import datasets
from text_to_image_gan.data import (
    _select_subset,
    _as_rgb_image,
    _rebalance_image_overlap,
    assert_subsets_disjoint,
    load_train_test_datasets,
    make_dataloader,
)
from text_to_image_gan.models.discriminator import DeeperDiscriminator
from text_to_image_gan.models.generator import DeeperGenerator
from text_to_image_gan.models.text_encoder import TextEncoder
from text_to_image_gan.alignment import CLIPAlignmentLoss
from text_to_image_gan.training import (
    _frozen_eval,
    _encode_caption_batch,
    _generate_with_caption_features,
    _save_checkpoint_if_due,
    _target_tensor,
)
from text_to_image_gan.evaluation import CLIPScorer
from text_to_image_gan.utils import (
    load_config,
    validate_config,
    validate_resume_config,
)

ROOT = Path(__file__).resolve().parents[1]


class MockTokenizer:
    def __call__(
        self,
        captions: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
        return_special_tokens_mask: bool = False,
    ) -> dict[str, torch.Tensor]:
        assert padding and truncation and return_tensors == "pt"
        token_ids = torch.zeros(len(captions), max_length, dtype=torch.long)
        attention_mask = torch.zeros_like(token_ids)
        special_tokens_mask = torch.zeros_like(token_ids)
        for index, caption in enumerate(captions):
            token_ids[index, 0] = 101
            token_ids[index, 1] = len(caption) % 99 + 1
            token_ids[index, 2] = 102
            attention_mask[index, :3] = 1
            special_tokens_mask[index, 0] = 1
            special_tokens_mask[index, 2] = 1
        result = {"input_ids": token_ids, "attention_mask": attention_mask}
        if return_special_tokens_mask:
            result["special_tokens_mask"] = special_tokens_mask
        return result


class MockBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768)
        self.embedding = nn.Embedding(128, 768)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        del attention_mask
        hidden = self.embedding(input_ids)
        return SimpleNamespace(
            pooler_output=hidden[:, 0], last_hidden_state=hidden
        )


def _config() -> dict[str, Any]:
    return load_config(ROOT / "config.yaml", require_dataset=False)


def test_text_encoder_uses_mocked_frozen_bert_and_trainable_projection() -> None:
    bert = MockBert()
    encoder = TextEncoder(
        max_length=32,
        output_dim=256,
        freeze_text_encoder=True,
        tokenizer=MockTokenizer(),
        bert=bert,
    )
    encoder.train()
    embeddings = encoder(["a red bird", "a blue flower"])
    assert embeddings.shape == (2, 256)
    assert not encoder.bert.training
    assert all(not parameter.requires_grad for parameter in encoder.bert.parameters())
    assert all(parameter.requires_grad for parameter in encoder.projection.parameters())
    embeddings.sum().backward()
    assert encoder.projection.weight.grad is not None
    assert bert.embedding.weight.grad is None


def test_text_encoder_returns_projected_words_and_masks_special_tokens() -> None:
    encoder = TextEncoder(
        max_length=8,
        output_dim=32,
        freeze_text_encoder=True,
        word_level_features=True,
        tokenizer=MockTokenizer(),
        bert=MockBert(),
    )
    sentence, words, word_mask = encoder(
        ["red flower", "yellow flower"], return_tokens=True
    )
    assert sentence.shape == (2, 32)
    assert words.shape == (2, 8, 32)
    assert word_mask.shape == (2, 8)
    assert word_mask[:, 0].sum() == 0
    assert word_mask[:, 1].all()
    assert word_mask[:, 2].sum() == 0


def test_generator_word_attention_uses_caption_tokens_and_backpropagates() -> None:
    generator = DeeperGenerator(
        text_dim=8,
        noise_dim=4,
        initial_channels=16,
        initial_size=4,
        channel_schedule=(16, 12, 8, 4, 3),
        batch_norm=False,
        word_level_attention=True,
    )
    words = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
    images = generator(
        torch.randn(2, 8), torch.randn(2, 4), word_features=words, word_mask=mask
    )
    assert images.shape == (2, 3, 64, 64)
    images.mean().backward()
    assert words.grad is not None
    assert torch.isfinite(words.grad).all()


def test_training_condition_helpers_pass_word_features_to_generator() -> None:
    class TokenEncoder(nn.Module):
        def forward(
            self, captions: list[str], *, return_tokens: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert return_tokens
            batch = len(captions)
            return (
                torch.randn(batch, 8),
                torch.randn(batch, 4, 8),
                torch.tensor([[True, True, False, False]]).expand(batch, -1),
            )

    generator = DeeperGenerator(
        text_dim=8,
        noise_dim=4,
        initial_channels=16,
        initial_size=4,
        channel_schedule=(16, 12, 8, 4, 3),
        batch_norm=False,
        word_level_attention=True,
    )
    sentence, words, mask = _encode_caption_batch(
        TokenEncoder(), generator, ["red flower", "yellow flower"]
    )
    images, kl_loss = _generate_with_caption_features(
        generator,
        sentence,
        torch.randn(2, 4),
        words,
        mask,
        return_kl=True,
    )
    assert images.shape == (2, 3, 64, 64)
    assert kl_loss.ndim == 0


class MockCLIPTokenizer:
    def __call__(self, captions: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        ids = torch.arange(len(captions), dtype=torch.long)[:, None]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class MockCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values.mean(dim=(2, 3)) * self.scale

    def get_text_features(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return torch.nn.functional.one_hot(input_ids[:, 0], num_classes=3).float()


def test_clip_alignment_loss_is_differentiable_and_freezes_teacher() -> None:
    model = MockCLIP()
    loss_module = CLIPAlignmentLoss(
        model=model, tokenizer=MockCLIPTokenizer(), temperature=0.1
    )
    source_images = torch.randn(3, 3, 64, 64, requires_grad=True)
    images = source_images.tanh()
    loss = loss_module(images, ["red flower", "yellow flower", "white flower"])
    loss.backward()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert source_images.grad is not None
    assert torch.isfinite(source_images.grad).all()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training is False


def test_generator_forward_produces_normalized_64px_images() -> None:
    config = _config()["generator"]
    generator = DeeperGenerator(
        text_dim=config["text_dim"],
        noise_dim=config["noise_dim"],
        initial_channels=config["initial_channels"],
        initial_size=config["initial_size"],
        channel_schedule=config["channel_schedule"],
        batch_norm=config["batch_norm"],
        hidden_activation=config["hidden_activation"],
        output_activation=config["output_activation"],
        output_batch_norm=config["output_batch_norm"],
    )
    images = generator(torch.randn(2, 256), torch.randn(2, 100))
    assert images.shape == (2, 3, 64, 64)
    assert float(images.min().detach()) >= -1.0
    assert float(images.max().detach()) <= 1.0


def test_discriminator_forward_returns_logits_and_optional_probabilities() -> None:
    config = _config()["discriminator"]
    discriminator = DeeperDiscriminator(
        image_channels=config["image_channels"],
        text_dim=config["text_dim"],
        image_channel_schedule=config["image_channel_schedule"],
        image_size=config["image_size"],
        feature_size=config["feature_size"],
        image_kernel_size=config["image_kernel_size"],
        image_stride=config["image_stride"],
        image_padding=config["image_padding"],
        batch_norm_image=config["batch_norm_image"],
        batch_norm_first_image_block=config["batch_norm_first_image_block"],
        joint_channels=config["joint_channels"],
        joint_kernel_sizes=config["joint_kernel_sizes"],
        joint_strides=config["joint_strides"],
        joint_paddings=config["joint_paddings"],
        batch_norm_joint=config["batch_norm_joint"],
        batch_norm_final_output=config["batch_norm_final_output"],
        activation=config["activation"],
        negative_slope=config["negative_slope"],
    )
    logits = discriminator(torch.randn(2, 3, 64, 64), torch.randn(2, 256))
    probabilities = discriminator.probabilities(logits)
    assert logits.shape == (2,)
    assert probabilities.shape == (2,)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_configuration_rejects_missing_dataset_id_and_bad_dimensions() -> None:
    config = _config()
    config["data"]["dataset_id"] = None
    with pytest.raises(ValueError, match="dataset_id"):
        validate_config(config)
    config["data"]["dataset_id"] = "example/dataset"
    config["generator"]["channel_schedule"] = [512, 256, 128]
    with pytest.raises(ValueError, match="four upsampling stages"):
        validate_config(config)


def test_configuration_accepts_approved_dimensions_without_dataset_for_model_tests() -> None:
    validate_config(_config(), require_dataset=False)


def test_webdataset_configuration_requires_all_shards_in_shuffle_buffer() -> None:
    config = _config()
    config["data"]["dataset_format"] = "webdataset"
    config["data"]["shard_count"] = 47
    config["data"]["streaming_max_buffer_input_shards"] = 46
    with pytest.raises(ValueError, match="must equal data.shard_count"):
        validate_config(config)


def test_identical_image_caption_pair_cannot_cross_train_test_boundary() -> None:
    image = Image.new("RGB", (4, 4), color=(100, 20, 200))
    record = {"image": image, "caption": "a sample"}
    with pytest.raises(ValueError, match="identical"):
        assert_subsets_disjoint([record], [record], "image")
    assert_subsets_disjoint([record], [record], "image", allow_overlap_count=1)


def test_rgb_image_conversion_accepts_encoded_jpeg_bytes() -> None:
    image = Image.new("RGB", (8, 6), color=(10, 20, 30))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")
    decoded = _as_rgb_image(encoded.getvalue())
    assert decoded.mode == "RGB"
    assert decoded.size == (8, 6)


def test_cached_partition_rebalances_exact_image_overlap(tmp_path: Path) -> None:
    examples = []
    for index, (split, color) in enumerate(
        (
            ("train", (10, 20, 30)),
            ("train", (40, 50, 60)),
            ("train", (70, 80, 90)),
            ("test", (10, 20, 30)),
        )
    ):
        image_path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (8, 8), color).save(image_path)
        examples.append(
            {"id": str(index), "image_file": image_path.name, "split": split}
        )
    rebalanced = _rebalance_image_overlap(
        examples, tmp_path, train_size=3, test_size=1
    )
    assert rebalanced == 1
    assert sum(example["split"] == "train" for example in examples) == 3
    assert sum(example["split"] == "test" for example in examples) == 1
    assert examples[0]["split"] == "train"
    assert examples[3]["split"] == "train"


def test_training_loader_includes_incomplete_final_batch() -> None:
    config = _config()
    config["training"]["batch_size"] = 4
    config["training"]["drop_last"] = False
    config["data"]["streaming"] = False
    config["data"]["num_workers"] = 0
    examples = [
        {"image": torch.full((3, 2, 2), float(index)), "caption": str(index)}
        for index in range(10)
    ]
    loader = make_dataloader(examples, config, train=True, epoch=0)
    seen = [caption for batch in loader for caption in batch["caption"]]
    assert len(seen) == 10
    assert set(seen) == {str(index) for index in range(10)}


def test_discriminator_and_generator_targets_are_separate() -> None:
    config = _config()["training"]
    logits = torch.zeros(3)
    discriminator_real_targets = _target_tensor(logits, config["real_label"])
    generator_targets = _target_tensor(logits, config["generator_target_label"])
    assert torch.all(discriminator_real_targets == 0.9)
    assert torch.all(generator_targets == 1.0)
    assert not torch.equal(discriminator_real_targets, generator_targets)


def test_discriminator_is_frozen_but_differentiable_to_generator_input() -> None:
    discriminator = nn.Linear(3, 1)
    discriminator.train()
    generator_input = torch.randn(2, 3, requires_grad=True)
    with _frozen_eval(discriminator):
        assert not discriminator.training
        logits = discriminator(generator_input)
        logits.sum().backward()
        assert all(parameter.grad is None for parameter in discriminator.parameters())
    assert discriminator.training
    assert all(parameter.requires_grad for parameter in discriminator.parameters())
    assert generator_input.grad is not None


def test_seeded_map_subset_selection_is_deterministic() -> None:
    records = [{"id": f"example-{index}"} for index in range(40)]
    first, first_provenance = _select_subset(
        records, size=9, seed=123, streaming=False, id_column="id"
    )
    second, second_provenance = _select_subset(
        records, size=9, seed=123, streaming=False, id_column="id"
    )
    assert first == second
    assert first_provenance == second_provenance
    assert first_provenance["indices"] == [int(item["id"].split("-")[1]) for item in first]
    assert first_provenance["ids"] == [item["id"] for item in first]


def test_dataset_loader_writes_selected_indices_and_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MockMapDataset:
        column_names = ["id", "image", "caption"]

        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.records[index]

        def select(self, indices: list[int]) -> MockMapDataset:
            return MockMapDataset([self.records[index] for index in indices])

    train_records = [
        {
            "id": f"train-{index}",
            "image": Image.new("RGB", (4, 4), (index, 0, 0)),
            "caption": f"train caption {index}",
        }
        for index in range(8)
    ]
    test_records = [
        {
            "id": f"test-{index}",
            "image": Image.new("RGB", (4, 4), (100 + index, 0, 0)),
            "caption": f"test caption {index}",
        }
        for index in range(6)
    ]
    def mock_load_dataset(
        _dataset_id: str, *, split: str, streaming: bool, **kwargs: Any
    ) -> MockMapDataset:
        del _dataset_id
        del streaming
        del kwargs
        return MockMapDataset(train_records if split == "train" else test_records)

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)
    config = _config()
    config["data"].update(
        {
            "dataset_id": "test/synthetic",
            "dataset_format": "huggingface",
            "image_column": "image",
            "id_column": "id",
            "caption_column": "caption",
            "streaming": False,
            "train_subset_size": 5,
            "test_subset_size": 3,
        }
    )
    config["paths"]["output_dir"] = str(tmp_path)
    train_subset, test_subset = load_train_test_datasets(config)
    assert len(train_subset) == 5
    assert len(test_subset) == 3
    manifest = json.loads(
        (tmp_path / "dataset_selection.json").read_text(encoding="utf-8")
    )
    assert manifest["train"]["method"] == "seeded_random_sample_without_replacement"
    assert len(manifest["train"]["indices"]) == 5
    assert len(manifest["train"]["ids"]) == 5
    assert len(manifest["test"]["indices"]) == 3
    assert len(manifest["test"]["ids"]) == 3


def _webdataset_test_config(tmp_path: Path) -> dict[str, Any]:
    config = _config()
    config["data"].update(
        {
            "dataset_id": "jackyhate/text-to-image-2M",
            "dataset_format": "webdataset",
            "dataset_revision": "revision",
            "shard_url_template": "https://example.invalid/data_{i:06d}.tar",
            "shard_count": 47,
            "source_split": "train",
            "train_split": "train",
            "test_split": "test",
            "image_column": "jpg",
            "caption_column": "json.prompt",
            "id_column": "__key__",
            "streaming": True,
            "streaming_shuffle_buffer_size": 100,
            "streaming_max_buffer_input_shards": 47,
            "streaming_duplicate_skip_limit": 1200,
            "partition_seed": 71,
            "cache_dir": str(tmp_path / "cache"),
            "train_subset_size": 40,
            "test_subset_size": 7,
        }
    )
    config["paths"]["output_dir"] = str(tmp_path / "output")
    return config


def _webdataset_test_records() -> list[dict[str, Any]]:
    return [
        {
            "__key__": f"sample-{index:05d}",
            "__url__": f"https://example.invalid/data_{index:06d}.tar",
            "jpg": Image.new(
                "RGB", (8, 8), (index, index * 3 % 256, index * 7 % 256)
            ),
            "json": {"prompt": f"prompt {index}"},
        }
        for index in range(47)
    ]


def test_webdataset_uses_all_ordered_shards_and_reuses_cached_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MockStream:
        column_names = ["__key__", "__url__", "jpg", "json"]

        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def shuffle(
            self,
            *,
            seed: int,
            buffer_size: int,
            max_buffer_input_shards: int,
        ) -> MockStream:
            assert seed == 42
            assert buffer_size == 100
            assert max_buffer_input_shards == 47
            shuffled_records = list(self.records)
            random.Random(seed).shuffle(shuffled_records)
            return MockStream(shuffled_records)

        def take(self, count: int) -> MockStream:
            return MockStream(self.records[:count])

        def __iter__(self):
            return iter(self.records)

    records = _webdataset_test_records()
    load_calls: list[dict[str, Any]] = []

    def mock_load_dataset(builder: str, **kwargs: Any) -> MockStream:
        load_calls.append({"builder": builder, **kwargs})
        return MockStream(records)

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)
    config = _webdataset_test_config(tmp_path)
    train_dataset, test_dataset = load_train_test_datasets(config)
    train_rows, test_rows = list(train_dataset), list(test_dataset)
    assert len(train_rows) == 40
    assert len(test_rows) == 7
    assert all(row["image"].shape == (3, 64, 64) for row in train_rows + test_rows)
    assert train_rows[0]["caption"].startswith("prompt ")
    assert load_calls[0]["builder"] == "webdataset"
    assert load_calls[0]["split"] == "train"
    assert load_calls[0]["streaming"] is True
    assert load_calls[0]["revision"] == "revision"
    expected_urls = [
        f"https://example.invalid/data_{index:06d}.tar" for index in range(47)
    ]
    assert load_calls[0]["data_files"]["train"] == expected_urls
    assert len(set(load_calls[0]["data_files"]["train"])) == 47

    manifest_path = tmp_path / "output" / "dataset_selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sampling_method"] == (
        "deterministic_shuffled_streaming_across_shards"
    )
    assert manifest["shards_configured"] == 47
    assert manifest["shards_encountered"] == 47
    assert manifest["selected_examples_per_shard"] == {
        f"data_{index:06d}.tar": 1 for index in range(47)
    }
    assert manifest["training_examples"] == 40
    assert manifest["testing_examples"] == 7
    assert manifest["duplicate_ids"] == []
    assert manifest["total_selected_images"] == 47
    assert manifest["all_shards_contributed"] is True
    assert {row["id"] for row in manifest["examples"] if row["split"] == "train"}.isdisjoint(
        row["id"] for row in manifest["examples"] if row["split"] == "test"
    )
    assert all(
        {
            "id",
            "__key__",
            "source_shard",
            "split",
            "image_file",
            "caption",
        } <= set(example)
        for example in manifest["examples"]
    )
    assert len(list((Path(manifest["cache_dir"]) / "images").iterdir())) == 47

    def fail_if_remote_load(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A valid selected-data cache should avoid remote loading.")

    monkeypatch.setattr(datasets, "load_dataset", fail_if_remote_load)
    cached_train, cached_test = load_train_test_datasets(config)
    assert len(cached_train) == 40
    assert len(cached_test) == 7
    cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert cached_manifest["cache_reused"] is True


def test_webdataset_selection_is_deterministic_with_partition_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MockStream:
        column_names = ["__key__", "__url__", "jpg", "json"]

        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def shuffle(
            self,
            *,
            seed: int,
            buffer_size: int,
            max_buffer_input_shards: int,
        ) -> MockStream:
            del buffer_size
            assert max_buffer_input_shards == 47
            ordered = list(self.records)
            random.Random(seed).shuffle(ordered)
            return MockStream(ordered)

        def take(self, count: int) -> MockStream:
            return MockStream(self.records[:count])

        def __iter__(self):
            return iter(self.records)

    def mock_load_dataset(_builder: str, **_kwargs: Any) -> MockStream:
        return MockStream(_webdataset_test_records())

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)
    first_config = _webdataset_test_config(tmp_path / "first")
    second_config = _webdataset_test_config(tmp_path / "second")
    load_train_test_datasets(first_config)
    load_train_test_datasets(second_config)
    first = json.loads(
        (tmp_path / "first/output/dataset_selection.json").read_text()
    )
    second = json.loads(
        (tmp_path / "second/output/dataset_selection.json").read_text()
    )
    assert first["examples"] == second["examples"]


def test_webdataset_distinguishes_keys_reused_across_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MockStream:
        column_names = ["__key__", "__url__", "jpg", "json"]

        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def shuffle(
            self,
            *,
            seed: int,
            buffer_size: int,
            max_buffer_input_shards: int,
        ) -> MockStream:
            del seed, buffer_size, max_buffer_input_shards
            return self

        def take(self, count: int) -> MockStream:
            return MockStream(self.records[:count])

        def __iter__(self):
            return iter(self.records)

    records = _webdataset_test_records()
    duplicate = dict(records[0])
    duplicate["__url__"] = "https://example.invalid/data_000046.tar"
    duplicate["jpg"] = Image.new("RGB", (8, 8), (250, 10, 70))
    records.insert(1, duplicate)
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *_args, **_kwargs: MockStream(records),
    )
    config = _webdataset_test_config(tmp_path)
    train_dataset, test_dataset = load_train_test_datasets(config)
    assert len(train_dataset) == 40
    assert len(test_dataset) == 7
    manifest = json.loads(
        (tmp_path / "output" / "dataset_selection.json").read_text()
    )
    assert len(manifest["examples"]) == 47
    assert len({example["id"] for example in manifest["examples"]}) == 47
    assert manifest["duplicate_ids"] == []
    assert manifest["duplicate_key_collisions_across_shards"] == 1
    assert manifest["duplicate_key_collision_examples"] == [
        {
            "key": "sample-00000",
            "source_shards": ["data_000000.tar", "data_000046.tar"],
        }
    ]


def test_webdataset_rejects_duplicate_ids_and_inadequate_shard_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MockStream:
        column_names = ["__key__", "__url__", "jpg", "json"]

        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def shuffle(
            self,
            *,
            seed: int,
            buffer_size: int,
            max_buffer_input_shards: int,
        ) -> MockStream:
            del seed, buffer_size, max_buffer_input_shards
            return self

        def take(self, count: int) -> MockStream:
            return MockStream(self.records[:count])

        def __iter__(self):
            return iter(self.records)

    config = _webdataset_test_config(tmp_path)
    duplicate_records = _webdataset_test_records()
    duplicate_records[-1]["__key__"] = duplicate_records[0]["__key__"]
    duplicate_records[-1]["__url__"] = duplicate_records[0]["__url__"]
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *_args, **_kwargs: MockStream(duplicate_records),
    )
    with pytest.raises(ValueError, match="only 46 unique examples"):
        load_train_test_datasets(config)

    coverage_config = _webdataset_test_config(tmp_path / "coverage")
    missing_shard_records = _webdataset_test_records()
    missing_shard_records[-1]["__url__"] = missing_shard_records[-2]["__url__"]
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *_args, **_kwargs: MockStream(missing_shard_records),
    )
    with pytest.raises(ValueError, match="does not cover every configured shard"):
        load_train_test_datasets(coverage_config)


def test_seeded_streaming_reservoir_selection_is_deterministic() -> None:
    records = [{"id": f"example-{index}"} for index in range(100)]
    first, first_provenance = _select_subset(
        iter(records), size=12, seed=37, streaming=True, id_column="id"
    )
    second, second_provenance = _select_subset(
        iter(records), size=12, seed=37, streaming=True, id_column="id"
    )
    assert first == second
    assert first_provenance == second_provenance
    assert len(first_provenance["indices"]) == len(set(first_provenance["indices"])) == 12


def test_resume_config_allows_only_epoch_increase_and_output_path_changes() -> None:
    saved = _config()
    active = _config()
    active["training"]["epochs"] += 20
    active["paths"]["output_dir"] = "new-output"
    active["paths"]["checkpoint_dir"] = "new-checkpoints"
    changes = validate_resume_config(saved, active, checkpoint_epoch=5)
    assert any("training.epochs increased" in change for change in changes)
    assert any("paths.output_dir changed" in change for change in changes)
    assert any("paths.checkpoint_dir changed" in change for change in changes)

    incompatible = _config()
    incompatible["training"]["generator_lr"] *= 2
    with pytest.raises(ValueError, match="training.generator_lr"):
        validate_resume_config(saved, incompatible, checkpoint_epoch=5)

    reduced = _config()
    reduced["training"]["epochs"] -= 1
    with pytest.raises(ValueError, match="training.epochs"):
        validate_resume_config(saved, reduced, checkpoint_epoch=5)


def test_final_epoch_checkpoint_is_saved_off_interval(tmp_path: Path) -> None:
    config = _config()
    checkpoint_path = _save_checkpoint_if_due(
        {"epoch": 7, "config": config},
        tmp_path,
        config,
        completed_epoch=7,
        total_epochs=7,
        interval=5,
    )
    assert checkpoint_path is not None
    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".yaml").exists()
    assert torch.load(checkpoint_path, weights_only=False)["epoch"] == 7


def test_evaluation_noise_and_scores_repeat_with_fixed_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config["data"]["dataset_id"] = "test/synthetic"
    config["paths"]["output_dir"] = str(tmp_path)
    config["evaluation"]["seed"] = 219

    class MockTextEncoder:
        def __call__(self, captions: list[str]) -> torch.Tensor:
            return torch.ones(len(captions), 2)

    class MockGenerator:
        def __call__(
            self, text_embeddings: torch.Tensor, noise: torch.Tensor
        ) -> torch.Tensor:
            del text_embeddings
            return noise[:, :1, None, None].expand(-1, 3, 4, 4)

    class MockScorer:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def metadata(self) -> dict[str, Any]:
            return {"model_id": self.kwargs["model_name"], "mocked": True}

        def score(
            self, images: list[torch.Tensor], captions: list[str]
        ) -> list[float]:
            del captions
            return [float(image.mean()) for image in images]

    batches = [
        {"caption": ["a red object", "a blue object"]},
        {"caption": ["a green object"]},
    ]
    def mock_load_trained_models(
        _checkpoint: str, _device: torch.device
    ) -> tuple[dict[str, Any], MockTextEncoder, MockGenerator, None]:
        del _checkpoint, _device
        return (
            config,
            MockTextEncoder(),
            MockGenerator(),
            None,
        )

    def mock_load_datasets(
        _supplied_config: dict[str, Any],
    ) -> tuple[None, None]:
        del _supplied_config
        return None, None

    def mock_make_dataloader(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        del _args, _kwargs
        return batches

    monkeypatch.setattr(
        evaluation_cli, "load_trained_models", mock_load_trained_models
    )
    monkeypatch.setattr(
        evaluation_cli, "load_train_test_datasets", mock_load_datasets
    )
    monkeypatch.setattr(
        evaluation_cli, "make_dataloader", mock_make_dataloader
    )
    monkeypatch.setattr(evaluation_cli, "CLIPScorer", MockScorer)

    first = evaluation_cli.evaluate_checkpoint("synthetic.pt", "cpu")
    second = evaluation_cli.evaluate_checkpoint("synthetic.pt", "cpu")
    assert first["generated_clip_cosine_similarity"] == second[
        "generated_clip_cosine_similarity"
    ]
    assert first["evaluation_seed"] == 219
    assert first["evaluation_metadata"]["model_id"] == config["evaluation"]["clip_model_name"]


def test_clip_scorer_records_and_uses_mocked_preprocessing() -> None:
    class MockImageProcessor:
        do_resize = True
        size = {"shortest_edge": 4}
        resample = 3
        do_center_crop = True
        crop_size = {"height": 4, "width": 4}
        do_rescale = True
        rescale_factor = 1 / 255
        do_normalize = True
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]

    class MockClipTokenizer:
        name_or_path = "mock/clip-tokenizer"
        model_max_length = 77

    class MockProcessor:
        image_processor = MockImageProcessor()
        tokenizer = MockClipTokenizer()

        def __init__(self) -> None:
            self.last_call: dict[str, Any] | None = None

        def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
            self.last_call = kwargs
            batch = len(kwargs["text"])
            return {
                "pixel_values": torch.zeros(batch, 3, 4, 4),
                "input_ids": torch.ones(batch, 2, dtype=torch.long),
                "attention_mask": torch.ones(batch, 2, dtype=torch.long),
            }

    class MockClipModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                _name_or_path="mock/clip-model", model_type="clip"
            )

        def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                (pixel_values.mean(dim=(1, 2, 3)), torch.ones(len(pixel_values))),
                dim=1,
            )

        def get_text_features(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            del attention_mask
            return torch.stack(
                (input_ids.float().mean(dim=1), torch.ones(len(input_ids))), dim=1
            )

    processor = MockProcessor()
    scorer = CLIPScorer(
        model_name="mock/clip-model",
        model=MockClipModel(),
        processor=processor,
        text_padding=True,
        text_truncation=True,
        text_max_length=23,
    )
    scores = scorer.score([torch.zeros(3, 4, 4)], ["a caption"])
    metadata = scorer.metadata()
    assert len(scores) == 1
    assert processor.last_call is not None
    assert processor.last_call["padding"] is True
    assert processor.last_call["truncation"] is True
    assert processor.last_call["max_length"] == 23
    assert metadata["model_config_name_or_path"] == "mock/clip-model"
    assert metadata["image_preprocessing"]["crop_size"] == {
        "height": 4,
        "width": 4,
    }
    assert metadata["text_preprocessing"]["max_length"] == 23
