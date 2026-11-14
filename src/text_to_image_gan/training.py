"""GAN training loop, checkpointing, and resume support."""

from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image

from text_to_image_gan.alignment import CLIPAlignmentLoss
from text_to_image_gan.data import load_train_test_datasets, make_dataloader
from text_to_image_gan.evaluation import plot_losses
from text_to_image_gan.models.discriminator import DeeperDiscriminator
from text_to_image_gan.models.generator import DeeperGenerator
from text_to_image_gan.models.text_encoder import TextEncoder
from text_to_image_gan.utils import (
    load_config,
    resolve_device,
    set_seed,
    validate_resume_config,
    write_config,
)


def build_models(config: dict[str, Any]) -> tuple[TextEncoder, DeeperGenerator, DeeperDiscriminator]:
    """Construct model components from validated configuration."""
    text_config = config["text_encoder"]
    generator_config = config["generator"]
    discriminator_config = config["discriminator"]
    text_encoder = TextEncoder(
        model_name=text_config["model_name"],
        max_length=int(text_config["max_length"]),
        output_dim=int(text_config["output_dim"]),
        freeze_text_encoder=bool(text_config["freeze_text_encoder"]),
        word_level_features=bool(generator_config.get("word_level_attention", False)),
    )
    generator = DeeperGenerator(
        text_dim=int(generator_config["text_dim"]),
        noise_dim=int(generator_config["noise_dim"]),
        initial_channels=int(generator_config["initial_channels"]),
        initial_size=int(generator_config["initial_size"]),
        channel_schedule=generator_config["channel_schedule"],
        batch_norm=bool(generator_config["batch_norm"]),
        hidden_activation=generator_config["hidden_activation"],
        output_activation=generator_config["output_activation"],
        output_batch_norm=bool(generator_config["output_batch_norm"]),
        conditioning_augmentation=bool(
            generator_config.get("conditioning_augmentation", False)
        ),
        word_level_attention=bool(generator_config.get("word_level_attention", False)),
    )
    discriminator = DeeperDiscriminator(
        image_channels=int(discriminator_config["image_channels"]),
        text_dim=int(discriminator_config["text_dim"]),
        image_channel_schedule=discriminator_config["image_channel_schedule"],
        image_size=int(discriminator_config["image_size"]),
        feature_size=int(discriminator_config["feature_size"]),
        image_kernel_size=int(discriminator_config["image_kernel_size"]),
        image_stride=int(discriminator_config["image_stride"]),
        image_padding=int(discriminator_config["image_padding"]),
        batch_norm_image=bool(discriminator_config["batch_norm_image"]),
        batch_norm_first_image_block=bool(
            discriminator_config["batch_norm_first_image_block"]
        ),
        joint_channels=discriminator_config["joint_channels"],
        joint_kernel_sizes=discriminator_config["joint_kernel_sizes"],
        joint_strides=discriminator_config["joint_strides"],
        joint_paddings=discriminator_config["joint_paddings"],
        batch_norm_joint=bool(discriminator_config["batch_norm_joint"]),
        batch_norm_final_output=bool(discriminator_config["batch_norm_final_output"]),
        activation=discriminator_config["activation"],
        negative_slope=float(discriminator_config["negative_slope"]),
    )
    return text_encoder, generator, discriminator


def _encode_caption_batch(
    text_encoder: TextEncoder,
    generator: DeeperGenerator,
    captions: list[str],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Encode the sentence and optional BERT token features required by the G."""
    if generator.word_level_attention_enabled:
        sentence, words, word_mask = text_encoder(captions, return_tokens=True)
        return sentence, words, word_mask
    return text_encoder(captions), None, None


def _generate_with_caption_features(
    generator: DeeperGenerator,
    sentence: torch.Tensor,
    noise: torch.Tensor,
    word_features: torch.Tensor | None = None,
    word_mask: torch.Tensor | None = None,
    *,
    return_kl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Call the generator consistently for sentence-only and token-attention modes."""
    kwargs: dict[str, Any] = {}
    if word_features is not None and word_mask is not None:
        kwargs.update(word_features=word_features, word_mask=word_mask)
    if return_kl:
        kwargs["return_kl"] = True
    return generator(sentence, noise, **kwargs)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _target_tensor(logits: torch.Tensor, label: float) -> torch.Tensor:
    return torch.full_like(logits, label)


def _require_finite_tensor(tensor: torch.Tensor, context: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{context} contains non-finite values; stopping.")


def _require_finite_gradients(
    parameters: list[nn.Parameter], context: str
) -> None:
    for parameter in parameters:
        if parameter.grad is not None:
            _require_finite_tensor(parameter.grad, f"{context} gradient")


def _require_finite_parameters(
    parameters: list[nn.Parameter], context: str
) -> None:
    for parameter in parameters:
        _require_finite_tensor(parameter, f"{context} parameter")


@contextmanager
def _frozen_eval(module: nn.Module) -> Generator[None, None, None]:
    """Temporarily disable parameter gradients and training behavior."""
    was_training = module.training
    original_requires_grad = [parameter.requires_grad for parameter in module.parameters()]
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
        module.train(was_training)


def _save_checkpoint_if_due(
    payload: dict[str, Any],
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    *,
    completed_epoch: int,
    total_epochs: int,
    interval: int,
) -> Path | None:
    """Save interval checkpoints and always save the final completed epoch."""
    if completed_epoch % interval != 0 and completed_epoch != total_epochs:
        return None
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"checkpoint_epoch_{completed_epoch:04d}.pt"
    torch.save(payload, checkpoint_path)
    write_config(config, checkpoint_path.with_suffix(".yaml"))
    return checkpoint_path


def _checkpoint_payload(
    *,
    epoch: int,
    config: dict[str, Any],
    text_encoder: TextEncoder,
    generator: DeeperGenerator,
    discriminator: DeeperDiscriminator,
    generator_optimizer: Adam,
    discriminator_optimizer: Adam,
    generator_scheduler: StepLR,
    discriminator_scheduler: StepLR,
    losses: dict[str, list[float]],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "config": config,
        "text_encoder": text_encoder.state_dict(),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "generator_scheduler": generator_scheduler.state_dict(),
        "discriminator_scheduler": discriminator_scheduler.state_dict(),
        "losses": losses,
        "rng_state": _capture_rng_state(),
    }


def _save_sample_grid(
    text_encoder: TextEncoder,
    generator: DeeperGenerator,
    captions: list[str],
    noise: torch.Tensor,
    path: Path,
) -> None:
    text_encoder_was_training = text_encoder.training
    generator_was_training = generator.training
    text_encoder.eval()
    generator.eval()
    with torch.no_grad():
        sentence, words, word_mask = _encode_caption_batch(
            text_encoder, generator, captions
        )
        images = _generate_with_caption_features(
            generator, sentence, noise[: len(captions)], words, word_mask
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, path, nrow=max(1, int(len(captions) ** 0.5)), normalize=True, value_range=(-1, 1))
    text_encoder.train(text_encoder_was_training)
    generator.train(generator_was_training)


def train(config_path: str | Path, device_name: str | None = None) -> None:
    """Train the GAN using configured data, saving resumable checkpoints."""
    config = load_config(config_path)
    training_config = config["training"]
    resume_checkpoint: dict[str, Any] | None = None
    resume_path = training_config.get("resume_from")
    if resume_path:
        resume_checkpoint = torch.load(
            resume_path,
            # Keep RNG state tensors on CPU; torch.set_rng_state requires a CPU
            # ByteTensor. Model and optimizer states are moved to the selected
            # device by their respective load_state_dict methods below.
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_config = resume_checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError("Resume checkpoint does not contain its saved configuration.")
        approved_changes = validate_resume_config(
            checkpoint_config,
            config,
            checkpoint_epoch=int(resume_checkpoint["epoch"]),
        )
        for change in approved_changes:
            print(f"Resume configuration change approved: {change}")
    set_seed(int(config["reproducibility"]["seed"]))
    device = resolve_device(device_name)

    text_encoder, generator, discriminator = build_models(config)
    text_encoder.to(device)
    generator.to(device)
    discriminator.to(device)
    betas = tuple(float(value) for value in training_config["adam_betas"])
    generator_parameters = _optimizer_parameters(text_encoder) + _optimizer_parameters(
        generator
    )
    generator_optimizer = Adam(
        generator_parameters,
        lr=float(training_config["generator_lr"]),
        betas=betas,
    )
    discriminator_optimizer = Adam(
        _optimizer_parameters(discriminator),
        lr=float(training_config["discriminator_lr"]),
        betas=betas,
    )
    generator_scheduler = StepLR(
        generator_optimizer,
        step_size=int(training_config["scheduler_step_size"]),
        gamma=float(training_config["scheduler_gamma"]),
    )
    discriminator_scheduler = StepLR(
        discriminator_optimizer,
        step_size=int(training_config["scheduler_step_size"]),
        gamma=float(training_config["scheduler_gamma"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    clip_alignment_weight = float(training_config.get("clip_alignment_weight", 0.0))
    clip_alignment = None
    if clip_alignment_weight > 0.0:
        clip_alignment = CLIPAlignmentLoss(
            model_name=str(
                training_config.get(
                    "clip_alignment_model_name",
                    config["evaluation"]["clip_model_name"],
                )
            ),
            temperature=float(training_config.get("clip_alignment_temperature", 0.07)),
        ).to(device)
    train_dataset, _ = load_train_test_datasets(config)

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, checkpoint_dir / "config.yaml")
    samples_dir = Path(config["paths"]["samples_dir"])
    losses: dict[str, list[float]] = {
        "generator": [],
        "discriminator": [],
        "clip_alignment": [],
    }
    start_epoch = 0
    if resume_checkpoint is not None:
        text_encoder.load_state_dict(resume_checkpoint["text_encoder"])
        generator.load_state_dict(resume_checkpoint["generator"])
        discriminator.load_state_dict(resume_checkpoint["discriminator"])
        generator_optimizer.load_state_dict(resume_checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(
            resume_checkpoint["discriminator_optimizer"]
        )
        generator_scheduler.load_state_dict(resume_checkpoint["generator_scheduler"])
        discriminator_scheduler.load_state_dict(
            resume_checkpoint["discriminator_scheduler"]
        )
        losses = resume_checkpoint["losses"]
        start_epoch = int(resume_checkpoint["epoch"])
        _restore_rng_state(resume_checkpoint["rng_state"])

    fixed_captions: list[str] | None = None
    sample_generator = torch.Generator(device=device)
    sample_generator.manual_seed(int(config["reproducibility"]["seed"]) + 10_000)
    sample_count = int(training_config["sample_grid_size"])
    fixed_noise = torch.randn(
        sample_count, int(config["generator"]["noise_dim"]),
        generator=sample_generator,
        device=device,
    )

    for epoch_index in range(start_epoch, int(training_config["epochs"])):
        text_encoder.train()
        generator.train()
        discriminator.train()
        epoch_generator_loss = 0.0
        epoch_discriminator_loss = 0.0
        epoch_alignment_loss = 0.0
        generator_batches = 0
        discriminator_batches = 0
        loader = make_dataloader(
            train_dataset, config, train=True, epoch=epoch_index
        )
        for batch in loader:
            real_images = batch["image"].to(device)
            captions = batch["caption"]
            if fixed_captions is None:
                fixed_captions = list(captions[:sample_count])

            with torch.no_grad():
                detached_text, detached_words, detached_word_mask = _encode_caption_batch(
                    text_encoder, generator, captions
                )
                fake_images = _generate_with_caption_features(
                    generator,
                    detached_text,
                    torch.randn(
                        len(captions),
                        int(config["generator"]["noise_dim"]),
                        device=device,
                    ),
                    detached_words,
                    detached_word_mask,
                )
            sigma = float(training_config["image_noise_sigma"])
            real_input = real_images + sigma * torch.randn_like(real_images)
            fake_input = fake_images.detach() + sigma * torch.randn_like(fake_images)
            real_logits = discriminator(real_input, detached_text)
            fake_logits = discriminator(fake_input, detached_text)
            real_targets = _target_tensor(
                real_logits, float(training_config["real_label"])
            )
            fake_targets = _target_tensor(
                fake_logits, float(training_config["fake_label"])
            )
            discriminator_loss = criterion(
                real_logits, real_targets
            ) + criterion(fake_logits, fake_targets)
            if bool(training_config.get("matching_aware_discriminator", False)):
                # Rotate captions within the batch to create real but mismatched pairs.
                # A batch of one cannot provide a non-identity mismatch.
                if len(captions) > 1:
                    mismatched_text = detached_text.roll(shifts=1, dims=0)
                    mismatched_logits = discriminator(real_input, mismatched_text)
                    mismatched_targets = _target_tensor(
                        mismatched_logits, float(training_config["fake_label"])
                    )
                    discriminator_loss = discriminator_loss + float(
                        training_config.get("mismatched_caption_weight", 1.0)
                    ) * criterion(mismatched_logits, mismatched_targets)
            _require_finite_tensor(discriminator_loss, "Discriminator loss")
            discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_loss.backward()
            discriminator_parameters = _optimizer_parameters(discriminator)
            _require_finite_gradients(
                discriminator_parameters, "Discriminator"
            )
            nn.utils.clip_grad_norm_(
                discriminator_parameters,
                float(training_config["gradient_clip_norm"]),
            )
            discriminator_optimizer.step()
            _require_finite_parameters(
                discriminator_parameters, "Discriminator"
            )
            discriminator_optimizer.zero_grad(set_to_none=True)
            epoch_discriminator_loss += float(discriminator_loss.detach())
            discriminator_batches += 1

            with _frozen_eval(discriminator):
                for _ in range(
                    int(training_config["generator_updates_per_discriminator_update"])
                ):
                    text_embedding, word_features, word_mask = _encode_caption_batch(
                        text_encoder, generator, captions
                    )
                    generated, conditioning_kl = _generate_with_caption_features(
                        generator,
                        text_embedding,
                        torch.randn(
                            len(captions),
                            int(config["generator"]["noise_dim"]),
                            device=device,
                        ),
                        word_features,
                        word_mask,
                        return_kl=True,
                    )
                    generator_input = generated
                    if training_config["noise_on_generator_updates"]:
                        generator_input = generated + sigma * torch.randn_like(generated)
                    generator_logits = discriminator(generator_input, text_embedding)
                    generator_targets = _target_tensor(
                        generator_logits,
                        float(training_config["generator_target_label"]),
                    )
                    generator_loss = criterion(generator_logits, generator_targets)
                    generator_loss = generator_loss + float(
                        training_config.get("conditioning_kl_weight", 0.0)
                    ) * conditioning_kl
                    if clip_alignment is not None:
                        alignment_loss = clip_alignment(generated, captions)
                        generator_loss = generator_loss + (
                            clip_alignment_weight * alignment_loss
                        )
                        epoch_alignment_loss += float(alignment_loss.detach())
                    _require_finite_tensor(generator_loss, "Generator loss")
                    generator_optimizer.zero_grad(set_to_none=True)
                    generator_loss.backward()
                    _require_finite_gradients(
                        generator_parameters, "Generator"
                    )
                    nn.utils.clip_grad_norm_(
                        generator_parameters,
                        float(training_config["gradient_clip_norm"]),
                    )
                    generator_optimizer.step()
                    _require_finite_parameters(
                        generator_parameters, "Generator and text-encoder"
                    )
                    epoch_generator_loss += float(generator_loss.detach())
                    generator_batches += 1

        if discriminator_batches == 0 or generator_batches == 0:
            raise RuntimeError("The training subset produced no complete batches.")
        mean_discriminator_loss = epoch_discriminator_loss / discriminator_batches
        mean_generator_loss = epoch_generator_loss / generator_batches
        mean_alignment_loss = (
            epoch_alignment_loss / generator_batches
            if clip_alignment is not None
            else 0.0
        )
        losses["discriminator"].append(mean_discriminator_loss)
        losses["generator"].append(mean_generator_loss)
        losses["clip_alignment"].append(mean_alignment_loss)
        generator_scheduler.step()
        discriminator_scheduler.step()
        completed_epoch = epoch_index + 1
        print(
            f"Epoch {completed_epoch}/{training_config['epochs']} "
            f"D={mean_discriminator_loss:.4f} G={mean_generator_loss:.4f} "
            f"CLIP={mean_alignment_loss:.4f}"
        )
        plot_losses(losses, config["paths"]["losses_plot"])

        if (
            fixed_captions
            and completed_epoch % int(training_config["sample_grid_interval_epochs"]) == 0
        ):
            _save_sample_grid(
                text_encoder,
                generator,
                fixed_captions,
                fixed_noise,
                samples_dir / f"epoch_{completed_epoch:04d}.png",
            )
        _save_checkpoint_if_due(
            _checkpoint_payload(
                epoch=completed_epoch,
                config=config,
                text_encoder=text_encoder,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator_scheduler=generator_scheduler,
                discriminator_scheduler=discriminator_scheduler,
                losses=losses,
            ),
            checkpoint_dir,
            config,
            completed_epoch=completed_epoch,
            total_epochs=int(training_config["epochs"]),
            interval=int(training_config["checkpoint_interval_epochs"]),
        )
