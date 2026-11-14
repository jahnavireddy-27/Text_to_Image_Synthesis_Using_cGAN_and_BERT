"""Text-conditioned generator producing normalized 64x64 RGB images."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatialWordAttention(nn.Module):
    """Let each image location retrieve the caption tokens it needs."""

    def __init__(self, channels: int, text_dim: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(channels, text_dim, kernel_size=1, bias=False)
        self.key = nn.Linear(text_dim, text_dim, bias=False)
        self.value = nn.Linear(text_dim, text_dim, bias=False)
        self.output = nn.Conv2d(text_dim, channels, kernel_size=1, bias=False)
        self.scale = text_dim**-0.5
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        image_features: torch.Tensor,
        word_features: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = image_features.shape
        if word_features.ndim != 3 or word_features.shape[0] != batch:
            raise ValueError("word_features must have shape (batch, tokens, text_dim).")
        if word_mask.shape != word_features.shape[:2]:
            raise ValueError("word_mask must match the batch and token dimensions.")
        if word_features.shape[2] != self.key.in_features:
            raise ValueError("word_features dimension does not match text_dim.")
        if torch.any(word_mask.sum(dim=1) == 0):
            raise ValueError("Each caption must have at least one unmasked word.")

        queries = self.query(image_features).flatten(2).transpose(1, 2)
        keys = self.key(word_features)
        values = self.value(word_features)
        scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        scores = scores.masked_fill(~word_mask[:, None, :].bool(), -1e4)
        attention = F.softmax(scores, dim=-1)
        context = torch.bmm(attention, values)
        context = context.transpose(1, 2).reshape(batch, -1, height, width)
        return image_features + self.gamma * self.output(context)


class DeeperGenerator(nn.Module):
    """Project text/noise to 512x4x4, then apply four upsampling blocks."""

    def __init__(
        self,
        text_dim: int = 256,
        noise_dim: int = 100,
        initial_channels: int = 512,
        initial_size: int = 4,
        channel_schedule: tuple[int, ...] | list[int] = (512, 256, 128, 64, 3),
        batch_norm: bool = True,
        hidden_activation: str = "relu",
        output_activation: str = "tanh",
        output_batch_norm: bool = False,
        conditioning_augmentation: bool = False,
        word_level_attention: bool = False,
    ) -> None:
        super().__init__()
        if len(channel_schedule) != 5:
            raise ValueError("channel_schedule must contain an initial block and 4 stages.")
        if channel_schedule[0] != initial_channels:
            raise ValueError("channel_schedule[0] must equal initial_channels.")
        if hidden_activation != "relu":
            raise ValueError("Only ReLU hidden activation is currently supported.")
        if output_activation != "tanh":
            raise ValueError("Only Tanh output activation is currently supported.")
        if any(channel <= 0 for channel in channel_schedule):
            raise ValueError("All generator channel counts must be positive.")
        self.text_dim = text_dim
        self.noise_dim = noise_dim
        self.conditioning_augmentation_enabled = conditioning_augmentation
        self.word_level_attention_enabled = word_level_attention
        self.conditioning_augmentation = (
            nn.Linear(text_dim, text_dim * 2) if conditioning_augmentation else None
        )
        self.initial_channels = initial_channels
        self.initial_size = initial_size
        self.projection = nn.Linear(
            text_dim + noise_dim, initial_channels * initial_size * initial_size
        )

        blocks: list[nn.Module] = []
        for index, (in_channels, out_channels) in enumerate(
            zip(channel_schedule[:-1], channel_schedule[1:])
        ):
            is_output = index == len(channel_schedule) - 2
            layers: list[nn.Module] = [
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=4, stride=2, padding=1
                )
            ]
            if is_output:
                if output_batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.Tanh())
            else:
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            blocks.append(nn.Sequential(*layers))
        self.upsampling = nn.Sequential(*blocks)
        self.word_attention = (
            nn.ModuleList(
                SpatialWordAttention(channels, text_dim)
                for channels in (initial_channels, *channel_schedule[1:-1])
            )
            if word_level_attention
            else None
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        noise: torch.Tensor,
        *,
        return_kl: bool = False,
        word_features: torch.Tensor | None = None,
        word_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return images of shape (batch, 3, 64, 64) in [-1, 1]."""
        if text_embedding.ndim != 2 or text_embedding.shape[1] != self.text_dim:
            raise ValueError(f"text_embedding must have shape (batch, {self.text_dim}).")
        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError(f"noise must have shape (batch, {self.noise_dim}).")
        if text_embedding.shape[0] != noise.shape[0]:
            raise ValueError("Text embedding and noise batch sizes must match.")
        if self.word_attention is not None:
            if word_features is None or word_mask is None:
                raise ValueError(
                    "word_features and word_mask are required when word-level "
                    "attention is enabled."
                )
            if word_features.shape[2] != self.text_dim:
                raise ValueError(f"word_features must end in dimension {self.text_dim}.")
        elif word_features is not None or word_mask is not None:
            raise ValueError("This generator was built without word-level attention.")
        kl_loss = text_embedding.new_zeros(())
        if self.conditioning_augmentation is not None:
            mean, log_variance = self.conditioning_augmentation(text_embedding).chunk(2, dim=1)
            log_variance = log_variance.clamp(min=-10.0, max=10.0)
            if self.training:
                condition = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
            else:
                condition = mean
            kl_loss = -0.5 * torch.mean(
                1.0 + log_variance - mean.square() - log_variance.exp()
            )
        else:
            condition = text_embedding
        combined = torch.cat((condition, noise), dim=1)
        projected = self.projection(combined)
        feature_map = projected.view(
            -1, self.initial_channels, self.initial_size, self.initial_size
        )
        if self.word_attention is not None:
            assert word_features is not None and word_mask is not None
            feature_map = self.word_attention[0](feature_map, word_features, word_mask)
            for index, block in enumerate(self.upsampling):
                feature_map = block(feature_map)
                if index + 1 < len(self.word_attention):
                    feature_map = self.word_attention[index + 1](
                        feature_map, word_features, word_mask
                    )
            images = feature_map
        else:
            images = self.upsampling(feature_map)
        if return_kl:
            return images, kl_loss
        return images
