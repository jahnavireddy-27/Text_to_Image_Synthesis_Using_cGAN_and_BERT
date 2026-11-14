"""Image-caption discriminator that returns real/fake logits."""

from __future__ import annotations

import torch
from torch import nn


class DeeperDiscriminator(nn.Module):
    """Reduce image-caption pairs to one raw real/fake logit per sample."""

    def __init__(
        self,
        image_channels: int = 3,
        text_dim: int = 256,
        image_channel_schedule: tuple[int, ...] | list[int] = (3, 64, 128, 256, 512),
        image_size: int = 64,
        feature_size: int = 4,
        image_kernel_size: int = 4,
        image_stride: int = 2,
        image_padding: int = 1,
        batch_norm_image: bool = True,
        batch_norm_first_image_block: bool = False,
        joint_channels: tuple[int, ...] | list[int] = (512, 1),
        joint_kernel_sizes: tuple[int, ...] | list[int] = (3, 4),
        joint_strides: tuple[int, ...] | list[int] = (1, 1),
        joint_paddings: tuple[int, ...] | list[int] = (1, 0),
        batch_norm_joint: bool = True,
        batch_norm_final_output: bool = False,
        activation: str = "leaky_relu",
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if len(image_channel_schedule) != 5:
            raise ValueError("image_channel_schedule must define four image blocks.")
        if image_channel_schedule[0] != image_channels:
            raise ValueError("image_channel_schedule[0] must equal image_channels.")
        if activation != "leaky_relu":
            raise ValueError("Only LeakyReLU discriminator activation is supported.")
        if not (
            len(joint_channels)
            == len(joint_kernel_sizes)
            == len(joint_strides)
            == len(joint_paddings)
        ):
            raise ValueError("All discriminator joint-head schedules must have equal lengths.")
        if not joint_channels or joint_channels[-1] != 1:
            raise ValueError("The final joint-head layer must produce one logit channel.")
        self.text_dim = text_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.feature_size = feature_size

        image_blocks: list[nn.Module] = []
        for index, (in_channels, out_channels) in enumerate(
            zip(image_channel_schedule[:-1], image_channel_schedule[1:])
        ):
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=image_kernel_size,
                    stride=image_stride,
                    padding=image_padding,
                    bias=False,
                )
            ]
            use_batch_norm = (
                batch_norm_first_image_block
                if index == 0
                else batch_norm_image
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            image_blocks.append(nn.Sequential(*layers))
        self.image_blocks = nn.Sequential(*image_blocks)

        joint_blocks: list[nn.Module] = []
        in_channels = image_channel_schedule[-1] + text_dim
        for index, (out_channels, kernel, stride, padding) in enumerate(
            zip(joint_channels, joint_kernel_sizes, joint_strides, joint_paddings)
        ):
            is_output = index == len(joint_channels) - 1
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            ]
            if is_output:
                if batch_norm_final_output:
                    layers.append(nn.BatchNorm2d(out_channels))
            else:
                if batch_norm_joint:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            joint_blocks.append(nn.Sequential(*layers))
            in_channels = out_channels
        self.joint_blocks = nn.Sequential(*joint_blocks)

    def forward(
        self, images: torch.Tensor, text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Return logits of shape (batch,), without applying sigmoid."""
        expected_shape = (self.image_channels, self.image_size, self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                "images must have shape "
                f"(batch, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]})."
            )
        if text_embedding.ndim != 2 or text_embedding.shape[1] != self.text_dim:
            raise ValueError(f"text_embedding must have shape (batch, {self.text_dim}).")
        if images.shape[0] != text_embedding.shape[0]:
            raise ValueError("Image and text embedding batch sizes must match.")
        image_features = self.image_blocks(images)
        if tuple(image_features.shape[-2:]) != (self.feature_size, self.feature_size):
            raise ValueError(
                "The image branch output does not match discriminator.feature_size."
            )
        expanded_text = text_embedding[:, :, None, None].expand(
            -1, -1, self.feature_size, self.feature_size
        )
        joint_features = torch.cat((image_features, expanded_text), dim=1)
        logits = self.joint_blocks(joint_features)
        if tuple(logits.shape[1:]) != (1, 1, 1):
            raise ValueError("The discriminator joint head must return a 1x1 logit.")
        return logits.flatten(start_dim=1).squeeze(1)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities when a consumer explicitly needs them."""
        return torch.sigmoid(logits)
