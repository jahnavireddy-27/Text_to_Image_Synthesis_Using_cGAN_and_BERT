"""Fail-fast checks that prevent non-finite GAN updates."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from text_to_image_gan.training import (
    _require_finite_gradients,
    _require_finite_parameters,
    _require_finite_tensor,
)


def test_finite_tensor_guard_accepts_finite_values() -> None:
    _require_finite_tensor(torch.tensor([0.0, 1.0]), "test loss")


def test_finite_tensor_guard_rejects_nan_and_infinity() -> None:
    with pytest.raises(FloatingPointError, match="test loss"):
        _require_finite_tensor(torch.tensor([float("nan")]), "test loss")
    with pytest.raises(FloatingPointError, match="test logits"):
        _require_finite_tensor(torch.tensor([float("inf")]), "test logits")


def test_gradient_and_parameter_guards_reject_non_finite_values() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError, match="generator gradient"):
        _require_finite_gradients([parameter], "generator")

    parameter.grad = None
    with pytest.raises(FloatingPointError, match="generator parameter"):
        _require_finite_parameters([nn.Parameter(torch.tensor([float("inf")]))], "generator")


def test_finite_gradient_and_parameter_guards_accept_finite_values() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([0.25])
    _require_finite_gradients([parameter], "generator")
    _require_finite_parameters([parameter], "generator")
