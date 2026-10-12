"""Shared validation of scientific inputs; booleans are not numeric controls."""

from __future__ import annotations

import math
from numbers import Real

import torch


def require_integer(name: str, value: object, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def require_finite(name: str, value: object, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def require_float_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def validate_samples(samples: torch.Tensor, *, n_punctures: int | None = None) -> None:
    require_float_tensor("samples", samples)
    if samples.ndim != 2 or min(samples.shape) < 1:
        raise ValueError("samples must have nonempty shape (draws, n_punctures)")
    if n_punctures is not None and samples.shape[1] != n_punctures:
        raise ValueError(f"samples must have {n_punctures} puncture coordinates")
