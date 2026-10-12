"""Shared numerical contracts for public tensor inputs."""

from __future__ import annotations

import torch
from torch import Tensor


def finite_tensor(value: Tensor, name: str, *, complex_allowed: bool = False) -> None:
    valid_type = value.is_floating_point() or (complex_allowed and value.is_complex())
    if not valid_type or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def finite_result(value: Tensor, name: str) -> Tensor:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite {name}")
    return value


def sample_count(value: int, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"sample_count must be an integer of at least {minimum}")
