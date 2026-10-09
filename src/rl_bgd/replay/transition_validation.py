"""Validation and staging for transitions entering replay buffers."""

from __future__ import annotations

from numbers import Real

import torch
from torch import Tensor


def replay_vector(
    value: object,
    *,
    name: str,
    reference: Tensor,
) -> Tensor:
    """Stage one finite floating vector with the exact replay field shape."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != tuple(reference.shape):
        raise ValueError(f"{name} shape mismatch")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite values")
    return (
        value.detach()
        .to(
            device=reference.device,
            dtype=reference.dtype,
        )
        .clone()
    )


def replay_reward(
    value: object,
    *,
    reference: Tensor,
) -> Tensor:
    """Stage one finite real reward scalar in the replay storage dtype."""

    if isinstance(value, bool):
        raise TypeError("replay reward must be a real scalar")
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("replay reward must contain exactly one value")
        if value.dtype == torch.bool or value.is_complex():
            raise TypeError("replay reward must be a real scalar")
        candidate = (
            value.detach()
            .to(
                device=reference.device,
                dtype=reference.dtype,
            )
            .reshape(())
        )
    elif isinstance(value, Real):
        candidate = torch.tensor(
            float(value),
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        raise TypeError("replay reward must be a real scalar")
    if not torch.isfinite(candidate).item():
        raise ValueError("replay reward must be finite")
    return candidate


def replay_boolean(value: object, *, name: str) -> bool:
    """Reject implicit truth-value coercion for replay boundary metadata."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def replay_insertion_step(value: object) -> int:
    """Require non-negative integer replay provenance."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("replay insertion_step must be an integer")
    if value < 0:
        raise ValueError("replay insertion_step must be non-negative")
    return value
