"""Strict validation and staging for scientific checkpoint payload data."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def checkpoint_finite_float(value: object, *, name: str) -> float:
    """Accept only real scalar values that are finite and not booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def checkpoint_float_list(value: object, *, name: str) -> list[float]:
    """Validate a checkpoint list of finite scalar values."""

    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return [
        checkpoint_finite_float(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def checkpoint_float_mapping(value: object, *, name: str) -> dict[str, float]:
    """Validate string-keyed finite scalar metrics without key/value coercion."""

    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary")
    result: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        result[key] = checkpoint_finite_float(
            item,
            name=f"{name}[{key!r}]",
        )
    return result


def checkpoint_observation(
    value: object,
    *,
    name: str,
    device: torch.device,
) -> Tensor:
    """Validate and stage a finite floating observation before live restore."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite values")
    return value.to(
        device=device,
        dtype=torch.float32,
    ).clone()



def checkpoint_generator_state(
    value: object,
    *,
    name: str,
    device: torch.device,
) -> Tensor:
    """Validate a torch.Generator state without mutating the live generator."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    candidate = value.detach().cpu().contiguous()
    if candidate.dtype != torch.uint8 or candidate.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional uint8 tensor")
    validator = torch.Generator(device=device)
    try:
        validator.set_state(candidate)
    except RuntimeError as exc:
        raise ValueError(f"{name} is invalid") from exc
    return candidate
