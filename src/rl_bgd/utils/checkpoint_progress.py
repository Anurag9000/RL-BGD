"""Fail-closed progress checks for resumable RL training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch


def checkpoint_integer(value: object, *, name: str) -> int:
    """Disallow lossy conversion of scientific checkpoint integer metadata."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def checkpoint_finite_float(value: object, *, name: str) -> float:
    """Accept only finite numeric checkpoint metadata without string/bool coercion."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def checkpoint_nonnegative_integer(value: object, *, name: str) -> int:
    """Accept only true integer counters that are non-negative."""

    value = checkpoint_integer(value, name=name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def checkpoint_tensor_like(
    value: object,
    *,
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Validate shape/dtype/finiteness and stage a saved tensor."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != reference.shape:
        raise ValueError(f"{name} shape mismatch")
    if value.dtype != reference.dtype:
        raise ValueError(f"{name} dtype mismatch")
    if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
        value
    ).all().item():
        raise ValueError(f"{name} contains non-finite values")
    return value.detach().to(device=reference.device).clone()


def checkpoint_step(value: object, *, name: str, limit: int) -> int:
    """Accept only real integer counters within the declared training budget."""

    value = checkpoint_integer(value, name=name)
    if not 0 <= value <= limit:
        raise ValueError(f"{name} is invalid")
    return value


def checkpoint_ppo_progress(
    payload: Mapping[str, object],
    *,
    total_steps: int,
    rollout_steps: int,
    label: str,
) -> tuple[int, int]:
    """Validate a checkpoint saved only after complete PPO rollout updates."""

    steps = checkpoint_step(
        payload.get("steps"),
        name=f"{label} steps",
        limit=total_steps,
    )
    expected_rollouts = (steps + rollout_steps - 1) // rollout_steps
    rollout_index = checkpoint_step(
        payload.get("rollout_index"),
        name=f"{label} rollout_index",
        limit=expected_rollouts,
    )
    if rollout_index != expected_rollouts or steps != min(
        rollout_index * rollout_steps, total_steps
    ):
        raise ValueError(f"{label} progress is inconsistent with rollout count")
    return steps, rollout_index


def checkpoint_boolean(value: object, *, name: str) -> bool:
    """Reject implicit coercion of saved recurrent episode-boundary flags."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value
