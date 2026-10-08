"""Fail-closed progress checks for resumable RL training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping


def checkpoint_step(value: object, *, name: str, limit: int) -> int:
    """Accept only real integer counters within the declared training budget."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
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
    if rollout_index != expected_rollouts:
        raise ValueError(f"{label} progress is inconsistent with rollout count")
    return steps, rollout_index


def checkpoint_boolean(value: object, *, name: str) -> bool:
    """Reject implicit coercion of saved recurrent episode-boundary flags."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value
