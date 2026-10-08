"""Helpers for fail-closed checkpoint restoration."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import TypeVar

StateT = TypeVar("StateT")


def transactional_state_load(
    state: StateT,
    *,
    current_state: Callable[[], StateT],
    apply: Callable[[StateT], None],
) -> None:
    """Apply a checkpoint or restore the exact prior state if application fails."""

    previous = deepcopy(current_state())
    try:
        apply(state)
    except Exception:
        try:
            apply(previous)
        except Exception as rollback_error:
            raise RuntimeError(
                "checkpoint load failed and rollback could not restore the prior state"
            ) from rollback_error
        raise
