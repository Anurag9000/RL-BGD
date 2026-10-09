"""Strict runtime configuration validation for scientific experiment controls."""

from __future__ import annotations

import math


def config_integer(value: object, *, name: str) -> int:
    """Require an actual integer rather than a coercible numeric value."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def config_positive_integer(value: object, *, name: str) -> int:
    """Require a strictly positive integer configuration value."""

    result = config_integer(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def config_nonnegative_integer(value: object, *, name: str) -> int:
    """Require a non-negative integer configuration value."""

    result = config_integer(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def config_boolean(value: object, *, name: str) -> bool:
    """Require a real boolean rather than relying on truth-value coercion."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def config_finite_float(value: object, *, name: str) -> float:
    """Require a finite numeric configuration scalar, without implicit coercion."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
