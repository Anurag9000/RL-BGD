"""Adaptation-speed, post-change, and recurrence metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rl_bgd.metrics.continual import lifetime_auc


def _validated_trace(
    steps: Sequence[float],
    values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Require a finite, strictly ordered trace for reproducible time metrics."""

    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("steps and values must be matching non-empty vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("adaptation trace must be finite")
    if np.any(np.diff(x) <= 0):
        raise ValueError("adaptation steps must be strictly increasing")
    return x, y


def time_to_fraction(
    steps: Sequence[float],
    values: Sequence[float],
    *,
    switch_step: float,
    reference: float,
    fraction: float = 0.8,
    baseline: float = 0.0,
    higher_is_better: bool = True,
) -> float | None:
    """Return steps after a change until a fraction of recovery is reached.

    The recovery threshold is baseline + fraction * (reference - baseline).
    Returns None when the threshold is never reached in the supplied trace.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    for name, value in (
        ("switch_step", switch_step),
        ("reference", reference),
        ("baseline", baseline),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    x, y = _validated_trace(steps, values)
    threshold = baseline + fraction * (reference - baseline)
    if not math.isfinite(threshold):
        raise ValueError("recovery threshold must be finite")
    mask = x >= switch_step
    indices = np.flatnonzero(mask)
    for index in indices:
        reached = y[index] >= threshold if higher_is_better else y[index] <= threshold
        if reached:
            return float(x[index] - switch_step)
    return None


def post_change_auc(
    steps: Sequence[float],
    values: Sequence[float],
    *,
    switch_step: float,
    window_steps: float,
    normalize_by_duration: bool = True,
) -> float:
    """Integrate the entire observed post-change window with linear interpolation.

    Reject incomplete coverage instead of silently substituting a shorter span.
    The first and last window values are interpolated between adjacent samples.
    """

    if not math.isfinite(switch_step):
        raise ValueError("switch_step must be finite")
    if not math.isfinite(window_steps) or window_steps <= 0:
        raise ValueError("window_steps must be positive and finite")
    x, y = _validated_trace(steps, values)
    window_end = switch_step + window_steps
    if (
        not math.isfinite(window_end)
        or switch_step < x[0]
        or window_end > x[-1]
    ):
        raise ValueError("post-change window must be covered by the observed trace")

    interior = x[(x > switch_step) & (x < window_end)]
    selected_x = np.concatenate(([switch_step], interior, [window_end]))
    selected_y = np.interp(selected_x, x, y)
    return lifetime_auc(
        selected_x.tolist(),
        selected_y.tolist(),
        normalize_by_duration=normalize_by_duration,
    )


@dataclass(frozen=True)
class RecurrenceMetrics:
    """Metrics for revisiting an environment after intervening contexts."""

    zero_shot_return: float
    reacquisition_steps: float | None
    recovered_performance: float


def recurrence_metrics(
    steps: Sequence[float],
    values: Sequence[float],
    *,
    revisit_step: float,
    reference: float,
    fraction: float = 0.8,
    asymptotic_window: int = 5,
) -> RecurrenceMetrics:
    """Summarize zero-shot return, reacquisition, and late revisit performance."""

    if isinstance(asymptotic_window, bool) or not isinstance(asymptotic_window, int):
        raise TypeError("asymptotic_window must be an integer")
    if asymptotic_window < 1:
        raise ValueError("asymptotic_window must be positive")
    if not math.isfinite(revisit_step) or not math.isfinite(reference):
        raise ValueError("revisit_step and reference must be finite")
    x, y = _validated_trace(steps, values)
    indices = np.flatnonzero(x >= revisit_step)
    if indices.size == 0:
        raise ValueError("trace contains no revisit observation")
    start = int(indices[0])
    reacquisition = time_to_fraction(
        x.tolist(),
        y.tolist(),
        switch_step=revisit_step,
        reference=reference,
        fraction=fraction,
    )
    tail = y[start:]
    window = tail[-min(asymptotic_window, tail.size) :]
    return RecurrenceMetrics(
        zero_shot_return=float(y[start]),
        reacquisition_steps=reacquisition,
        recovered_performance=float(window.mean()),
    )
