"""Adaptation-speed, post-change, and recurrence metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rl_bgd.metrics.continual import lifetime_auc


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
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("steps and values must be matching non-empty vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("adaptation trace must be finite")
    threshold = baseline + fraction * (reference - baseline)
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
    """Compute curve area within a fixed post-change window."""

    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    mask = (x >= switch_step) & (x <= switch_step + window_steps)
    if mask.sum() == 0:
        raise ValueError("no observations fall inside the requested window")
    selected_x = x[mask]
    selected_y = y[mask]
    if selected_x.size == 1:
        return float(selected_y[0])
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

    if asymptotic_window < 1:
        raise ValueError("asymptotic_window must be positive")
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
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
