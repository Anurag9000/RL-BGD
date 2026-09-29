"""Standard continual-learning metrics with explicit assumptions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _vector(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("expected a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("metric inputs must be finite")
    return array


def _performance_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("performance matrix must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("performance matrix must be finite")
    return matrix


def final_average_performance(final_scores: Sequence[float]) -> float:
    """Return arithmetic mean of final per-task performance."""

    return float(_vector(final_scores).mean())


def forgetting(
    performance_matrix: Sequence[Sequence[float]],
) -> tuple[np.ndarray, float]:
    """Compute diagonal-after-learning minus final performance.

    Rows represent evaluation checkpoints after sequential training stages and
    columns represent tasks. Task i is assumed to have just been learned at row
    i. The last task has no later stage in which to forget, so it is excluded.
    """

    matrix = _performance_matrix(performance_matrix)
    tasks = min(matrix.shape[0], matrix.shape[1])
    if tasks < 2:
        return np.empty(0, dtype=np.float64), 0.0
    values = np.asarray(
        [matrix[index, index] - matrix[-1, index] for index in range(tasks - 1)],
        dtype=np.float64,
    )
    return values, float(values.mean())


def backward_transfer(
    performance_matrix: Sequence[Sequence[float]],
) -> float:
    """Return final minus post-learning performance on previous tasks."""

    values, _ = forgetting(performance_matrix)
    if values.size == 0:
        return 0.0
    return float((-values).mean())


def forward_transfer(
    pre_learning_scores: Sequence[float],
    reference_scores: Sequence[float],
) -> float:
    """Return mean pre-learning improvement over a matched reference baseline.

    This generic definition is not the Continual World AUC-based definition;
    benchmark-specific code must use the canonical CW metric when applicable.
    """

    observed = _vector(pre_learning_scores)
    reference = _vector(reference_scores)
    if observed.shape != reference.shape:
        raise ValueError("forward-transfer inputs must have matching shapes")
    return float((observed - reference).mean())


def lifetime_auc(
    steps: Sequence[float],
    values: Sequence[float],
    *,
    normalize_by_duration: bool = True,
) -> float:
    """Trapezoidal area under a lifetime performance curve."""

    x = _vector(steps)
    y = _vector(values)
    if x.shape != y.shape:
        raise ValueError("steps and values must have matching shapes")
    if x.size < 2:
        return float(y[0])
    if np.any(np.diff(x) <= 0):
        raise ValueError("steps must be strictly increasing")
    area = float(np.trapezoid(y, x))
    if normalize_by_duration:
        area /= float(x[-1] - x[0])
    return area


def plasticity_retention(
    continual_auc: float,
    single_task_auc: float,
) -> float:
    """Compute PR = continual learning AUC / matched single-task AUC."""

    if not np.isfinite(continual_auc) or not np.isfinite(single_task_auc):
        raise ValueError("AUC inputs must be finite")
    if single_task_auc == 0:
        raise ZeroDivisionError("single_task_auc must be non-zero")
    return float(continual_auc / single_task_auc)
