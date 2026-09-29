"""Evaluation-only metrics for surprise-based environment-change detection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class ChangeDetectionMetrics:
    """Event-level change-detection summary."""

    matched_events: int
    true_events: int
    detected_events: int
    mean_detection_delay: float | None
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_alarms_per_million_steps: float


def change_detection_metrics(
    true_change_steps: Sequence[int],
    detected_steps: Sequence[int],
    *,
    tolerance_steps: int,
    total_steps: int,
) -> ChangeDetectionMetrics:
    """Match each true change to the earliest unused subsequent detection."""

    if tolerance_steps < 0 or total_steps < 1:
        raise ValueError("invalid change-detection horizon")
    true_steps = sorted(set(int(step) for step in true_change_steps))
    detections = sorted(set(int(step) for step in detected_steps))
    if any(step < 0 or step >= total_steps for step in true_steps + detections):
        raise ValueError("event step lies outside evaluation horizon")

    unused = set(detections)
    delays: list[int] = []
    for change in true_steps:
        candidates = [
            detection
            for detection in unused
            if change <= detection <= change + tolerance_steps
        ]
        if candidates:
            matched = min(candidates)
            unused.remove(matched)
            delays.append(matched - change)

    matched_count = len(delays)
    false_positives = len(detections) - matched_count
    precision = (
        matched_count / len(detections)
        if detections
        else 0.0
    )
    recall = (
        matched_count / len(true_steps)
        if true_steps
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    negative_steps = max(total_steps - len(true_steps), 1)
    return ChangeDetectionMetrics(
        matched_events=matched_count,
        true_events=len(true_steps),
        detected_events=len(detections),
        mean_detection_delay=(
            float(np.mean(delays))
            if delays
            else None
        ),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        false_positive_rate=float(false_positives / negative_steps),
        false_alarms_per_million_steps=float(
            false_positives * 1_000_000.0 / total_steps
        ),
    )


def change_event_labels(
    total_steps: int,
    change_steps: Sequence[int],
    *,
    positive_window: int,
) -> np.ndarray:
    """Mark post-change windows as positives for score-based evaluation."""

    if total_steps < 1 or positive_window < 1:
        raise ValueError("total_steps and positive_window must be positive")
    labels = np.zeros(total_steps, dtype=np.int8)
    for change in change_steps:
        if change < 0 or change >= total_steps:
            raise ValueError("change step lies outside evaluation horizon")
        labels[change : min(total_steps, change + positive_window)] = 1
    return labels


def binary_auroc(
    scores: Sequence[float],
    labels: Sequence[int | bool],
) -> float:
    """Compute AUROC using the Mann-Whitney rank statistic with tie handling."""

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int8)
    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if score_array.shape != label_array.shape or score_array.size == 0:
        raise ValueError("scores and labels must be matching non-empty vectors")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    if not np.isin(label_array, [0, 1]).all():
        raise ValueError("labels must be binary")
    positives = int(label_array.sum())
    negatives = int(label_array.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    ranks = rankdata(score_array, method="average")
    positive_rank_sum = float(ranks[label_array == 1].sum())
    u_statistic = (
        positive_rank_sum
        - positives * (positives + 1) / 2.0
    )
    return float(u_statistic / (positives * negatives))
