"""Bootstrap statistics for provenance-preserving paper aggregation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapEstimate:
    """Mean, sample spread, and percentile bootstrap confidence interval."""

    mean: float
    std: float
    ci_low: float
    ci_high: float
    n: int
    confidence: float
    resamples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _values(
    values: Sequence[float],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        tuple(float(value) for value in values),
        dtype=np.float64,
    )
    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must contain at least one scalar"
        )
    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains nonfinite values"
        )
    return array


def _validate_bootstrap(
    *,
    confidence: float,
    resamples: int,
) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "confidence must lie strictly between 0 and 1"
        )
    if resamples < 1:
        raise ValueError(
            "bootstrap resamples must be positive"
        )


def _percentile_interval(
    samples: np.ndarray,
    *,
    confidence: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        samples,
        (
            alpha / 2.0,
            1.0 - alpha / 2.0,
        ),
    )
    return float(lower), float(upper)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapEstimate:
    """Bootstrap the mean across independent seed-level observations."""

    _validate_bootstrap(
        confidence=confidence,
        resamples=resamples,
    )
    array = _values(
        values,
        name="bootstrap values",
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        array.size,
        size=(
            resamples,
            array.size,
        ),
    )
    sampled_means = array[
        indices
    ].mean(axis=1)
    ci_low, ci_high = (
        _percentile_interval(
            sampled_means,
            confidence=confidence,
        )
    )
    std = (
        float(
            array.std(
                ddof=1
            )
        )
        if array.size > 1
        else 0.0
    )
    return BootstrapEstimate(
        mean=float(array.mean()),
        std=std,
        ci_low=ci_low,
        ci_high=ci_high,
        n=int(array.size),
        confidence=confidence,
        resamples=resamples,
    )


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapEstimate:
    """Bootstrap matched-seed left-minus-right differences."""

    left_values = _values(
        left,
        name="left paired values",
    )
    right_values = _values(
        right,
        name="right paired values",
    )
    if left_values.shape != right_values.shape:
        raise ValueError(
            "paired bootstrap inputs must have identical lengths"
        )
    differences = (
        left_values - right_values
    )
    return bootstrap_mean_ci(
        differences.tolist(),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def hierarchical_bootstrap_mean(
    seed_task_values: Mapping[
        int,
        Sequence[float],
    ],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapEstimate:
    """Bootstrap seed then task within seed without treating tasks as independent.

    The point estimate gives each seed equal weight by first averaging the task
    measurements within that seed. Each bootstrap replicate resamples seeds
    with replacement and then resamples task measurements within every sampled
    seed before averaging the seed means.
    """

    _validate_bootstrap(
        confidence=confidence,
        resamples=resamples,
    )
    if not seed_task_values:
        raise ValueError(
            "hierarchical bootstrap requires at least one seed"
        )

    normalized: dict[
        int,
        np.ndarray,
    ] = {}
    for seed_id, values in seed_task_values.items():
        normalized[
            int(seed_id)
        ] = _values(
            values,
            name=(
                "hierarchical values "
                f"for seed {seed_id}"
            ),
        )

    seed_ids = np.asarray(
        sorted(normalized),
        dtype=np.int64,
    )
    seed_means = np.asarray(
        [
            normalized[
                int(seed_id)
            ].mean()
            for seed_id in seed_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(
        resamples,
        dtype=np.float64,
    )

    for replicate in range(
        resamples
    ):
        sampled_seed_positions = (
            rng.integers(
                0,
                seed_ids.size,
                size=seed_ids.size,
            )
        )
        resampled_seed_means: list[
            float
        ] = []
        for position in sampled_seed_positions:
            seed_id = int(
                seed_ids[position]
            )
            task_values = normalized[
                seed_id
            ]
            task_positions = (
                rng.integers(
                    0,
                    task_values.size,
                    size=task_values.size,
                )
            )
            resampled_seed_means.append(
                float(
                    task_values[
                        task_positions
                    ].mean()
                )
            )
        bootstrap_means[
            replicate
        ] = float(
            np.mean(
                resampled_seed_means
            )
        )

    ci_low, ci_high = (
        _percentile_interval(
            bootstrap_means,
            confidence=confidence,
        )
    )
    std = (
        float(
            seed_means.std(
                ddof=1
            )
        )
        if seed_means.size > 1
        else 0.0
    )
    mean = float(
        seed_means.mean()
    )
    if not all(
        math.isfinite(value)
        for value in (
            mean,
            std,
            ci_low,
            ci_high,
        )
    ):
        raise FloatingPointError(
            "hierarchical bootstrap produced nonfinite statistics"
        )
    return BootstrapEstimate(
        mean=mean,
        std=std,
        ci_low=ci_low,
        ci_high=ci_high,
        n=int(
            seed_means.size
        ),
        confidence=confidence,
        resamples=resamples,
    )
