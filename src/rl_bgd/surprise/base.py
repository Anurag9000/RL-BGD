"""Shared online surprise normalization and retention mapping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EMANormalizerConfig:
    """Exponential running center/scale and surprise smoothing."""

    decay: float = 0.99
    smoothing_decay: float = 0.9
    initial_variance: float = 1.0
    epsilon: float = 1e-6

    def validate(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must lie in [0, 1)")
        if not 0.0 <= self.smoothing_decay < 1.0:
            raise ValueError("smoothing_decay must lie in [0, 1)")
        if self.initial_variance <= 0:
            raise ValueError("initial_variance must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class SurpriseObservation:
    """One normalized online surprise observation."""

    raw: float
    center: float
    scale: float
    normalized: float
    smoothed: float
    count: int


class EMASurpriseNormalizer:
    """Stateful online normalizer that never needs task-boundary callbacks."""

    def __init__(self, config: EMANormalizerConfig | None = None) -> None:
        self.config = config or EMANormalizerConfig()
        self.config.validate()
        self.count = 0
        self.center = 0.0
        self.variance = self.config.initial_variance
        self.smoothed = 0.0

    def observe(self, value: float) -> SurpriseObservation:
        if not math.isfinite(value):
            raise FloatingPointError("surprise statistic must be finite")
        if self.count == 0:
            self.center = value
            self.variance = self.config.initial_variance
            normalized = 0.0
            self.smoothed = 0.0
        else:
            scale_before = math.sqrt(max(self.variance, self.config.epsilon))
            normalized = abs(value - self.center) / (scale_before + self.config.epsilon)
            delta = value - self.center
            self.center = (
                self.config.decay * self.center
                + (1.0 - self.config.decay) * value
            )
            self.variance = (
                self.config.decay * self.variance
                + (1.0 - self.config.decay) * delta * delta
            )
            self.smoothed = (
                self.config.smoothing_decay * self.smoothed
                + (1.0 - self.config.smoothing_decay) * normalized
            )
        self.count += 1
        scale = math.sqrt(max(self.variance, self.config.epsilon))
        return SurpriseObservation(
            raw=value,
            center=self.center,
            scale=scale,
            normalized=normalized,
            smoothed=self.smoothed,
            count=self.count,
        )

    def state_dict(self) -> dict[str, float | int]:
        return {
            "version": 1,
            "count": self.count,
            "center": self.center,
            "variance": self.variance,
            "smoothed": self.smoothed,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported surprise-normalizer checkpoint version")
        count = int(state["count"])
        center = float(state["center"])
        variance = float(state["variance"])
        smoothed = float(state["smoothed"])
        if count < 0 or variance <= 0:
            raise ValueError("invalid surprise-normalizer checkpoint state")
        if not all(math.isfinite(x) for x in (center, variance, smoothed)):
            raise FloatingPointError("nonfinite surprise-normalizer checkpoint state")
        self.count = count
        self.center = center
        self.variance = variance
        self.smoothed = smoothed


@dataclass(frozen=True)
class RetentionMappingConfig:
    """Monotone map from smoothed surprise to posterior retention."""

    lambda_min: float = 0.5
    kappa: float = 1.0

    def validate(self) -> None:
        if not 0.0 < self.lambda_min <= 1.0:
            raise ValueError("lambda_min must lie in (0, 1]")
        if self.kappa < 0:
            raise ValueError("kappa must be non-negative")


def surprise_to_retention(
    smoothed_surprise: float,
    config: RetentionMappingConfig,
) -> float:
    """Map surprise S to lambda_min + (1-lambda_min) exp(-kappa S)."""

    config.validate()
    if not math.isfinite(smoothed_surprise) or smoothed_surprise < 0:
        raise ValueError("smoothed_surprise must be finite and non-negative")
    return config.lambda_min + (1.0 - config.lambda_min) * math.exp(
        -config.kappa * smoothed_surprise
    )
