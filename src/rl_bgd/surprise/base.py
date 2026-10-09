"""Shared online surprise normalization and retention mapping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from rl_bgd.utils.config_validation import config_finite_float


def _checkpoint_int(
    value: object,
    *,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _checkpoint_float(
    value: object,
    *,
    name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"{name} must be numeric")
    return float(value)


@dataclass(frozen=True)
class EMANormalizerConfig:
    """Exponential running center/scale and surprise smoothing."""

    decay: float = 0.99
    smoothing_decay: float = 0.9
    initial_variance: float = 1.0
    epsilon: float = 1e-6

    def validate(self) -> None:
        decay = config_finite_float(self.decay, name="decay")
        smoothing_decay = config_finite_float(self.smoothing_decay, name="smoothing_decay")
        initial_variance = config_finite_float(self.initial_variance, name="initial_variance")
        epsilon = config_finite_float(self.epsilon, name="epsilon")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must lie in [0, 1)")
        if not 0.0 <= smoothing_decay < 1.0:
            raise ValueError("smoothing_decay must lie in [0, 1)")
        if initial_variance <= 0:
            raise ValueError("initial_variance must be positive")
        if epsilon <= 0:
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
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("surprise statistic must be numeric")
        if not math.isfinite(value):
            raise FloatingPointError("surprise statistic must be finite")
        if self.count == 0:
            next_center = float(value)
            next_variance = self.config.initial_variance
            normalized = 0.0
            next_smoothed = 0.0
        else:
            scale_before = math.sqrt(max(self.variance, self.config.epsilon))
            delta = value - self.center
            normalized = abs(delta) / (scale_before + self.config.epsilon)
            next_center = self.config.decay * self.center + (1.0 - self.config.decay) * value
            next_variance = (
                self.config.decay * self.variance + (1.0 - self.config.decay) * delta * delta
            )
            next_smoothed = (
                self.config.smoothing_decay * self.smoothed
                + (1.0 - self.config.smoothing_decay) * normalized
            )
        if not all(
            math.isfinite(number)
            for number in (next_center, next_variance, normalized, next_smoothed)
        ):
            raise FloatingPointError("surprise normalization produced non-finite statistics")
        self.center = next_center
        self.variance = next_variance
        self.smoothed = next_smoothed
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

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "config": asdict(self.config),
            "count": self.count,
            "center": self.center,
            "variance": self.variance,
            "smoothed": self.smoothed,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        version = _checkpoint_int(
            state.get("version"),
            name="surprise-normalizer checkpoint version",
        )
        if version != 2:
            raise ValueError("unsupported surprise-normalizer checkpoint version")
        if state.get("config") != asdict(self.config):
            raise ValueError("surprise-normalizer checkpoint configuration mismatch")
        count = _checkpoint_int(
            state["count"],
            name="surprise count",
        )
        center = _checkpoint_float(
            state["center"],
            name="surprise center",
        )
        variance = _checkpoint_float(
            state["variance"],
            name="surprise variance",
        )
        smoothed = _checkpoint_float(
            state["smoothed"],
            name="surprise smoothed",
        )
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
        lambda_min = config_finite_float(self.lambda_min, name="lambda_min")
        kappa = config_finite_float(self.kappa, name="kappa")
        if not 0.0 < lambda_min <= 1.0:
            raise ValueError("lambda_min must lie in (0, 1]")
        if kappa < 0:
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
