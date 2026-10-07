"""TD/Bellman-residual surprise for adaptive posterior replasticization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import torch
from torch import Tensor

from rl_bgd.surprise.base import (
    EMANormalizerConfig,
    EMASurpriseNormalizer,
    RetentionMappingConfig,
    SurpriseObservation,
)


@dataclass(frozen=True)
class TDSurpriseConfig:
    aggregation: Literal["median_abs", "mean_abs"] = "median_abs"
    normalizer: EMANormalizerConfig = field(default_factory=EMANormalizerConfig)

    def validate(self) -> None:
        if self.aggregation not in {"median_abs", "mean_abs"}:
            raise ValueError(f"unsupported TD surprise aggregation: {self.aggregation}")
        self.normalizer.validate()


@dataclass(frozen=True)
class AdaptiveTDRetentionConfig:
    """Configuration for TD-surprise-driven posterior tempering."""

    surprise: TDSurpriseConfig = field(default_factory=TDSurpriseConfig)
    mapping: RetentionMappingConfig = field(default_factory=RetentionMappingConfig)

    def validate(self) -> None:
        self.surprise.validate()
        self.mapping.validate()


class TDSurprise:
    """Online normalized surprise from TD/Bellman residual magnitudes."""

    def __init__(self, config: TDSurpriseConfig | None = None) -> None:
        self.config = config or TDSurpriseConfig()
        self.config.validate()
        self.normalizer = EMASurpriseNormalizer(self.config.normalizer)

    def observe(self, td_errors: Tensor) -> SurpriseObservation:
        values = td_errors.detach().float().abs().reshape(-1)
        if values.numel() == 0:
            raise ValueError("TD surprise requires at least one residual")
        if not torch.isfinite(values).all():
            raise FloatingPointError("TD errors contain nonfinite values")
        if self.config.aggregation == "median_abs":
            raw = float(values.median().item())
        else:
            raw = float(values.mean().item())
        return self.normalizer.observe(raw)

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "config": asdict(self.config),
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("version") != 2:
            raise ValueError("unsupported TD-surprise checkpoint version")
        if state.get("config") != asdict(self.config):
            raise ValueError("TD-surprise checkpoint configuration mismatch")
        normalizer_state = state["normalizer"]
        if not isinstance(normalizer_state, dict):
            raise TypeError("TD-surprise normalizer state must be a dictionary")
        self.normalizer.load_state_dict(normalizer_state)
