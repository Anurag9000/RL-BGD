"""Ensemble-disagreement surprise primitive."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from rl_bgd.surprise.base import (
    EMANormalizerConfig,
    EMASurpriseNormalizer,
    RetentionMappingConfig,
    SurpriseObservation,
)
from rl_bgd.utils.checkpoint_progress import checkpoint_integer


@dataclass(frozen=True)
class AdaptiveEnsembleRetentionConfig:
    """Configuration for critic-disagreement-driven posterior tempering."""

    normalizer: EMANormalizerConfig = field(default_factory=EMANormalizerConfig)
    mapping: RetentionMappingConfig = field(default_factory=RetentionMappingConfig)

    def validate(self) -> None:
        self.normalizer.validate()
        self.mapping.validate()


class EnsembleDisagreementSurprise:
    """Normalize mean predictive variance across ensemble members."""

    def __init__(self, normalizer: EMANormalizerConfig | None = None) -> None:
        self.normalizer = EMASurpriseNormalizer(normalizer)

    def observe(self, predictions: Tensor) -> SurpriseObservation:
        if predictions.ndim < 2 or predictions.shape[0] < 2:
            raise ValueError(
                "ensemble predictions must have shape [members, batch, ...] with >=2 members"
            )
        values = predictions.detach().float()
        if not torch.isfinite(values).all():
            raise FloatingPointError("ensemble predictions contain nonfinite values")
        raw = float(values.var(dim=0, unbiased=False).mean().item())
        return self.normalizer.observe(raw)

    def state_dict(self) -> dict[str, object]:
        return {"version": 1, "normalizer": self.normalizer.state_dict()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        version = checkpoint_integer(
            state.get("version"),
            name="ensemble-surprise checkpoint version",
        )
        if version != 1:
            raise ValueError("unsupported ensemble-surprise checkpoint version")
        payload = state["normalizer"]
        if not isinstance(payload, dict):
            raise TypeError("ensemble surprise normalizer state must be a dictionary")
        self.normalizer.load_state_dict(payload)
