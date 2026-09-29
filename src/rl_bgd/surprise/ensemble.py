"""Ensemble-disagreement surprise primitive."""

from __future__ import annotations

import torch
from torch import Tensor

from rl_bgd.surprise.base import (
    EMANormalizerConfig,
    EMASurpriseNormalizer,
    SurpriseObservation,
)


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
        if state.get("version") != 1:
            raise ValueError("unsupported ensemble-surprise checkpoint version")
        payload = state["normalizer"]
        if not isinstance(payload, dict):
            raise TypeError("ensemble surprise normalizer state must be a dictionary")
        self.normalizer.load_state_dict(payload)
