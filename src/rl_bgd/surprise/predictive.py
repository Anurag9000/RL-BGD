"""Predictive negative-log-likelihood surprise primitive."""

from __future__ import annotations

import torch
from torch import Tensor

from rl_bgd.surprise.base import (
    EMANormalizerConfig,
    EMASurpriseNormalizer,
    SurpriseObservation,
)


class PredictiveSurprise:
    """Normalize model-provided per-sample negative log likelihoods."""

    def __init__(self, normalizer: EMANormalizerConfig | None = None) -> None:
        self.normalizer = EMASurpriseNormalizer(normalizer)

    def observe_nll(self, negative_log_likelihood: Tensor) -> SurpriseObservation:
        values = negative_log_likelihood.detach().float().reshape(-1)
        if values.numel() == 0:
            raise ValueError("predictive surprise requires at least one NLL value")
        if not torch.isfinite(values).all():
            raise FloatingPointError("predictive NLL contains nonfinite values")
        return self.normalizer.observe(float(values.mean().item()))

    def state_dict(self) -> dict[str, object]:
        return {"version": 1, "normalizer": self.normalizer.state_dict()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported predictive-surprise checkpoint version")
        payload = state["normalizer"]
        if not isinstance(payload, dict):
            raise TypeError("predictive surprise normalizer state must be a dictionary")
        self.normalizer.load_state_dict(payload)
