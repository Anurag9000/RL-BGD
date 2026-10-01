"""Predictive negative-log-likelihood surprise and online dynamics model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from rl_bgd.models.mlp import MLP
from rl_bgd.surprise.base import (
    EMANormalizerConfig,
    EMASurpriseNormalizer,
    RetentionMappingConfig,
    SurpriseObservation,
)


@dataclass(frozen=True)
class AdaptivePredictiveRetentionConfig:
    """Online world-model surprise controls for posterior replasticization."""

    normalizer: EMANormalizerConfig = field(default_factory=EMANormalizerConfig)
    mapping: RetentionMappingConfig = field(default_factory=RetentionMappingConfig)
    hidden_dims: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    gradient_clip_norm: float | None = 10.0
    min_log_std: float = -6.0
    max_log_std: float = 2.0

    def validate(self) -> None:
        self.normalizer.validate()
        self.mapping.validate()
        if any(width < 1 for width in self.hidden_dims):
            raise ValueError("predictive-model hidden widths must be positive")
        if self.learning_rate <= 0:
            raise ValueError("predictive-model learning rate must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("predictive-model gradient clip must be positive")
        if self.min_log_std >= self.max_log_std:
            raise ValueError("predictive-model log-std bounds are invalid")


class GaussianTransitionModel(nn.Module):
    """Predict next observation and reward with a diagonal Gaussian likelihood."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 64),
        min_log_std: float = -6.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("predictive-model dimensions must be positive")
        if any(width < 1 for width in hidden_dims):
            raise ValueError("predictive-model hidden widths must be positive")
        if min_log_std >= max_log_std:
            raise ValueError("predictive-model log-std bounds are invalid")
        self.observation_dim = observation_dim
        self.output_dim = observation_dim + 1
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.mean_model = MLP(
            observation_dim + action_dim,
            self.output_dim,
            hidden_dims=hidden_dims,
        )
        self.log_std = nn.Parameter(
            torch.zeros(self.output_dim, dtype=torch.float32)
        )

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        inputs = torch.cat((observations, actions), dim=-1)
        mean = self.mean_model(inputs)
        log_std = self.log_std.clamp(
            self.min_log_std,
            self.max_log_std,
        ).expand_as(mean)
        return mean, log_std

    def negative_log_likelihood(
        self,
        observations: Tensor,
        actions: Tensor,
        next_observations: Tensor,
        rewards: Tensor,
    ) -> Tensor:
        """Return one joint transition NLL per replay item."""

        mean, log_std = self(observations, actions)
        target = torch.cat(
            (
                next_observations,
                rewards.reshape(-1, 1),
            ),
            dim=-1,
        )
        if target.shape != mean.shape:
            raise ValueError("predictive-model target shape mismatch")
        inv_variance = torch.exp(-2.0 * log_std)
        elementwise = (
            0.5 * (target - mean).square() * inv_variance
            + log_std
            + 0.5 * math.log(2.0 * math.pi)
        )
        nll = elementwise.sum(dim=-1, keepdim=True)
        if not torch.isfinite(nll).all():
            raise FloatingPointError("predictive model produced nonfinite NLL")
        return nll


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
