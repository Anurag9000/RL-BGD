"""Squashed Gaussian policy used by Soft Actor-Critic."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from rl_bgd.models.mlp import MLP


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed diagonal Gaussian policy with affine action scaling."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: Sequence[int] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
            raise ValueError("action bounds must be vectors of action_dim")
        if not torch.all(action_high > action_low):
            raise ValueError("action_high must exceed action_low elementwise")
        feature_dim = hidden_dims[-1] if hidden_dims else max(64, observation_dim)
        self.backbone = MLP(
            observation_dim,
            feature_dim,
            hidden_dims=hidden_dims[:-1] if hidden_dims else (),
        )
        self.mean_head = nn.Linear(feature_dim, action_dim)
        self.log_std_head = nn.Linear(feature_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", action_scale.float())
        self.register_buffer("action_bias", action_bias.float())

    def distribution_parameters(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(observation)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(
            self.log_std_min, self.log_std_max
        )
        return mean, log_std

    def sample(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        normal = Normal(mean, log_std.exp())
        pre_tanh = normal.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        correction = 2.0 * (
            math.log(2.0)
            - pre_tanh
            - torch.nn.functional.softplus(-2.0 * pre_tanh)
        )
        log_prob = normal.log_prob(pre_tanh) - correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.log(self.action_scale).sum()
        deterministic = (
            torch.tanh(mean) * self.action_scale + self.action_bias
        )
        return action, log_prob, deterministic

    def deterministic(self, observation: Tensor) -> Tensor:
        mean, _ = self.distribution_parameters(observation)
        return torch.tanh(mean) * self.action_scale + self.action_bias
