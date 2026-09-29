"""Policy and value networks for continuous-action PPO."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from rl_bgd.models.mlp import MLP


class PPOSquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy with exact action log-prob evaluation."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: Sequence[int] = (256, 256),
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
            raise ValueError("action bounds must be vectors of action_dim")
        if not torch.all(action_high > action_low):
            raise ValueError("action_high must exceed action_low elementwise")

        self.backbone = MLP(
            observation_dim,
            action_dim,
            hidden_dims=hidden_dims,
        )
        self.log_std = nn.Parameter(
            torch.full(
                (action_dim,),
                float(initial_log_std),
            )
        )
        self.register_buffer(
            "action_scale",
            ((action_high - action_low) / 2.0).float(),
        )
        self.register_buffer(
            "action_bias",
            ((action_high + action_low) / 2.0).float(),
        )

    def distribution(self, observation: Tensor) -> Normal:
        mean = self.backbone(observation)
        log_std = self.log_std.clamp(-20.0, 2.0).expand_as(mean)
        return Normal(mean, log_std.exp())

    def _from_pre_tanh(
        self,
        normal: Normal,
        pre_tanh: Tensor,
    ) -> tuple[Tensor, Tensor]:
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        correction = 2.0 * (
            math.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh)
        )
        log_prob = (normal.log_prob(pre_tanh) - correction).sum(
            dim=-1,
            keepdim=True,
        )
        log_prob -= torch.log(self.action_scale).sum()
        return action, log_prob

    def sample(
        self,
        observation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normal = self.distribution(observation)
        pre_tanh = normal.rsample()
        action, log_prob = self._from_pre_tanh(
            normal,
            pre_tanh,
        )
        deterministic = torch.tanh(normal.mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic

    def evaluate_actions(
        self,
        observation: Tensor,
        action: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normal = self.distribution(observation)
        normalized = (action - self.action_bias) / self.action_scale
        normalized = normalized.clamp(
            -1.0 + 1e-6,
            1.0 - 1e-6,
        )
        pre_tanh = 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))
        _, log_prob = self._from_pre_tanh(
            normal,
            pre_tanh,
        )
        entropy = normal.entropy().sum(
            dim=-1,
            keepdim=True,
        )
        return log_prob, entropy

    def deterministic(self, observation: Tensor) -> Tensor:
        normal = self.distribution(observation)
        return torch.tanh(normal.mean) * self.action_scale + self.action_bias

    def forward(
        self,
        observation: Tensor,
        action: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Functional-call compatible action evaluation for BGD-PPO."""
        return self.evaluate_actions(
            observation,
            action,
        )


class ValueNetwork(nn.Module):
    """Scalar state-value approximator."""

    def __init__(
        self,
        observation_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = MLP(
            observation_dim,
            1,
            hidden_dims=hidden_dims,
        )

    def forward(self, observation: Tensor) -> Tensor:
        return self.net(observation)
