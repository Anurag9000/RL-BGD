"""Bayesian PPO networks for the UCL oracle-boundary baseline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from rl_bgd.baselines.ucl import UCLBayesianLayer, positive_sigma


class UCLBayesianLinear(nn.Module):
    """Bayesian linear layer with row-shared weight uncertainty."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        ratio: float = 1.0 / 32.0,
    ) -> None:
        super().__init__()
        if min(in_features, out_features) < 1:
            raise ValueError("UCL Bayesian layer dimensions must be positive")
        if not 0.0 < ratio < 1.0:
            raise ValueError("UCL Bayesian variance ratio must lie in (0, 1)")
        self.in_features = in_features
        self.out_features = out_features

        total_variance = 2.0 / float(in_features)
        noise_variance = total_variance * ratio
        mean_variance = total_variance - noise_variance
        noise_std = math.sqrt(noise_variance)
        mean_std = math.sqrt(mean_variance)
        bound = math.sqrt(3.0) * mean_std
        rho_init = math.log(math.expm1(noise_std))

        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-bound, bound)
        )
        self.weight_rho = nn.Parameter(torch.full((out_features, 1), rho_init))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), rho_init))

    def forward(
        self,
        x: Tensor,
        *,
        sample_weights: bool,
    ) -> Tensor:
        if sample_weights:
            weight = self.weight_mu + positive_sigma(self.weight_rho) * (
                torch.randn_like(self.weight_mu)
            )
            bias = self.bias_mu + positive_sigma(self.bias_rho) * (torch.randn_like(self.bias_mu))
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class UCLBayesianMLP(nn.Module):
    """Bayesian hidden trunk with deterministic output head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (64, 64),
        ratio: float = 1.0 / 32.0,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("UCL MLP requires at least one hidden layer")
        dims = [input_dim, *hidden_dims]
        self.hidden_layers = nn.ModuleList(
            UCLBayesianLinear(
                dims[index],
                dims[index + 1],
                ratio=ratio,
            )
            for index in range(len(hidden_dims))
        )
        self.output = nn.Linear(hidden_dims[-1], output_dim)

    def forward(
        self,
        x: Tensor,
        *,
        sample_weights: bool,
    ) -> Tensor:
        for module in self.hidden_layers:
            layer = cast(UCLBayesianLinear, module)
            x = torch.tanh(
                layer(
                    x,
                    sample_weights=sample_weights,
                )
            )
        return self.output(x)


class UCLPPOActor(nn.Module):
    """Tanh-squashed Gaussian policy with Bayesian hidden features."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: Sequence[int] = (64, 64),
        ratio: float = 1.0 / 32.0,
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.backbone = UCLBayesianMLP(
            observation_dim,
            action_dim,
            hidden_dims=hidden_dims,
            ratio=ratio,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))
        self.action_scale: Tensor
        self.action_bias: Tensor
        self.register_buffer(
            "action_scale",
            ((action_high - action_low) / 2.0).float(),
        )
        self.register_buffer(
            "action_bias",
            ((action_high + action_low) / 2.0).float(),
        )

    @property
    def bayesian_layers(self) -> Sequence[UCLBayesianLinear]:
        return tuple(
            cast(UCLBayesianLinear, layer)
            for layer in self.backbone.hidden_layers
        )

    def distribution(
        self,
        observation: Tensor,
        *,
        sample_weights: bool,
    ) -> Normal:
        mean = self.backbone(
            observation,
            sample_weights=sample_weights,
        )
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
        *,
        sample_weights: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normal = self.distribution(
            observation,
            sample_weights=sample_weights,
        )
        pre_tanh = normal.rsample()
        action, log_prob = self._from_pre_tanh(normal, pre_tanh)
        deterministic = torch.tanh(normal.mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic

    def evaluate_actions(
        self,
        observation: Tensor,
        action: Tensor,
        *,
        sample_weights: bool,
    ) -> tuple[Tensor, Tensor]:
        normal = self.distribution(
            observation,
            sample_weights=sample_weights,
        )
        normalized = (action - self.action_bias) / self.action_scale
        normalized = normalized.clamp(
            -1.0 + 1e-6,
            1.0 - 1e-6,
        )
        pre_tanh = 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))
        _, log_prob = self._from_pre_tanh(normal, pre_tanh)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def deterministic(self, observation: Tensor) -> Tensor:
        normal = self.distribution(
            observation,
            sample_weights=False,
        )
        return torch.tanh(normal.mean) * self.action_scale + self.action_bias


class UCLValueNetwork(nn.Module):
    """Bayesian hidden value model with deterministic scalar head."""

    def __init__(
        self,
        observation_dim: int,
        *,
        hidden_dims: Sequence[int] = (64, 64),
        ratio: float = 1.0 / 32.0,
    ) -> None:
        super().__init__()
        self.net = UCLBayesianMLP(
            observation_dim,
            1,
            hidden_dims=hidden_dims,
            ratio=ratio,
        )

    @property
    def bayesian_layers(self) -> Sequence[UCLBayesianLinear]:
        return tuple(
            cast(UCLBayesianLinear, layer)
            for layer in self.net.hidden_layers
        )

    def forward(
        self,
        observation: Tensor,
        *,
        sample_weights: bool,
    ) -> Tensor:
        return self.net(
            observation,
            sample_weights=sample_weights,
        )
