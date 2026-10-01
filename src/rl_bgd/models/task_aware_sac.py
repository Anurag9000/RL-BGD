"""Task-aware multi-head actor/critic models for canonical Continual World."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class SharedTaskFeatureMLP(nn.Module):
    """Shared body matching the historical CW default topology."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1 or not hidden_dims:
            raise ValueError("invalid shared feature dimensions")
        layers: list[nn.Module] = []
        previous = input_dim
        for index, hidden in enumerate(hidden_dims):
            if hidden < 1:
                raise ValueError("hidden dimensions must be positive")
            layers.append(nn.Linear(previous, hidden))
            if index == 0 and use_layer_norm:
                layers.append(nn.LayerNorm(hidden))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.LeakyReLU(negative_slope=0.2))
            previous = hidden
        self.net = nn.Sequential(*layers)
        self.output_dim = int(hidden_dims[-1])

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class TaskAwareSquashedGaussianActor(nn.Module):
    """Shared actor body with one Gaussian head per sequence occurrence."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        num_tasks: int,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: Sequence[int] = (256, 256, 256, 256),
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_tasks < 1 or observation_dim <= num_tasks:
            raise ValueError("observation must include a task one-hot suffix")
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.num_tasks = num_tasks
        self.base_observation_dim = observation_dim - num_tasks
        self.action_dim = action_dim
        self.body = SharedTaskFeatureMLP(
            self.base_observation_dim,
            hidden_dims,
            use_layer_norm=use_layer_norm,
        )
        self.mean_head = nn.Linear(
            self.body.output_dim,
            action_dim * num_tasks,
        )
        self.log_std_head = nn.Linear(
            self.body.output_dim,
            action_dim * num_tasks,
        )
        self.register_buffer(
            "action_scale",
            ((action_high - action_low) / 2.0).float(),
        )
        self.register_buffer(
            "action_bias",
            ((action_high + action_low) / 2.0).float(),
        )

    def task_weights(self, observation: Tensor) -> Tensor:
        return observation[..., -self.num_tasks :]

    def _select_head(
        self,
        raw: Tensor,
        observation: Tensor,
    ) -> Tensor:
        reshaped = raw.reshape(
            *raw.shape[:-1],
            self.action_dim,
            self.num_tasks,
        )
        weights = self.task_weights(observation).unsqueeze(-2)
        return (reshaped * weights).sum(dim=-1)

    def distribution(self, observation: Tensor) -> Normal:
        features = self.body(
            observation[..., : self.base_observation_dim]
        )
        mean = self._select_head(
            self.mean_head(features),
            observation,
        )
        log_std = self._select_head(
            self.log_std_head(features),
            observation,
        ).clamp(-20.0, 2.0)
        return Normal(mean, log_std.exp())

    def sample(
        self,
        observation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normal = self.distribution(observation)
        pre_tanh = normal.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        correction = 2.0 * (
            math.log(2.0)
            - pre_tanh
            - torch.nn.functional.softplus(-2.0 * pre_tanh)
        )
        log_prob = (
            normal.log_prob(pre_tanh) - correction
        ).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(self.action_scale).sum()
        deterministic = (
            torch.tanh(normal.mean)
            * self.action_scale
            + self.action_bias
        )
        return action, log_prob, deterministic

    def deterministic(self, observation: Tensor) -> Tensor:
        normal = self.distribution(observation)
        return (
            torch.tanh(normal.mean)
            * self.action_scale
            + self.action_bias
        )


class TaskAwareQNetwork(nn.Module):
    """Shared critic body with one scalar Q head per task occurrence."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        num_tasks: int,
        hidden_dims: Sequence[int] = (256, 256, 256, 256),
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_tasks < 1 or observation_dim <= num_tasks:
            raise ValueError("observation must include a task one-hot suffix")
        self.num_tasks = num_tasks
        self.base_observation_dim = observation_dim - num_tasks
        self.body = SharedTaskFeatureMLP(
            self.base_observation_dim + action_dim,
            hidden_dims,
            use_layer_norm=use_layer_norm,
        )
        self.head = nn.Linear(
            self.body.output_dim,
            num_tasks,
        )

    def forward(
        self,
        observation: Tensor,
        action: Tensor,
    ) -> Tensor:
        task_weights = observation[..., -self.num_tasks :]
        base_observation = observation[
            ..., : self.base_observation_dim
        ]
        features = self.body(
            torch.cat(
                [base_observation, action],
                dim=-1,
            )
        )
        all_heads = self.head(features)
        return (
            all_heads * task_weights
        ).sum(dim=-1, keepdim=True)
