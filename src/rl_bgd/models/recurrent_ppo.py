"""Recurrent policy and value networks for hidden-context PPO."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from rl_bgd.models.mlp import MLP


class RecurrentPPOSquashedGaussianActor(nn.Module):
    """GRU policy with exact log-prob evaluation for tanh-squashed actions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        recurrent_hidden_dim: int = 64,
        encoder_hidden_dims: Sequence[int] = (64,),
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim, recurrent_hidden_dim) < 1:
            raise ValueError("recurrent PPO dimensions must be positive")
        if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
            raise ValueError("action bounds must be vectors of action_dim")
        if not torch.all(action_high > action_low):
            raise ValueError("action_high must exceed action_low elementwise")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.encoder = MLP(
            observation_dim,
            recurrent_hidden_dim,
            hidden_dims=encoder_hidden_dims,
        )
        self.recurrent = nn.GRUCell(
            recurrent_hidden_dim,
            recurrent_hidden_dim,
        )
        self.mean_head = nn.Linear(
            recurrent_hidden_dim,
            action_dim,
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

    def initial_state(
        self,
        batch_size: int = 1,
    ) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return torch.zeros(
            batch_size,
            self.recurrent_hidden_dim,
            device=self.action_scale.device,
            dtype=torch.float32,
        )

    def _features_step(
        self,
        observation: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        encoded = self.encoder(observation)
        return self.recurrent(
            encoded,
            hidden,
        )

    def _distribution_from_hidden(
        self,
        hidden: Tensor,
    ) -> Normal:
        mean = self.mean_head(hidden)
        log_std = self.log_std.clamp(-20.0, 2.0).expand_as(mean)
        return Normal(
            mean,
            log_std.exp(),
        )

    def _from_pre_tanh(
        self,
        normal: Normal,
        pre_tanh: Tensor,
    ) -> tuple[Tensor, Tensor]:
        squashed = torch.tanh(pre_tanh)
        action = (
            squashed * self.action_scale
            + self.action_bias
        )
        correction = 2.0 * (
            math.log(2.0)
            - pre_tanh
            - torch.nn.functional.softplus(
                -2.0 * pre_tanh
            )
        )
        log_prob = (
            normal.log_prob(pre_tanh)
            - correction
        ).sum(
            dim=-1,
            keepdim=True,
        )
        log_prob -= torch.log(
            self.action_scale
        ).sum()
        return action, log_prob

    def sample_step(
        self,
        observation: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample one action and return the post-observation hidden state."""

        next_hidden = self._features_step(
            observation,
            hidden,
        )
        normal = self._distribution_from_hidden(
            next_hidden
        )
        pre_tanh = normal.rsample()
        action, log_prob = self._from_pre_tanh(
            normal,
            pre_tanh,
        )
        deterministic = (
            torch.tanh(normal.mean)
            * self.action_scale
            + self.action_bias
        )
        return (
            action,
            log_prob,
            deterministic,
            next_hidden,
        )

    def deterministic_step(
        self,
        observation: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        next_hidden = self._features_step(
            observation,
            hidden,
        )
        normal = self._distribution_from_hidden(
            next_hidden
        )
        action = (
            torch.tanh(normal.mean)
            * self.action_scale
            + self.action_bias
        )
        return action, next_hidden

    def _sequence_hidden(
        self,
        observations: Tensor,
        initial_hidden: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if observations.ndim != 2:
            raise ValueError(
                "recurrent PPO observations must have shape [time, features]"
            )
        if episode_starts.shape != (
            observations.shape[0],
            1,
        ):
            raise ValueError(
                "episode_starts must have shape [time, 1]"
            )
        hidden = initial_hidden.reshape(
            1,
            self.recurrent_hidden_dim,
        )
        outputs: list[Tensor] = []
        for index in range(
            observations.shape[0]
        ):
            if bool(
                episode_starts[
                    index,
                    0,
                ].item()
            ):
                hidden = torch.zeros_like(
                    hidden
                )
            hidden = self._features_step(
                observations[
                    index
                ].unsqueeze(0),
                hidden,
            )
            outputs.append(hidden)
        return (
            torch.cat(
                outputs,
                dim=0,
            ),
            hidden,
        )

    def evaluate_actions_sequence(
        self,
        observations: Tensor,
        actions: Tensor,
        initial_hidden: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden_sequence, final_hidden = (
            self._sequence_hidden(
                observations,
                initial_hidden,
                episode_starts,
            )
        )
        normal = self._distribution_from_hidden(
            hidden_sequence
        )
        normalized = (
            actions - self.action_bias
        ) / self.action_scale
        normalized = normalized.clamp(
            -1.0 + 1e-6,
            1.0 - 1e-6,
        )
        pre_tanh = 0.5 * (
            torch.log1p(normalized)
            - torch.log1p(-normalized)
        )
        _, log_prob = self._from_pre_tanh(
            normal,
            pre_tanh,
        )
        entropy = normal.entropy().sum(
            dim=-1,
            keepdim=True,
        )
        return (
            log_prob,
            entropy,
            final_hidden,
        )

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        initial_hidden: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Functional-call compatible recurrent action evaluation."""

        return self.evaluate_actions_sequence(
            observations,
            actions,
            initial_hidden,
            episode_starts,
        )


class RecurrentValueNetwork(nn.Module):
    """GRU state-value network with explicit hidden-state reset masks."""

    def __init__(
        self,
        observation_dim: int,
        *,
        recurrent_hidden_dim: int = 64,
        encoder_hidden_dims: Sequence[int] = (64,),
    ) -> None:
        super().__init__()
        if min(
            observation_dim,
            recurrent_hidden_dim,
        ) < 1:
            raise ValueError(
                "recurrent value dimensions must be positive"
            )
        self.recurrent_hidden_dim = (
            recurrent_hidden_dim
        )
        self.encoder = MLP(
            observation_dim,
            recurrent_hidden_dim,
            hidden_dims=encoder_hidden_dims,
        )
        self.recurrent = nn.GRUCell(
            recurrent_hidden_dim,
            recurrent_hidden_dim,
        )
        self.value_head = nn.Linear(
            recurrent_hidden_dim,
            1,
        )

    def initial_state(
        self,
        batch_size: int = 1,
    ) -> Tensor:
        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive"
            )
        parameter = next(
            self.parameters()
        )
        return torch.zeros(
            batch_size,
            self.recurrent_hidden_dim,
            device=parameter.device,
            dtype=torch.float32,
        )

    def step(
        self,
        observation: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(observation)
        next_hidden = self.recurrent(
            encoded,
            hidden,
        )
        return (
            self.value_head(
                next_hidden
            ),
            next_hidden,
        )

    def forward(
        self,
        observations: Tensor,
        initial_hidden: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if observations.ndim != 2:
            raise ValueError(
                "recurrent value observations must have shape [time, features]"
            )
        if episode_starts.shape != (
            observations.shape[0],
            1,
        ):
            raise ValueError(
                "episode_starts must have shape [time, 1]"
            )
        hidden = initial_hidden.reshape(
            1,
            self.recurrent_hidden_dim,
        )
        values: list[Tensor] = []
        for index in range(
            observations.shape[0]
        ):
            if bool(
                episode_starts[
                    index,
                    0,
                ].item()
            ):
                hidden = torch.zeros_like(
                    hidden
                )
            value, hidden = self.step(
                observations[
                    index
                ].unsqueeze(0),
                hidden,
            )
            values.append(value)
        return (
            torch.cat(
                values,
                dim=0,
            ),
            hidden,
        )
