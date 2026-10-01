"""Recurrent actor and critics for hidden-context Soft Actor-Critic."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from rl_bgd.models.mlp import MLP


class RecurrentSACActor(nn.Module):
    """Observation-history GRU with a tanh-squashed Gaussian SAC head."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        recurrent_hidden_dim: int = 64,
        encoder_hidden_dims: Sequence[int] = (64,),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if (
            min(
                observation_dim,
                action_dim,
                recurrent_hidden_dim,
            )
            < 1
        ):
            raise ValueError("recurrent SAC dimensions must be positive")
        if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
            raise ValueError("action bounds must be vectors of action_dim")
        if not torch.all(action_high > action_low):
            raise ValueError("action_high must exceed action_low elementwise")
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
        self.log_std_head = nn.Linear(
            recurrent_hidden_dim,
            action_dim,
        )
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
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

    def step_hidden(
        self,
        observation: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        return self.recurrent(
            self.encoder(observation),
            hidden,
        )

    def _distribution(
        self,
        hidden: Tensor,
    ) -> Normal:
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(
            self.log_std_min,
            self.log_std_max,
        )
        return Normal(
            mean,
            log_std.exp(),
        )

    def _from_pre_tanh(
        self,
        normal: Normal,
        pre_tanh: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
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
        return (
            action,
            log_prob,
        )

    def sample_step(
        self,
        observation: Tensor,
        hidden: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        next_hidden = self.step_hidden(
            observation,
            hidden,
        )
        normal = self._distribution(next_hidden)
        pre_tanh = normal.rsample()
        (
            action,
            log_prob,
        ) = self._from_pre_tanh(
            normal,
            pre_tanh,
        )
        deterministic = torch.tanh(normal.mean) * self.action_scale + self.action_bias
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
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        next_hidden = self.step_hidden(
            observation,
            hidden,
        )
        normal = self._distribution(next_hidden)
        action = torch.tanh(normal.mean) * self.action_scale + self.action_bias
        return (
            action,
            next_hidden,
        )

    def hidden_sequence(
        self,
        observations: Tensor,
        episode_starts: Tensor,
        initial_hidden: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        if observations.ndim != 3:
            raise ValueError("observations must have shape [batch, time, features]")
        if episode_starts.shape != (
            observations.shape[0],
            observations.shape[1],
            1,
        ):
            raise ValueError("episode_starts must have shape [batch, time, 1]")
        batch_size = observations.shape[0]
        hidden = (
            self.initial_state(batch_size)
            if initial_hidden is None
            else initial_hidden.reshape(
                batch_size,
                self.recurrent_hidden_dim,
            )
        )
        outputs: list[Tensor] = []
        for index in range(observations.shape[1]):
            keep = (
                ~episode_starts[
                    :,
                    index,
                ]
            ).to(hidden.dtype)
            hidden = hidden * keep
            hidden = self.step_hidden(
                observations[
                    :,
                    index,
                ],
                hidden,
            )
            outputs.append(hidden)
        return (
            torch.stack(
                outputs,
                dim=1,
            ),
            hidden,
        )

    def sample_sequence(
        self,
        observations: Tensor,
        episode_starts: Tensor,
        initial_hidden: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        (
            hidden_sequence,
            final_hidden,
        ) = self.hidden_sequence(
            observations,
            episode_starts,
            initial_hidden,
        )
        normal = self._distribution(hidden_sequence)
        pre_tanh = normal.rsample()
        (
            action,
            log_prob,
        ) = self._from_pre_tanh(
            normal,
            pre_tanh,
        )
        deterministic = torch.tanh(normal.mean) * self.action_scale + self.action_bias
        return (
            action,
            log_prob,
            deterministic,
            hidden_sequence,
            final_hidden,
        )

    def sample_next_from_hidden(
        self,
        next_observations: Tensor,
        current_hidden: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        if next_observations.shape[:2] != current_hidden.shape[:2]:
            raise ValueError("next observations/current hidden batch-time shapes differ")
        batch_size, time_steps = next_observations.shape[:2]
        next_hidden = self.step_hidden(
            next_observations.reshape(
                batch_size * time_steps,
                -1,
            ),
            current_hidden.reshape(
                batch_size * time_steps,
                -1,
            ),
        ).reshape(
            batch_size,
            time_steps,
            -1,
        )
        normal = self._distribution(next_hidden)
        pre_tanh = normal.rsample()
        (
            action,
            log_prob,
        ) = self._from_pre_tanh(
            normal,
            pre_tanh,
        )
        deterministic = torch.tanh(normal.mean) * self.action_scale + self.action_bias
        return (
            action,
            log_prob,
            deterministic,
            next_hidden,
        )

    def forward(
        self,
        observations: Tensor,
        episode_starts: Tensor,
        initial_hidden: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Functional-call compatible sequence sampling."""

        return self.sample_sequence(
            observations,
            episode_starts,
            initial_hidden,
        )


class RecurrentQNetwork(nn.Module):
    """Q network whose recurrent state summarizes observation history."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        recurrent_hidden_dim: int = 64,
        encoder_hidden_dims: Sequence[int] = (64,),
        q_hidden_dims: Sequence[int] = (
            64,
            64,
        ),
    ) -> None:
        super().__init__()
        if (
            min(
                observation_dim,
                action_dim,
                recurrent_hidden_dim,
            )
            < 1
        ):
            raise ValueError("recurrent critic dimensions must be positive")
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
        self.q_head = MLP(
            recurrent_hidden_dim + action_dim,
            1,
            hidden_dims=q_hidden_dims,
        )

    def initial_state(
        self,
        batch_size: int,
    ) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.recurrent_hidden_dim,
            device=parameter.device,
            dtype=torch.float32,
        )

    def step_hidden(
        self,
        observation: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        return self.recurrent(
            self.encoder(observation),
            hidden,
        )

    def hidden_sequence(
        self,
        observations: Tensor,
        episode_starts: Tensor,
        initial_hidden: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        if observations.ndim != 3:
            raise ValueError("critic observations must have shape [batch, time, features]")
        if episode_starts.shape != (
            observations.shape[0],
            observations.shape[1],
            1,
        ):
            raise ValueError("critic episode_starts shape mismatch")
        batch_size = observations.shape[0]
        hidden = (
            self.initial_state(batch_size)
            if initial_hidden is None
            else initial_hidden.reshape(
                batch_size,
                self.recurrent_hidden_dim,
            )
        )
        outputs: list[Tensor] = []
        for index in range(observations.shape[1]):
            hidden = hidden * (
                ~episode_starts[
                    :,
                    index,
                ]
            ).to(hidden.dtype)
            hidden = self.step_hidden(
                observations[
                    :,
                    index,
                ],
                hidden,
            )
            outputs.append(hidden)
        return (
            torch.stack(
                outputs,
                dim=1,
            ),
            hidden,
        )

    def q_from_hidden(
        self,
        hidden: Tensor,
        actions: Tensor,
    ) -> Tensor:
        if hidden.shape[:2] != actions.shape[:2]:
            raise ValueError("critic hidden/action batch-time shapes differ")
        return self.q_head(
            torch.cat(
                [
                    hidden,
                    actions,
                ],
                dim=-1,
            )
        )

    def next_q_from_hidden(
        self,
        next_observations: Tensor,
        actions: Tensor,
        current_hidden: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        if next_observations.shape[:2] != current_hidden.shape[:2]:
            raise ValueError("next observation/current hidden batch-time shapes differ")
        batch_size, time_steps = next_observations.shape[:2]
        next_hidden = self.step_hidden(
            next_observations.reshape(
                batch_size * time_steps,
                -1,
            ),
            current_hidden.reshape(
                batch_size * time_steps,
                -1,
            ),
        ).reshape(
            batch_size,
            time_steps,
            -1,
        )
        return (
            self.q_from_hidden(
                next_hidden,
                actions,
            ),
            next_hidden,
        )

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        episode_starts: Tensor,
        initial_hidden: Tensor | None = None,
    ) -> Tensor:
        hidden, _ = self.hidden_sequence(
            observations,
            episode_starts,
            initial_hidden,
        )
        return self.q_from_hidden(
            hidden,
            actions,
        )
