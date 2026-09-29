"""State-action critic network."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from rl_bgd.models.mlp import MLP


class QNetwork(nn.Module):
    """Scalar Q(s, a) approximator."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = MLP(
            observation_dim + action_dim,
            1,
            hidden_dims=hidden_dims,
        )

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        if observation.shape[:-1] != action.shape[:-1]:
            raise ValueError("observation/action batch dimensions must match")
        return self.net(torch.cat([observation, action], dim=-1))
