"""On-policy rollout storage and GAE for PPO."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import torch
from torch import Tensor


@dataclass(frozen=True)
class PPORolloutBatch:
    """One PPO minibatch with behavior-policy statistics frozen."""

    observations: Tensor
    actions: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    returns: Tensor
    old_values: Tensor


class RolloutBuffer:
    """Fixed-horizon on-policy storage with time-limit-aware GAE."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        if min(
            capacity,
            observation_dim,
            action_dim,
        ) < 1:
            raise ValueError("rollout dimensions/capacity must be positive")
        self.capacity = capacity
        self.device = torch.device(device)
        self.size = 0
        self.observations = torch.empty(
            (capacity, observation_dim),
            device=self.device,
        )
        self.actions = torch.empty(
            (capacity, action_dim),
            device=self.device,
        )
        self.rewards = torch.empty(
            (capacity, 1),
            device=self.device,
        )
        self.terminated = torch.empty(
            (capacity, 1),
            dtype=torch.bool,
            device=self.device,
        )
        self.truncated = torch.empty(
            (capacity, 1),
            dtype=torch.bool,
            device=self.device,
        )
        self.values = torch.empty(
            (capacity, 1),
            device=self.device,
        )
        self.next_values = torch.empty(
            (capacity, 1),
            device=self.device,
        )
        self.log_probs = torch.empty(
            (capacity, 1),
            device=self.device,
        )
        self.advantages: Tensor | None = None
        self.returns: Tensor | None = None

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: Tensor,
        action: Tensor,
        reward: float,
        *,
        terminated: bool,
        truncated: bool,
        value: Tensor,
        next_value: Tensor,
        log_prob: Tensor,
    ) -> None:
        if self.size >= self.capacity:
            raise RuntimeError("rollout buffer is full")
        index = self.size
        self.observations[index].copy_(
            observation.to(
                self.device,
                dtype=torch.float32,
            ).reshape(-1)
        )
        self.actions[index].copy_(
            action.to(
                self.device,
                dtype=torch.float32,
            ).reshape(-1)
        )
        self.rewards[index, 0] = float(reward)
        self.terminated[index, 0] = terminated
        self.truncated[index, 0] = truncated
        self.values[index, 0] = value.detach().to(
            self.device,
            dtype=torch.float32,
        ).reshape(())
        self.next_values[index, 0] = next_value.detach().to(
            self.device,
            dtype=torch.float32,
        ).reshape(())
        self.log_probs[index, 0] = log_prob.detach().to(
            self.device,
            dtype=torch.float32,
        ).reshape(())
        self.size += 1
        self.advantages = None
        self.returns = None

    def compute_gae(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        normalize_advantages: bool = True,
    ) -> None:
        if self.size < 1:
            raise RuntimeError("cannot compute GAE for empty rollout")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")

        advantages = torch.zeros(
            (self.size, 1),
            device=self.device,
        )
        gae = torch.zeros(
            (1,),
            device=self.device,
        )
        for index in range(
            self.size - 1,
            -1,
            -1,
        ):
            bootstrap_mask = (~self.terminated[index]).float()
            continuation_mask = (
                ~(
                    self.terminated[index]
                    | self.truncated[index]
                )
            ).float()
            delta = (
                self.rewards[index]
                + gamma
                * bootstrap_mask
                * self.next_values[index]
                - self.values[index]
            )
            gae = (
                delta
                + gamma
                * gae_lambda
                * continuation_mask
                * gae
            )
            advantages[index] = gae

        returns = advantages + self.values[: self.size]
        if normalize_advantages and self.size > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std(
                    unbiased=False,
                )
                + 1e-8
            )
        self.advantages = advantages
        self.returns = returns

    def batches(
        self,
        minibatch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Iterator[PPORolloutBatch]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError(
                "compute_gae must be called before minibatching"
            )
        if minibatch_size < 1:
            raise ValueError("minibatch_size must be positive")

        order = torch.randperm(
            self.size,
            device=self.device,
            generator=generator,
        )
        for start in range(
            0,
            self.size,
            minibatch_size,
        ):
            indices = order[start : start + minibatch_size]
            yield PPORolloutBatch(
                observations=self.observations[indices],
                actions=self.actions[indices],
                old_log_probs=self.log_probs[indices],
                advantages=self.advantages[indices],
                returns=self.returns[indices],
                old_values=self.values[indices],
            )
