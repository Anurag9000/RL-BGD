"""On-policy rollout storage and GAE for PPO."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from rl_bgd.utils.checkpoint_progress import checkpoint_integer


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
        if (
            min(
                capacity,
                observation_dim,
                action_dim,
            )
            < 1
        ):
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
        self.values[index, 0] = (
            value.detach()
            .to(
                self.device,
                dtype=torch.float32,
            )
            .reshape(())
        )
        self.next_values[index, 0] = (
            next_value.detach()
            .to(
                self.device,
                dtype=torch.float32,
            )
            .reshape(())
        )
        self.log_probs[index, 0] = (
            log_prob.detach()
            .to(
                self.device,
                dtype=torch.float32,
            )
            .reshape(())
        )
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
            continuation_mask = (~(self.terminated[index] | self.truncated[index])).float()
            delta = (
                self.rewards[index]
                + gamma * bootstrap_mask * self.next_values[index]
                - self.values[index]
            )
            gae = delta + gamma * gae_lambda * continuation_mask * gae
            advantages[index] = gae

        returns = advantages + self.values[: self.size]
        if normalize_advantages and self.size > 1:
            advantages = (advantages - advantages.mean()) / (
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
            raise RuntimeError("compute_gae must be called before minibatching")
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

    def state_dict(self) -> dict[str, Any]:
        """Serialize partial or update-ready on-policy rollout state."""

        size = self.size
        return {
            "version": 1,
            "capacity": self.capacity,
            "observation_dim": int(self.observations.shape[1]),
            "action_dim": int(self.actions.shape[1]),
            "size": size,
            "observations": self.observations[:size].clone(),
            "actions": self.actions[:size].clone(),
            "rewards": self.rewards[:size].clone(),
            "terminated": self.terminated[:size].clone(),
            "truncated": self.truncated[:size].clone(),
            "values": self.values[:size].clone(),
            "next_values": self.next_values[:size].clone(),
            "log_probs": self.log_probs[:size].clone(),
            "advantages": (None if self.advantages is None else self.advantages.clone()),
            "returns": (None if self.returns is None else self.returns.clone()),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        """Restore behavior-policy statistics without recomputing them."""

        if checkpoint_integer(state.get("version"), name="PPO rollout version") != 1:
            raise ValueError("unsupported PPO rollout checkpoint version")
        if checkpoint_integer(state["capacity"], name="PPO rollout capacity") != self.capacity:
            raise ValueError("PPO rollout checkpoint capacity mismatch")
        if checkpoint_integer(state["observation_dim"], name="PPO rollout observation_dim") != int(self.observations.shape[1]):
            raise ValueError("PPO rollout observation dimension mismatch")
        if checkpoint_integer(state["action_dim"], name="PPO rollout action_dim") != int(self.actions.shape[1]):
            raise ValueError("PPO rollout action dimension mismatch")

        size = checkpoint_integer(state["size"], name="PPO rollout size")
        if not 0 <= size <= self.capacity:
            raise ValueError("invalid PPO rollout checkpoint size")
        fields = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "values": self.values,
            "next_values": self.next_values,
            "log_probs": self.log_probs,
        }
        checked_tensors: dict[str, Tensor] = {}
        for name, target in fields.items():
            source = state[name]
            if not isinstance(
                source,
                Tensor,
            ):
                raise TypeError(f"PPO rollout checkpoint field {name} must be a tensor")
            if source.shape != target[:size].shape:
                raise ValueError(f"PPO rollout checkpoint shape mismatch for {name}")
            if source.dtype != target.dtype:
                raise ValueError(f"PPO rollout checkpoint dtype mismatch for {name}")
            if source.is_floating_point() and not torch.isfinite(source).all().item():
                raise ValueError("PPO rollout checkpoint has non-finite values for " + name)
            checked_tensors[name] = source

        advantages = state.get("advantages")
        returns = state.get("returns")
        if (advantages is None) != (returns is None):
            raise ValueError(
                "PPO rollout checkpoint must contain both advantages and returns or neither"
            )
        if advantages is not None:
            if not isinstance(
                advantages,
                Tensor,
            ) or not isinstance(
                returns,
                Tensor,
            ):
                raise TypeError("PPO rollout advantages/returns must be tensors")
            expected_shape = (size, 1)
            if advantages.shape != expected_shape or returns.shape != expected_shape:
                raise ValueError("PPO rollout checkpoint advantage/return shape mismatch")
            if advantages.dtype != torch.float32 or returns.dtype != torch.float32:
                raise ValueError("PPO rollout checkpoint advantage/return dtype mismatch")
            if not torch.isfinite(advantages).all().item():
                raise ValueError("PPO rollout checkpoint has non-finite advantages")
            if not torch.isfinite(returns).all().item():
                raise ValueError("PPO rollout checkpoint has non-finite returns")
        if size == 0 and advantages is not None:
            raise ValueError("empty PPO rollout cannot have computed advantages")

        for name, target in fields.items():
            target[:size].copy_(checked_tensors[name].to(device=self.device))
        self.advantages = (
            None if advantages is None else advantages.to(device=self.device).clone()
        )
        self.returns = None if returns is None else returns.to(device=self.device).clone()
        self.size = size
