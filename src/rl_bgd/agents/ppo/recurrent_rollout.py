"""Sequence-preserving on-policy storage for recurrent PPO."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from rl_bgd.utils.checkpoint_progress import checkpoint_integer


@dataclass(frozen=True)
class RecurrentPPORolloutBatch:
    """One contiguous truncated-BPTT chunk."""

    observations: Tensor
    actions: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    returns: Tensor
    old_values: Tensor
    episode_starts: Tensor
    initial_actor_hidden: Tensor
    initial_value_hidden: Tensor


class RecurrentRolloutBuffer:
    """Fixed-horizon rollout with recurrent behavior-state snapshots."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        actor_hidden_dim: int,
        value_hidden_dim: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        if (
            min(
                capacity,
                observation_dim,
                action_dim,
                actor_hidden_dim,
                value_hidden_dim,
            )
            < 1
        ):
            raise ValueError("recurrent rollout dimensions/capacity must be positive")
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
        self.episode_starts = torch.empty(
            (capacity, 1),
            dtype=torch.bool,
            device=self.device,
        )
        self.actor_hiddens = torch.empty(
            (
                capacity,
                actor_hidden_dim,
            ),
            device=self.device,
        )
        self.value_hiddens = torch.empty(
            (
                capacity,
                value_hidden_dim,
            ),
            device=self.device,
        )
        self.advantages: Tensor | None = None
        self.returns: Tensor | None = None

    def __len__(
        self,
    ) -> int:
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
        episode_start: bool,
        actor_hidden: Tensor,
        value_hidden: Tensor,
    ) -> None:
        if self.size >= self.capacity:
            raise RuntimeError("recurrent rollout buffer is full")
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
        self.rewards[
            index,
            0,
        ] = float(reward)
        self.terminated[
            index,
            0,
        ] = terminated
        self.truncated[
            index,
            0,
        ] = truncated
        self.values[
            index,
            0,
        ] = (
            value.detach()
            .to(
                self.device,
                dtype=torch.float32,
            )
            .reshape(())
        )
        self.next_values[
            index,
            0,
        ] = (
            next_value.detach()
            .to(
                self.device,
                dtype=torch.float32,
            )
            .reshape(())
        )
        self.log_probs[
            index,
            0,
        ] = (
            log_prob.detach()
            .to(
                self.device,
                dtype=torch.float32,
            )
            .reshape(())
        )
        self.episode_starts[
            index,
            0,
        ] = episode_start
        self.actor_hiddens[index].copy_(
            actor_hidden.detach()
            .to(
                self.device,
                dtype=torch.float32,
            )
            .reshape(-1)
        )
        self.value_hiddens[index].copy_(
            value_hidden.detach()
            .to(
                self.device,
                dtype=torch.float32,
            )
            .reshape(-1)
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
            raise RuntimeError("cannot compute GAE for empty recurrent rollout")
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
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        self.advantages = advantages
        self.returns = returns

    def sequence_batches(
        self,
        sequence_length: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Iterator[RecurrentPPORolloutBatch]:
        """Yield shuffled contiguous chunks while preserving temporal order."""

        if self.advantages is None or self.returns is None:
            raise RuntimeError("compute_gae must be called before recurrent minibatching")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        starts = torch.arange(
            0,
            self.size,
            sequence_length,
            device=self.device,
        )
        order = torch.randperm(
            starts.numel(),
            device=self.device,
            generator=generator,
        )
        for index in order.tolist():
            start = int(starts[index].item())
            end = min(
                start + sequence_length,
                self.size,
            )
            yield RecurrentPPORolloutBatch(
                observations=self.observations[start:end],
                actions=self.actions[start:end],
                old_log_probs=self.log_probs[start:end],
                advantages=self.advantages[start:end],
                returns=self.returns[start:end],
                old_values=self.values[start:end],
                episode_starts=self.episode_starts[start:end],
                initial_actor_hidden=self.actor_hiddens[start].clone(),
                initial_value_hidden=self.value_hiddens[start].clone(),
            )

    def state_dict(
        self,
    ) -> dict[str, Any]:
        size = self.size
        return {
            "version": 1,
            "capacity": self.capacity,
            "size": size,
            "observation_dim": int(self.observations.shape[1]),
            "action_dim": int(self.actions.shape[1]),
            "actor_hidden_dim": int(self.actor_hiddens.shape[1]),
            "value_hidden_dim": int(self.value_hiddens.shape[1]),
            "observations": self.observations[:size].clone(),
            "actions": self.actions[:size].clone(),
            "rewards": self.rewards[:size].clone(),
            "terminated": self.terminated[:size].clone(),
            "truncated": self.truncated[:size].clone(),
            "values": self.values[:size].clone(),
            "next_values": self.next_values[:size].clone(),
            "log_probs": self.log_probs[:size].clone(),
            "episode_starts": self.episode_starts[:size].clone(),
            "actor_hiddens": self.actor_hiddens[:size].clone(),
            "value_hiddens": self.value_hiddens[:size].clone(),
            "advantages": (None if self.advantages is None else self.advantages.clone()),
            "returns": (None if self.returns is None else self.returns.clone()),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if checkpoint_integer(state.get("version"), name="recurrent rollout version") != 1:
            raise ValueError("unsupported recurrent rollout checkpoint version")
        checks = {
            "capacity": self.capacity,
            "observation_dim": int(self.observations.shape[1]),
            "action_dim": int(self.actions.shape[1]),
            "actor_hidden_dim": int(self.actor_hiddens.shape[1]),
            "value_hidden_dim": int(self.value_hiddens.shape[1]),
        }
        for name, expected in checks.items():
            if checkpoint_integer(state[name], name=f"recurrent rollout {name}") != expected:
                raise ValueError(f"recurrent rollout {name} mismatch")
        size = checkpoint_integer(state["size"], name="recurrent rollout size")
        if not 0 <= size <= self.capacity:
            raise ValueError("invalid recurrent rollout size")
        fields = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "values": self.values,
            "next_values": self.next_values,
            "log_probs": self.log_probs,
            "episode_starts": self.episode_starts,
            "actor_hiddens": self.actor_hiddens,
            "value_hiddens": self.value_hiddens,
        }
        checked_tensors: dict[str, Tensor] = {}
        for name, target in fields.items():
            source = state[name]
            if not isinstance(
                source,
                Tensor,
            ):
                raise TypeError(f"recurrent rollout field {name} must be a tensor")
            if source.shape != target[:size].shape:
                raise ValueError(f"recurrent rollout shape mismatch for {name}")
            if source.dtype != target.dtype:
                raise ValueError(f"recurrent rollout dtype mismatch for {name}")
            if source.is_floating_point() and not torch.isfinite(source).all().item():
                raise ValueError("recurrent rollout has non-finite values for " + name)
            checked_tensors[name] = source

        if size > 1:
            previous_done = (
                checked_tensors["terminated"][:-1] | checked_tensors["truncated"][:-1]
            )
            if not torch.equal(checked_tensors["episode_starts"][1:], previous_done):
                raise ValueError("recurrent rollout episode_start boundary mismatch")

        advantages = state.get("advantages")
        returns = state.get("returns")
        if (advantages is None) != (returns is None):
            raise ValueError(
                "recurrent rollout must contain both advantages and returns or neither"
            )
        if advantages is not None:
            if not isinstance(
                advantages,
                Tensor,
            ) or not isinstance(
                returns,
                Tensor,
            ):
                raise TypeError("recurrent rollout advantages/returns must be tensors")
            expected_shape = (size, 1)
            if advantages.shape != expected_shape or returns.shape != expected_shape:
                raise ValueError("recurrent rollout advantage/return shape mismatch")
            if advantages.dtype != torch.float32 or returns.dtype != torch.float32:
                raise ValueError("recurrent rollout advantage/return dtype mismatch")
            if not torch.isfinite(advantages).all().item():
                raise ValueError("recurrent rollout has non-finite advantages")
            if not torch.isfinite(returns).all().item():
                raise ValueError("recurrent rollout has non-finite returns")
        if size == 0 and advantages is not None:
            raise ValueError("empty recurrent rollout cannot have computed advantages")

        prepared_tensors = {
            name: source.to(device=self.device).clone()
            for name, source in checked_tensors.items()
        }
        prepared_advantages = (
            None if advantages is None else advantages.to(device=self.device).clone()
        )
        prepared_returns = (
            None if returns is None else returns.to(device=self.device).clone()
        )

        for name, target in fields.items():
            target[:size].copy_(prepared_tensors[name])
        self.advantages = prepared_advantages
        self.returns = prepared_returns
        self.size = size
