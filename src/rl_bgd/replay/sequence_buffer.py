"""Chronological sequence replay with burn-in and evidence metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class SequenceReplayBatch:
    """One batch of chronological replay windows."""

    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    terminated: Tensor
    truncated: Tensor
    episode_starts: Tensor
    transition_ids: Tensor
    insertion_steps: Tensor
    usage_counts: Tensor
    fresh: Tensor
    burn_in: int

    @property
    def unroll_slice(self) -> slice:
        return slice(self.burn_in, None)

    @property
    def unroll_observations(self) -> Tensor:
        return self.observations[:, self.unroll_slice]

    @property
    def unroll_actions(self) -> Tensor:
        return self.actions[:, self.unroll_slice]

    @property
    def unroll_rewards(self) -> Tensor:
        return self.rewards[:, self.unroll_slice]

    @property
    def unroll_next_observations(self) -> Tensor:
        return self.next_observations[:, self.unroll_slice]

    @property
    def unroll_terminated(self) -> Tensor:
        return self.terminated[:, self.unroll_slice]

    @property
    def unroll_truncated(self) -> Tensor:
        return self.truncated[:, self.unroll_slice]

    @property
    def bootstrap_mask(self) -> Tensor:
        """Bootstrap through time-limit truncations but not true terminals."""

        return (~self.unroll_terminated).float()


class SequenceReplayBuffer:
    """Ring replay buffer that samples contiguous logical windows.

    Storage may wrap physically, but sampled windows are reconstructed in
    insertion order. Burn-in transitions reconstruct recurrent state and do not
    increment Bayesian evidence-use counts. Only optimized unroll transitions
    count as replay evidence.
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        *,
        storage_device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("observation/action dimensions must be positive")
        self.capacity = capacity
        self.device = torch.device(storage_device)
        self.observations = torch.empty(
            (capacity, observation_dim),
            device=self.device,
            dtype=dtype,
        )
        self.actions = torch.empty(
            (capacity, action_dim),
            device=self.device,
            dtype=dtype,
        )
        self.rewards = torch.empty(
            (capacity, 1),
            device=self.device,
            dtype=dtype,
        )
        self.next_observations = torch.empty(
            (capacity, observation_dim),
            device=self.device,
            dtype=dtype,
        )
        self.terminated = torch.empty(
            (capacity, 1),
            device=self.device,
            dtype=torch.bool,
        )
        self.truncated = torch.empty(
            (capacity, 1),
            device=self.device,
            dtype=torch.bool,
        )
        self.episode_starts = torch.empty(
            (capacity, 1),
            device=self.device,
            dtype=torch.bool,
        )
        self.transition_ids = torch.empty(
            (capacity, 1),
            device=self.device,
            dtype=torch.long,
        )
        self.insertion_steps = torch.empty(
            (capacity, 1),
            device=self.device,
            dtype=torch.long,
        )
        self.usage_counts = torch.zeros(
            (capacity, 1),
            device=self.device,
            dtype=torch.long,
        )
        self._fresh = torch.zeros(
            (capacity, 1),
            device=self.device,
            dtype=torch.bool,
        )
        self._size = 0
        self._position = 0
        self._next_transition_id = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: Tensor,
        action: Tensor,
        reward: float | Tensor,
        next_observation: Tensor,
        *,
        terminated: bool,
        truncated: bool,
        episode_start: bool,
        insertion_step: int,
    ) -> int:
        index = self._position
        self.observations[index].copy_(
            observation.to(
                self.device,
                dtype=self.observations.dtype,
            )
        )
        self.actions[index].copy_(
            action.to(
                self.device,
                dtype=self.actions.dtype,
            )
        )
        self.rewards[index, 0] = torch.as_tensor(
            reward,
            device=self.device,
            dtype=self.rewards.dtype,
        )
        self.next_observations[index].copy_(
            next_observation.to(
                self.device,
                dtype=self.next_observations.dtype,
            )
        )
        self.terminated[index, 0] = terminated
        self.truncated[index, 0] = truncated
        self.episode_starts[index, 0] = episode_start
        transition_id = self._next_transition_id
        self.transition_ids[index, 0] = transition_id
        self.insertion_steps[index, 0] = insertion_step
        self.usage_counts[index, 0] = 0
        self._fresh[index, 0] = True
        self._next_transition_id += 1
        self._position = (self._position + 1) % self.capacity
        self._size = min(
            self._size + 1,
            self.capacity,
        )
        return transition_id

    def _logical_physical_indices(self) -> Tensor:
        oldest = (self._position - self._size) % self.capacity
        return (
            oldest
            + torch.arange(
                self._size,
                device=self.device,
                dtype=torch.long,
            )
        ) % self.capacity

    def logical_transition_ids(self) -> Tensor:
        """Return resident transition IDs in chronological order."""

        indices = self._logical_physical_indices()
        return self.transition_ids[indices].clone()

    def _record_evidence_usage(
        self,
        physical_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Increment counts exactly even when sampled windows overlap."""

        flat = physical_indices.reshape(-1)
        prior = self.usage_counts[
            flat,
            0,
        ].clone()
        fresh_before = self._fresh[
            flat,
            0,
        ].clone()

        sorted_indices, order = torch.sort(flat)
        positions = torch.arange(
            flat.numel(),
            device=self.device,
            dtype=torch.long,
        )
        group_start = torch.ones_like(
            positions,
            dtype=torch.bool,
        )
        if flat.numel() > 1:
            group_start[1:] = sorted_indices[1:] != sorted_indices[:-1]
        start_positions = torch.where(
            group_start,
            positions,
            torch.zeros_like(positions),
        )
        latest_start = torch.cummax(
            start_positions,
            dim=0,
        ).values
        ranks_sorted = positions - latest_start
        ranks = torch.empty_like(ranks_sorted)
        ranks[order] = ranks_sorted

        usage = (prior + ranks + 1).reshape(
            *physical_indices.shape,
            1,
        )
        fresh = (fresh_before & (ranks == 0)).reshape(
            *physical_indices.shape,
            1,
        )

        increments = torch.ones(
            flat.shape,
            device=self.device,
            dtype=self.usage_counts.dtype,
        )
        self.usage_counts[
            :,
            0,
        ].index_add_(
            0,
            flat,
            increments,
        )
        self._fresh[
            torch.unique(flat),
            0,
        ] = False
        return usage, fresh

    def sample_sequences(
        self,
        batch_size: int,
        *,
        burn_in: int,
        unroll: int,
        generator: torch.Generator | None = None,
    ) -> SequenceReplayBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if burn_in < 0 or unroll < 1:
            raise ValueError("burn_in must be non-negative and unroll positive")
        window = burn_in + unroll
        available_windows = self._size - window + 1
        if available_windows < batch_size:
            raise ValueError("not enough chronological windows for requested sequence batch")

        logical = self._logical_physical_indices()
        chosen = torch.randperm(
            available_windows,
            device=self.device,
            generator=generator,
        )[:batch_size]
        offsets = torch.arange(
            window,
            device=self.device,
            dtype=torch.long,
        )
        logical_positions = chosen.unsqueeze(1) + offsets.unsqueeze(0)
        physical = logical[logical_positions]
        unroll_physical = physical[
            :,
            burn_in:,
        ]
        usage_counts, fresh = self._record_evidence_usage(unroll_physical)
        return SequenceReplayBatch(
            observations=self.observations[physical],
            actions=self.actions[physical],
            rewards=self.rewards[physical],
            next_observations=self.next_observations[physical],
            terminated=self.terminated[physical],
            truncated=self.truncated[physical],
            episode_starts=self.episode_starts[physical],
            transition_ids=self.transition_ids[physical],
            insertion_steps=self.insertion_steps[physical],
            usage_counts=usage_counts,
            fresh=fresh,
            burn_in=burn_in,
        )

    def state_dict(self) -> dict[str, Any]:
        size = self._size
        return {
            "version": 1,
            "capacity": self.capacity,
            "size": size,
            "position": self._position,
            "next_transition_id": self._next_transition_id,
            "observation_dim": int(self.observations.shape[1]),
            "action_dim": int(self.actions.shape[1]),
            "observations": self.observations[:size].clone(),
            "actions": self.actions[:size].clone(),
            "rewards": self.rewards[:size].clone(),
            "next_observations": self.next_observations[:size].clone(),
            "terminated": self.terminated[:size].clone(),
            "truncated": self.truncated[:size].clone(),
            "episode_starts": self.episode_starts[:size].clone(),
            "transition_ids": self.transition_ids[:size].clone(),
            "insertion_steps": self.insertion_steps[:size].clone(),
            "usage_counts": self.usage_counts[:size].clone(),
            "fresh": self._fresh[:size].clone(),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported sequence replay checkpoint version")
        expected = {
            "capacity": self.capacity,
            "observation_dim": int(self.observations.shape[1]),
            "action_dim": int(self.actions.shape[1]),
        }
        for name, value in expected.items():
            saved = state[name]
            if isinstance(saved, bool) or not isinstance(saved, int):
                raise TypeError(f"sequence replay {name} must be an integer")
            if saved != value:
                raise ValueError(f"sequence replay {name} mismatch")
        size = state["size"]
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("sequence replay size must be an integer")
        if not 0 <= size <= self.capacity:
            raise ValueError("invalid sequence replay size")
        fields = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_starts": self.episode_starts,
            "transition_ids": self.transition_ids,
            "insertion_steps": self.insertion_steps,
            "usage_counts": self.usage_counts,
            "fresh": self._fresh,
        }
        for name, target in fields.items():
            source = state[name]
            if not isinstance(
                source,
                Tensor,
            ):
                raise TypeError(f"sequence replay field {name} must be a tensor")
            if source.shape != target[:size].shape:
                raise ValueError(f"sequence replay shape mismatch for {name}")
            if source.dtype != target.dtype:
                raise ValueError(f"sequence replay dtype mismatch for {name}")
            target[:size].copy_(
                source.to(
                    device=self.device,
                    dtype=target.dtype,
                )
            )
        position = state["position"]
        next_transition_id = state["next_transition_id"]
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("sequence replay position must be an integer")
        if isinstance(next_transition_id, bool) or not isinstance(
            next_transition_id,
            int,
        ):
            raise TypeError("sequence replay next transition id must be an integer")
        if not 0 <= position < self.capacity:
            raise ValueError("invalid sequence replay position")
        if size < self.capacity and position != size:
            raise ValueError("sequence replay position is inconsistent with size")
        if next_transition_id < 0:
            raise ValueError("sequence replay next transition id must be non-negative")

        usage = self.usage_counts[:size]
        fresh = self._fresh[:size]
        if torch.any(usage < 0):
            raise ValueError("sequence replay contains negative usage counts")
        if not torch.equal(
            fresh,
            usage == 0,
        ):
            raise ValueError("sequence replay freshness disagrees with usage counts")

        if size:
            transition_ids = self.transition_ids[:size, 0]
            if torch.any(transition_ids < 0):
                raise ValueError("sequence replay contains negative transition IDs")
            if torch.unique(transition_ids).numel() != size:
                raise ValueError("sequence replay contains duplicate transition IDs")
            expected_next = int(transition_ids.max().item()) + 1
        else:
            expected_next = 0
        if next_transition_id != expected_next:
            raise ValueError("sequence replay next transition ID is inconsistent")
        if size:
            oldest = (position - size) % self.capacity
            logical = (oldest + torch.arange(size, device=self.device)) % self.capacity
            expected_ids = torch.arange(
                next_transition_id - size,
                next_transition_id,
                device=self.device,
                dtype=self.transition_ids.dtype,
            )
            if not torch.equal(self.transition_ids[logical, 0], expected_ids):
                raise ValueError("sequence replay chronological transition IDs are inconsistent")

        self._size = size
        self._position = position
        self._next_transition_id = next_transition_id
