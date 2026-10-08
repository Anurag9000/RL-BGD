"""Device-aware replay buffer with evidence-reuse metadata."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _checkpoint_int(
    value: object,
    *,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class ReplayBatch:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    terminated: Tensor
    truncated: Tensor
    transition_ids: Tensor
    insertion_steps: Tensor
    usage_counts: Tensor
    fresh: Tensor

    @property
    def bootstrap_mask(self) -> Tensor:
        """Preserve bootstrap across time-limit truncations."""
        return (~self.terminated).float()


class ReplayBuffer:
    """Fixed-capacity tensor replay buffer with evidence metadata."""

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
            raise ValueError("capacity must be >= 1")
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        device = torch.device(storage_device)
        self.capacity = capacity
        self.device = device
        self.observations = torch.empty(
            (capacity, observation_dim),
            device=device,
            dtype=dtype,
        )
        self.actions = torch.empty((capacity, action_dim), device=device, dtype=dtype)
        self.rewards = torch.empty((capacity, 1), device=device, dtype=dtype)
        self.next_observations = torch.empty(
            (capacity, observation_dim),
            device=device,
            dtype=dtype,
        )
        self.terminated = torch.empty((capacity, 1), device=device, dtype=torch.bool)
        self.truncated = torch.empty((capacity, 1), device=device, dtype=torch.bool)
        self.transition_ids = torch.empty((capacity, 1), device=device, dtype=torch.long)
        self.insertion_steps = torch.empty((capacity, 1), device=device, dtype=torch.long)
        self.usage_counts = torch.zeros((capacity, 1), device=device, dtype=torch.long)
        self._fresh = torch.zeros((capacity, 1), device=device, dtype=torch.bool)
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
        insertion_step: int,
    ) -> int:
        index = self._position
        self.observations[index].copy_(observation.to(self.device))
        self.actions[index].copy_(action.to(self.device))
        self.rewards[index, 0] = torch.as_tensor(
            reward,
            device=self.device,
            dtype=self.rewards.dtype,
        )
        self.next_observations[index].copy_(next_observation.to(self.device))
        self.terminated[index, 0] = terminated
        self.truncated[index, 0] = truncated
        transition_id = self._next_transition_id
        self.transition_ids[index, 0] = transition_id
        self.insertion_steps[index, 0] = insertion_step
        self.usage_counts[index, 0] = 0
        self._fresh[index, 0] = True
        self._next_transition_id += 1
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return transition_id

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> ReplayBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._size < batch_size:
            raise ValueError("not enough replay items to sample requested batch")
        indices = torch.randperm(
            self._size,
            device=self.device,
            generator=generator,
        )[:batch_size]
        prior_usage = self.usage_counts[indices].clone()
        fresh = self._fresh[indices].clone()
        self.usage_counts[indices] += 1
        self._fresh[indices] = False
        return ReplayBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            terminated=self.terminated[indices],
            truncated=self.truncated[indices],
            transition_ids=self.transition_ids[indices],
            insertion_steps=self.insertion_steps[indices],
            usage_counts=prior_usage + 1,
            fresh=fresh,
        )

    def state_dict(self) -> dict[str, object]:
        size = self._size
        return {
            "version": 1,
            "capacity": self.capacity,
            "size": size,
            "position": self._position,
            "next_transition_id": self._next_transition_id,
            "observations": self.observations[:size].clone(),
            "actions": self.actions[:size].clone(),
            "rewards": self.rewards[:size].clone(),
            "next_observations": self.next_observations[:size].clone(),
            "terminated": self.terminated[:size].clone(),
            "truncated": self.truncated[:size].clone(),
            "transition_ids": self.transition_ids[:size].clone(),
            "insertion_steps": self.insertion_steps[:size].clone(),
            "usage_counts": self.usage_counts[:size].clone(),
            "fresh": self._fresh[:size].clone(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if _checkpoint_int(state.get("version"), name="replay version") != 1:
            raise ValueError("unsupported replay checkpoint version")
        if (
            _checkpoint_int(
                state["capacity"],
                name="replay capacity",
            )
            != self.capacity
        ):
            raise ValueError("replay checkpoint capacity mismatch")
        size = _checkpoint_int(
            state["size"],
            name="replay size",
        )
        if not 0 <= size <= self.capacity:
            raise ValueError("invalid replay checkpoint size")
        tensor_fields = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "transition_ids": self.transition_ids,
            "insertion_steps": self.insertion_steps,
            "usage_counts": self.usage_counts,
            "fresh": self._fresh,
        }
        checked_tensors: dict[str, Tensor] = {}
        for key, target in tensor_fields.items():
            source = state[key]
            if not isinstance(source, Tensor):
                raise TypeError(f"replay checkpoint field {key} must be a tensor")
            if source.shape != target[:size].shape:
                raise ValueError(f"replay checkpoint shape mismatch for {key}")
            if source.dtype != target.dtype:
                raise ValueError(f"replay checkpoint dtype mismatch for {key}")
            if source.is_floating_point() and not torch.isfinite(source).all().item():
                raise ValueError("replay checkpoint contains non-finite values for " + key)
            checked_tensors[key] = source
        position = _checkpoint_int(
            state["position"],
            name="replay position",
        )
        if not 0 <= position < self.capacity:
            raise ValueError("invalid replay checkpoint position")
        if size < self.capacity and position != size:
            raise ValueError("replay checkpoint position is inconsistent with size")

        next_transition_id = _checkpoint_int(
            state["next_transition_id"],
            name="next transition id",
        )
        if next_transition_id < 0:
            raise ValueError("next transition id must be non-negative")

        usage = checked_tensors["usage_counts"]
        fresh = checked_tensors["fresh"]
        if torch.any(usage < 0):
            raise ValueError("replay checkpoint contains negative usage counts")
        if not torch.equal(
            fresh,
            usage == 0,
        ):
            raise ValueError("replay checkpoint freshness disagrees with usage counts")

        if size:
            transition_ids = checked_tensors["transition_ids"][:, 0]
            if torch.any(transition_ids < 0):
                raise ValueError("replay checkpoint contains negative transition IDs")
            if torch.unique(transition_ids).numel() != size:
                raise ValueError("replay checkpoint contains duplicate transition IDs")
            expected_next = int(transition_ids.max().item()) + 1
        else:
            expected_next = 0
        if next_transition_id != expected_next:
            raise ValueError("replay checkpoint next transition ID is inconsistent")
        if size:
            oldest = (position - size) % self.capacity
            logical = (oldest + torch.arange(size, device=transition_ids.device)) % self.capacity
            expected_ids = torch.arange(
                next_transition_id - size,
                next_transition_id,
                device=transition_ids.device,
                dtype=transition_ids.dtype,
            )
            if not torch.equal(checked_tensors["transition_ids"][logical, 0], expected_ids):
                raise ValueError("replay checkpoint chronological transition IDs are inconsistent")

        for key, target in tensor_fields.items():
            target[:size].copy_(checked_tensors[key].to(device=self.device))

        self._size = size
        self._position = position
        self._next_transition_id = next_transition_id
