"""Deterministic capacity and temporary-allocation estimates for sequence replay.

All byte counts describe payload tensors, excluding PyTorch allocator overhead,
model parameters, GRU activations, optimizer state and simulator memory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from rl_bgd.utils.config_validation import (
    config_nonnegative_integer,
    config_positive_integer,
)


@dataclass(frozen=True)
class SequenceReplayMemoryEstimate:
    capacity: int
    observation_dim: int
    action_dim: int
    dtype_bytes: int
    float_storage_bytes: int
    integer_storage_bytes: int
    boolean_storage_bytes: int
    total_storage_bytes: int
    sampled_batch_bytes: int
    sampling_index_bytes_upper_bound: int
    checkpoint_tensor_bytes_upper_bound: int
    restore_peak_tensor_bytes_lower_bound: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_sequence_replay_memory(
    capacity: int,
    observation_dim: int,
    action_dim: int,
    *,
    batch_size: int,
    burn_in: int,
    unroll: int,
    dtype: torch.dtype = torch.float32,
) -> SequenceReplayMemoryEstimate:
    """Estimate exact persistent field bytes and bounded replay-only temporaries.

    The temporary estimate includes the copied sampled fields, chronological
    index construction, and usage/freshness labels, but not downstream model
    activations. Checkpoints clone occupied replay entries only; the reported
    checkpoint bound assumes full capacity.
    """
    capacity = config_positive_integer(capacity, name="replay capacity")
    observation_dim = config_positive_integer(observation_dim, name="observation_dim")
    action_dim = config_positive_integer(action_dim, name="action_dim")
    batch_size = config_positive_integer(batch_size, name="sequence batch_size")
    burn_in = config_nonnegative_integer(burn_in, name="burn_in")
    unroll = config_positive_integer(unroll, name="unroll")
    if capacity < burn_in + unroll:
        raise ValueError("replay capacity cannot hold one requested window")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("replay dtype must be a floating-point torch.dtype")
    floating_bytes = torch.empty((), dtype=dtype).element_size()
    integer_bytes = torch.empty((), dtype=torch.long).element_size()
    boolean_bytes = torch.empty((), dtype=torch.bool).element_size()

    # Observations, actions, reward, next observations.
    float_per_transition = (2 * observation_dim + action_dim + 1) * floating_bytes
    # Transition ID, insertion step and evidence-use count.
    integer_per_transition = 3 * integer_bytes
    # Termination, truncation, episode-start and freshness indicators.
    boolean_per_transition = 4 * boolean_bytes
    float_storage = capacity * float_per_transition
    integer_storage = capacity * integer_per_transition
    boolean_storage = capacity * boolean_per_transition
    total = float_storage + integer_storage + boolean_storage

    window = burn_in + unroll
    # Sampled full-window fields contain two integer labels and three masks.
    copied_window_bytes = batch_size * window * (
        float_per_transition + 2 * integer_bytes + 3 * boolean_bytes
    )
    # Evidence-count and fresh flags are sampled for unroll positions only.
    sampled_batch = copied_window_bytes + batch_size * unroll * (
        integer_bytes + boolean_bytes
    )
    # Chronological logical indices, chosen starts, physical windows and
    # unroll indices; overstates simultaneous storage if views are reused.
    sampled_indices = integer_bytes * (
        capacity + batch_size + 2 * batch_size * window + batch_size * unroll
    )
    return SequenceReplayMemoryEstimate(
        capacity=capacity,
        observation_dim=observation_dim,
        action_dim=action_dim,
        dtype_bytes=floating_bytes,
        float_storage_bytes=float_storage,
        integer_storage_bytes=integer_storage,
        boolean_storage_bytes=boolean_storage,
        total_storage_bytes=total,
        sampled_batch_bytes=sampled_batch,
        sampling_index_bytes_upper_bound=sampled_indices,
        checkpoint_tensor_bytes_upper_bound=total,
        restore_peak_tensor_bytes_lower_bound=2 * total,
    )
