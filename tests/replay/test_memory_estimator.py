"""Deterministic replay memory accounting with no large allocations."""

from __future__ import annotations

import pytest
import torch

from rl_bgd.replay.memory import estimate_sequence_replay_memory


def test_replay_memory_estimate_matches_exact_field_shapes() -> None:
    estimate = estimate_sequence_replay_memory(10, 2, 1, batch_size=2, burn_in=1, unroll=2)
    assert estimate.dtype_bytes == 4
    assert estimate.float_storage_bytes == 10 * 24
    assert estimate.integer_storage_bytes == 10 * 24
    assert estimate.boolean_storage_bytes == 10 * 4
    assert estimate.total_storage_bytes == 520
    assert estimate.sampled_batch_bytes == 294
    assert estimate.checkpoint_tensor_bytes_upper_bound == 520
    assert estimate.restore_peak_tensor_bytes_lower_bound == 1040
    assert estimate.to_dict()["capacity"] == 10


def test_large_replay_capacity_is_estimated_without_allocating_it() -> None:
    estimate = estimate_sequence_replay_memory(
        10_000_000, 45, 4, batch_size=128, burn_in=15, unroll=1
    )
    bytes_per_transition = 4 * (2 * 45 + 4 + 1) + 3 * 8 + 4
    assert estimate.total_storage_bytes == 10_000_000 * bytes_per_transition
    assert estimate.total_storage_bytes > 4_000_000_000
    assert estimate.sampled_batch_bytes > 0


def test_replay_memory_estimator_accounts_for_float64_storage() -> None:
    single = estimate_sequence_replay_memory(
        8, 2, 1, batch_size=1, burn_in=0, unroll=1, dtype=torch.float32
    )
    double = estimate_sequence_replay_memory(
        8, 2, 1, batch_size=1, burn_in=0, unroll=1, dtype=torch.float64
    )
    assert double.float_storage_bytes == 2 * single.float_storage_bytes
    assert double.integer_storage_bytes == single.integer_storage_bytes
    assert double.boolean_storage_bytes == single.boolean_storage_bytes


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_replay_estimator_rejects_invalid_capacity(bad: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        estimate_sequence_replay_memory(
            bad,
            2,
            1,
            batch_size=1,
            burn_in=0,
            unroll=1,  # type: ignore[arg-type]
        )


def test_replay_estimator_rejects_window_over_capacity() -> None:
    with pytest.raises(ValueError, match="cannot hold one"):
        estimate_sequence_replay_memory(3, 2, 1, batch_size=1, burn_in=2, unroll=2)


@pytest.mark.parametrize("dtype", [torch.long, torch.bool])
def test_replay_estimator_rejects_nonfloat_dtype(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="floating-point"):
        estimate_sequence_replay_memory(8, 2, 1, batch_size=1, burn_in=0, unroll=1, dtype=dtype)
