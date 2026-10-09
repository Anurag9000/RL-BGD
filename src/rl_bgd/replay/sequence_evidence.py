"""Bayesian evidence accounting for recurrent sequence replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_bgd.replay.evidence_accounting import (
    ReplayEvidenceConfig,
    _validated_replay_metadata,
)
from rl_bgd.replay.sequence_buffer import (
    SequenceReplayBatch,
)


@dataclass(frozen=True)
class SequenceReplayEvidenceSummary:
    """Unroll-position evidence weights and diagnostics."""

    weights: Tensor
    mean_weight: float
    min_weight: float
    max_weight: float
    fresh_fraction: float
    mean_usage_count: float
    effective_sample_size: float


def _effective_sample_size(
    weights: Tensor,
) -> float:
    flat = weights.detach().float().reshape(-1)
    numerator = flat.sum().square()
    denominator = flat.square().sum()
    if denominator.item() == 0.0:
        return 0.0
    return float((numerator / denominator).item())


def sequence_replay_evidence_weights(
    batch: SequenceReplayBatch,
    config: ReplayEvidenceConfig,
) -> SequenceReplayEvidenceSummary:
    """Construct uncertainty weights for optimized sequence positions only."""

    config.validate()
    usage, fresh = _validated_replay_metadata(
        batch.usage_counts,
        batch.fresh,
        ndim=3,
        label="sequence",
    )

    inverse_usage = usage.reciprocal()
    if config.mode == "all_replay":
        weights = torch.ones_like(usage)
    elif config.mode == "fresh_only_uncertainty":
        weights = fresh.float()
    elif config.mode == "inverse_reuse_weight":
        weights = inverse_usage
    else:
        scale = inverse_usage.mean()
        weights = torch.ones_like(usage) * scale
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise FloatingPointError("invalid sequence replay evidence weights")
    return SequenceReplayEvidenceSummary(
        weights=weights,
        mean_weight=float(weights.mean().item()),
        min_weight=float(weights.min().item()),
        max_weight=float(weights.max().item()),
        fresh_fraction=float(fresh.float().mean().item()),
        mean_usage_count=float(usage.mean().item()),
        effective_sample_size=(_effective_sample_size(weights)),
    )


def weighted_sequence_evidence_mean(
    per_transition_loss: Tensor,
    weights: Tensor,
) -> Tensor:
    """Reduce unroll losses without renormalizing away evidence mass."""

    if per_transition_loss.ndim < 2:
        raise ValueError("sequence loss must retain batch and time dimensions")
    if weights.ndim != 3 or weights.shape[-1] != 1:
        raise ValueError("sequence evidence weights must have shape [batch, time, 1]")
    if per_transition_loss.shape[:2] != weights.shape[:2]:
        raise ValueError("sequence loss/evidence batch-time shapes differ")
    if per_transition_loss.numel() == 0 or weights.numel() == 0:
        raise ValueError("sequence evidence reduction requires non-empty input")
    if not torch.isfinite(per_transition_loss).all():
        raise FloatingPointError("non-finite sequence evidence losses")
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise FloatingPointError("invalid sequence evidence reduction weights")
    flattened = per_transition_loss.reshape(
        *per_transition_loss.shape[:2],
        -1,
    )
    per_transition = flattened.mean(
        dim=-1,
        keepdim=True,
    )
    return (per_transition * weights.to(per_transition)).mean()
