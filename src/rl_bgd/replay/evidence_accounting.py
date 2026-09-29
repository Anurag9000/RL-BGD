"""Replay-evidence accounting for Bayesian uncertainty updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from rl_bgd.replay.buffer import ReplayBatch

ReplayEvidenceMode = Literal[
    "all_replay",
    "fresh_only_uncertainty",
    "inverse_reuse_weight",
    "normalized_batch_evidence",
]


@dataclass(frozen=True)
class ReplayEvidenceConfig:
    """Controls how replay contributes to posterior uncertainty.

    The ordinary RL objective still trains posterior means on the sampled batch.
    These modes only change the uncertainty-gradient channel, isolating evidence
    reuse from changes to the replay optimization distribution.
    """

    mode: ReplayEvidenceMode = "all_replay"

    def validate(self) -> None:
        if self.mode not in {
            "all_replay",
            "fresh_only_uncertainty",
            "inverse_reuse_weight",
            "normalized_batch_evidence",
        }:
            raise ValueError(f"unsupported replay evidence mode: {self.mode}")


@dataclass(frozen=True)
class ReplayEvidenceSummary:
    """Weights and diagnostics for one replay batch."""

    mode: ReplayEvidenceMode
    weights: Tensor
    mean_weight: float
    min_weight: float
    max_weight: float
    fresh_fraction: float
    mean_usage_count: float
    effective_sample_size: float


def _effective_sample_size(weights: Tensor) -> float:
    flat = weights.detach().float().reshape(-1)
    numerator = flat.sum().square()
    denominator = flat.square().sum()
    if denominator.item() == 0.0:
        return 0.0
    return float((numerator / denominator).item())


def replay_evidence_weights(
    batch: ReplayBatch,
    config: ReplayEvidenceConfig,
) -> ReplayEvidenceSummary:
    """Construct uncertainty-evidence weights from replay metadata.

    Semantics:
    - all_replay: every replay use has unit evidence weight.
    - fresh_only_uncertainty: only a transition's first sampled use changes sigma.
    - inverse_reuse_weight: use j contributes 1 / usage_count_j.
    - normalized_batch_evidence: preserve within-batch composition but scale the
      entire uncertainty update by the batch mean inverse reuse count.

    Losses are later reduced as mean(weight * per_item_loss), deliberately not
    divided by sum(weights). Therefore lower weights represent less Bayesian
    evidence instead of merely reweighting the same total evidence mass.
    """

    config.validate()
    usage = batch.usage_counts.detach().float()
    if usage.ndim != 2 or usage.shape[1] != 1:
        raise ValueError("usage_counts must have shape [batch, 1]")
    if torch.any(usage < 1):
        raise ValueError("usage_counts must be >= 1 for sampled replay items")
    fresh = batch.fresh.detach()
    if fresh.shape != usage.shape:
        raise ValueError("fresh metadata must match usage_counts shape")

    inverse_usage = usage.reciprocal()
    if config.mode == "all_replay":
        weights = torch.ones_like(usage)
    elif config.mode == "fresh_only_uncertainty":
        weights = fresh.float()
    elif config.mode == "inverse_reuse_weight":
        weights = inverse_usage
    else:
        batch_scale = inverse_usage.mean()
        weights = torch.ones_like(usage) * batch_scale

    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise FloatingPointError("invalid replay evidence weights")

    return ReplayEvidenceSummary(
        mode=config.mode,
        weights=weights,
        mean_weight=float(weights.mean().item()),
        min_weight=float(weights.min().item()),
        max_weight=float(weights.max().item()),
        fresh_fraction=float(fresh.float().mean().item()),
        mean_usage_count=float(usage.mean().item()),
        effective_sample_size=_effective_sample_size(weights),
    )


def weighted_evidence_mean(per_item_loss: Tensor, weights: Tensor) -> Tensor:
    """Reduce per-transition losses without renormalizing evidence mass."""

    if per_item_loss.ndim == 0:
        raise ValueError("per_item_loss must retain a batch dimension")
    if weights.ndim != 2 or weights.shape[1] != 1:
        raise ValueError("weights must have shape [batch, 1]")
    if per_item_loss.shape[0] != weights.shape[0]:
        raise ValueError("loss and evidence weights have different batch sizes")
    flattened = per_item_loss.reshape(per_item_loss.shape[0], -1)
    per_transition = flattened.mean(dim=1, keepdim=True)
    return (per_transition * weights.to(per_transition)).mean()
