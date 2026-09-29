"""Controlled posterior forgetting for diagonal Gaussians."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def _validate_lambda(retention: float) -> None:
    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must lie in [0, 1]")


def temper_gaussian_tensor(
    mean: Tensor,
    std: Tensor,
    prior_mean: Tensor,
    prior_std: Tensor,
    *,
    retention: float,
) -> tuple[Tensor, Tensor]:
    """Return normalized q_prev^retention * p0^(1-retention)."""
    _validate_lambda(retention)
    values = (mean, std, prior_mean, prior_std)
    if any(x.shape != mean.shape for x in values):
        raise ValueError("mean/std/prior tensors must have matching shapes")
    if torch.any(std <= 0) or torch.any(prior_std <= 0):
        raise ValueError("standard deviations must be strictly positive")
    mean32, std32, pmean32, pstd32 = [x.float() for x in values]
    tau_prev = std32.reciprocal().square()
    tau0 = pstd32.reciprocal().square()
    tau = retention * tau_prev + (1.0 - retention) * tau0
    tempered_mean = (
        retention * tau_prev * mean32 + (1.0 - retention) * tau0 * pmean32
    ) / tau
    tempered_std = torch.rsqrt(tau)
    return tempered_mean, tempered_std


def temper_diagonal_gaussian(
    means: Mapping[str, Tensor],
    stds: Mapping[str, Tensor],
    prior_means: Mapping[str, Tensor],
    prior_stds: Mapping[str, Tensor],
    *,
    retention: float,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    keys = set(means)
    if any(set(mapping) != keys for mapping in (stds, prior_means, prior_stds)):
        raise ValueError("all posterior mappings must share keys")
    new_means: dict[str, Tensor] = {}
    new_stds: dict[str, Tensor] = {}
    for name in means:
        new_means[name], new_stds[name] = temper_gaussian_tensor(
            means[name],
            stds[name],
            prior_means[name],
            prior_stds[name],
            retention=retention,
        )
    return new_means, new_stds
