"""Posterior diagnostics used by logs and mechanistic analyses."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor


def _flatten(mapping: Mapping[str, Tensor]) -> Tensor:
    if not mapping:
        raise ValueError("cannot summarize an empty mapping")
    return torch.cat([value.detach().float().reshape(-1) for value in mapping.values()])


def posterior_diagnostics(
    stds: Mapping[str, Tensor],
    *,
    eta: float,
    evidence_temperature: float = 1.0,
    collapse_threshold: float = 1e-4,
) -> dict[str, float]:
    if eta <= 0 or evidence_temperature <= 0:
        raise ValueError("eta and evidence_temperature must be strictly positive")
    sigma = _flatten(stds)
    if torch.any(sigma <= 0) or not torch.isfinite(sigma).all():
        raise FloatingPointError("posterior standard deviations must be finite and positive")
    precision = sigma.reciprocal().square()
    effective_lr = eta * evidence_temperature * sigma.square()
    entropy = 0.5 * torch.log(2.0 * math.pi * math.e * sigma.square())
    quantiles = torch.quantile(sigma, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    lr_quantiles = torch.quantile(effective_lr, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "sigma_mean": sigma.mean().item(),
        "sigma_median": sigma.median().item(),
        "sigma_std": sigma.std(unbiased=False).item(),
        "sigma_min": sigma.min().item(),
        "sigma_max": sigma.max().item(),
        "sigma_q05": quantiles[0].item(),
        "sigma_q25": quantiles[1].item(),
        "sigma_q50": quantiles[2].item(),
        "sigma_q75": quantiles[3].item(),
        "sigma_q95": quantiles[4].item(),
        "fraction_sigma_below_threshold": (sigma < collapse_threshold).float().mean().item(),
        "posterior_entropy_mean": entropy.mean().item(),
        "precision_mean": precision.mean().item(),
        "effective_lr_mean": effective_lr.mean().item(),
        "effective_lr_q05": lr_quantiles[0].item(),
        "effective_lr_q50": lr_quantiles[2].item(),
        "effective_lr_q95": lr_quantiles[4].item(),
    }


def layerwise_posterior_diagnostics(
    stds: Mapping[str, Tensor],
    *,
    eta: float,
    evidence_temperature: float = 1.0,
) -> dict[str, dict[str, float]]:
    return {
        name: posterior_diagnostics(
            {name: std},
            eta=eta,
            evidence_temperature=evidence_temperature,
        )
        for name, std in stds.items()
    }
