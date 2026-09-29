"""Monte Carlo helpers for Bayesian parameter updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def aggregate_bgd_statistics(
    gradients: Sequence[Mapping[str, Tensor]],
    epsilons: Sequence[Mapping[str, Tensor]],
    *,
    uncertainty_gradients: Sequence[Mapping[str, Tensor]] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Compute g_bar and c = E[g_uncertainty * epsilon] in FP32."""

    if not gradients or len(gradients) != len(epsilons):
        raise ValueError("gradients and epsilons must be non-empty and have equal length")
    evidence_gradients = gradients if uncertainty_gradients is None else uncertainty_gradients
    if len(evidence_gradients) != len(epsilons):
        raise ValueError("uncertainty gradients must match the Monte Carlo sample count")
    keys = set(gradients[0])
    collections = (gradients, epsilons, evidence_gradients)
    if any(any(set(item) != keys for item in collection) for collection in collections):
        raise ValueError("all gradient/epsilon dictionaries must share keys")

    g_bar: dict[str, Tensor] = {}
    c: dict[str, Tensor] = {}
    for name in keys:
        stacked_g = torch.stack([item[name].detach().float() for item in gradients])
        stacked_evidence_g = torch.stack(
            [item[name].detach().float() for item in evidence_gradients]
        )
        stacked_eps = torch.stack([item[name].detach().float() for item in epsilons])
        g_bar[name] = stacked_g.mean(dim=0)
        c[name] = (stacked_evidence_g * stacked_eps).mean(dim=0)
    return g_bar, c
