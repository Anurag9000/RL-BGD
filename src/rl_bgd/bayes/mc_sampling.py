"""Monte Carlo helpers for Bayesian parameter updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def aggregate_bgd_statistics(
    gradients: Sequence[Mapping[str, Tensor]],
    epsilons: Sequence[Mapping[str, Tensor]],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Compute g_bar and c = E[g * epsilon] in FP32."""
    if not gradients or len(gradients) != len(epsilons):
        raise ValueError("gradients and epsilons must be non-empty and have equal length")
    keys = set(gradients[0])
    if any(set(item) != keys for item in gradients) or any(
        set(item) != keys for item in epsilons
    ):
        raise ValueError("all gradient/epsilon dictionaries must share keys")
    g_bar: dict[str, Tensor] = {}
    c: dict[str, Tensor] = {}
    for name in keys:
        stacked_g = torch.stack([item[name].detach().float() for item in gradients])
        stacked_eps = torch.stack([item[name].detach().float() for item in epsilons])
        g_bar[name] = stacked_g.mean(dim=0)
        c[name] = (stacked_g * stacked_eps).mean(dim=0)
    return g_bar, c
