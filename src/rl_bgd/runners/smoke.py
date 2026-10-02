"""Dependency-light quadratic BGD smoke run."""

from __future__ import annotations

import json
from collections.abc import Mapping

import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import (
    DiagonalGaussianPosterior,
    PosteriorBounds,
)
from rl_bgd.utils.device import get_device_info, resolve_device
from rl_bgd.utils.randomness import seed_everything


class ScalarParameter(nn.Module):
    def __init__(self, initial: float = 2.0) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([initial], dtype=torch.float32))


def run_smoke(*, steps: int = 25, seed: int = 0, device: str = "auto") -> dict[str, object]:
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    model = ScalarParameter().to(resolved)
    posterior = DiagonalGaussianPosterior.from_module(
        model,
        prior_std=0.5,
        bounds=PosteriorBounds(sigma_min=1e-5, sigma_max=2.0),
    )
    updater = BGDUpdater(posterior, BGDConfig(eta=0.25, mc_samples=8, antithetic=True))
    target = torch.tensor([0.0], device=resolved)

    def objective(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return 0.5 * (params["theta"] - target).square().sum()

    first_abs_mean = float(posterior.means["theta"].abs().item())
    last = None
    for _ in range(steps):
        last = updater.step(objective)
    posterior.sync_module(model)
    final_abs_mean = float(posterior.means["theta"].abs().item())
    if not final_abs_mean < first_abs_mean:
        raise RuntimeError("quadratic smoke run did not reduce absolute posterior mean")
    assert last is not None
    return {
        "device": get_device_info(device).as_dict(),
        "steps": steps,
        "initial_abs_mean": first_abs_mean,
        "final_abs_mean": final_abs_mean,
        "diagnostics": last.diagnostics,
    }


def main() -> None:
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
