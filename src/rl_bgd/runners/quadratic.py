"""Executable synthetic continual-quadratic experiment."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import (
    DiagonalGaussianPosterior,
    PosteriorBounds,
)
from rl_bgd.envs.synthetic.quadratic import (
    QuadraticStream,
    diagonal_quadratic,
)
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


class VectorParameter(nn.Module):
    def __init__(self, dimension: int, *, device: torch.device) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(dimension, device=device))


@dataclass(frozen=True)
class QuadraticRunConfig:
    dimension: int = 20
    segment_steps: int = 50
    segments: int = 3
    eta: float = 0.2
    prior_std: float = 0.5
    mc_samples: int = 8
    temper_retention: float = 1.0
    seed: int = 0
    device: str = "auto"


def run_quadratic(
    config: QuadraticRunConfig,
) -> dict[str, object]:
    if config.segments < 2:
        raise ValueError("segments must be >= 2")
    seed_everything(config.seed, deterministic=True)
    device = resolve_device(config.device)
    optima = [(-1.0 if i % 2 else 1.0) for i in range(config.segments)]
    tasks = [
        diagonal_quadratic(
            config.dimension,
            optimum=value,
            curvature=1.0 + 0.25 * i,
            name=f"task_{i}",
            device=device,
        )
        for i, value in enumerate(optima)
    ]
    stream = QuadraticStream(
        tasks,
        segment_steps=config.segment_steps,
        mode="abrupt",
    )
    model = VectorParameter(config.dimension, device=device)
    posterior = DiagonalGaussianPosterior.from_module(
        model,
        prior_std=config.prior_std,
        bounds=PosteriorBounds(sigma_min=1e-6, sigma_max=5.0),
    )
    updater = BGDUpdater(
        posterior,
        BGDConfig(
            eta=config.eta,
            mc_samples=config.mc_samples,
            antithetic=config.mc_samples % 2 == 0,
            temper_retention=config.temper_retention,
        ),
    )
    total_steps = config.segment_steps * config.segments
    losses: list[float] = []
    boundaries: list[dict[str, float | int]] = []
    result = None
    for step in range(total_steps):
        active_task = stream.task_at(step)

        def objective(
            params: Mapping[str, torch.Tensor],
        ) -> torch.Tensor:
            return active_task.loss(params["theta"])

        result = updater.step(objective)
        losses.append(result.mean_loss)
        if stream.is_evaluation_boundary(step):
            boundaries.append(
                {
                    "step": step,
                    "sigma_mean": result.diagnostics["sigma_mean"],
                    "loss": result.mean_loss,
                }
            )
    assert result is not None
    posterior.sync_module(model)
    return {
        "dimension": config.dimension,
        "total_steps": total_steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "sigma_mean": result.diagnostics["sigma_mean"],
        "effective_lr_mean": result.diagnostics["effective_lr_mean"],
        "boundaries": boundaries,
    }


def main() -> None:
    print(
        json.dumps(
            run_quadratic(QuadraticRunConfig()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
