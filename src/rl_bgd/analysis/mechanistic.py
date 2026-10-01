"""Reproducible mechanistic BGD analyses and paper figures."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import (
    DiagonalGaussianPosterior,
    PosteriorBounds,
)
from rl_bgd.envs.synthetic.quadratic import diagonal_quadratic
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


class MechanisticVector(nn.Module):
    def __init__(
        self,
        dimension: int,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.theta = nn.Parameter(
            torch.zeros(
                dimension,
                device=device,
                dtype=torch.float32,
            )
        )


@dataclass(frozen=True)
class MechanisticAnalysisConfig:
    dimension: int = 24
    consolidation_steps: int = 40
    adaptation_steps: int = 16
    mc_samples: int = 64
    curvature_samples: int = 512
    eta: float = 0.15
    prior_std: float = 0.5
    perturbation_size: float = 0.25
    seed: int = 150
    device: str = "auto"

    def validate(self) -> None:
        if self.dimension < 8:
            raise ValueError("mechanistic analysis requires dimension >= 8")
        if (
            min(
                self.consolidation_steps,
                self.adaptation_steps,
                self.mc_samples,
                self.curvature_samples,
            )
            < 1
        ):
            raise ValueError("mechanistic analysis budgets must be positive")
        if self.eta <= 0 or self.prior_std <= 0:
            raise ValueError("mechanistic BGD scales must be positive")
        if self.perturbation_size <= 0:
            raise ValueError("perturbation_size must be positive")
        if self.mc_samples % 2:
            raise ValueError("mechanistic mc_samples must be even")
        if self.curvature_samples % 2:
            raise ValueError("curvature_samples must be even")


def _rank(values: Tensor) -> Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(
        values,
        dtype=torch.float32,
    )
    ranks[order] = torch.arange(
        values.numel(),
        device=values.device,
        dtype=torch.float32,
    )
    return ranks


def _pearson(left: Tensor, right: Tensor) -> float:
    x = left.float().reshape(-1)
    y = right.float().reshape(-1)
    if x.numel() != y.numel() or x.numel() < 2:
        raise ValueError("correlation vectors must align and contain >=2 values")
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((x * y).sum().div(denominator).item())


def _spearman(left: Tensor, right: Tensor) -> float:
    return _pearson(_rank(left), _rank(right))


def _new_posterior(
    config: MechanisticAnalysisConfig,
    *,
    device: torch.device,
) -> tuple[MechanisticVector, DiagonalGaussianPosterior, BGDUpdater]:
    model = MechanisticVector(
        config.dimension,
        device=device,
    )
    posterior = DiagonalGaussianPosterior.from_module(
        model,
        prior_std=config.prior_std,
        bounds=PosteriorBounds(
            sigma_min=1e-7,
            sigma_max=5.0,
        ),
    )
    updater = BGDUpdater(
        posterior,
        BGDConfig(
            eta=config.eta,
            mc_samples=config.mc_samples,
            antithetic=True,
        ),
    )
    return model, posterior, updater


def _loss_objective(task: object):
    def objective(params: dict[str, Tensor]) -> Tensor:
        return task.loss(params["theta"])

    return objective


def _clone_updater(
    state: dict[str, object],
    config: MechanisticAnalysisConfig,
    *,
    device: torch.device,
) -> BGDUpdater:
    _, _, updater = _new_posterior(
        config,
        device=device,
    )
    updater.load_state_dict(state)
    return updater


def _adapt_with_freeze(
    initial_state: dict[str, object],
    config: MechanisticAnalysisConfig,
    target: object,
    frozen_indices: Tensor | None,
    *,
    device: torch.device,
    generator_seed: int,
) -> float:
    updater = _clone_updater(
        initial_state,
        config,
        device=device,
    )
    frozen_mean: Tensor | None = None
    frozen_sigma: Tensor | None = None
    if frozen_indices is not None:
        frozen_mean = updater.posterior.means["theta"][frozen_indices].clone()
        frozen_sigma = updater.posterior.stds["theta"][frozen_indices].clone()

    generator = torch.Generator(device=device).manual_seed(generator_seed)
    objective = _loss_objective(target)
    for _ in range(config.adaptation_steps):
        updater.step(
            objective,
            generator=generator,
        )
        if frozen_indices is not None:
            assert frozen_mean is not None
            assert frozen_sigma is not None
            updater.posterior.means["theta"][frozen_indices] = frozen_mean
            updater.posterior.stds["theta"][frozen_indices] = frozen_sigma

    return float(target.loss(updater.posterior.means["theta"]).item())


def _save_parameter_csv(
    path: Path,
    *,
    curvature: Tensor,
    sigma: Tensor,
    precision: Tensor,
    movement: Tensor,
    perturbation_importance: Tensor,
    curvature_signal: Tensor,
    curvature_expected: Tensor,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dimension",
                "curvature",
                "sigma",
                "precision",
                "movement",
                "perturbation_importance",
                "curvature_signal",
                "curvature_expected",
            ]
        )
        for index in range(curvature.numel()):
            writer.writerow(
                [
                    index,
                    float(curvature[index].item()),
                    float(sigma[index].item()),
                    float(precision[index].item()),
                    float(movement[index].item()),
                    float(perturbation_importance[index].item()),
                    float(curvature_signal[index].item()),
                    float(curvature_expected[index].item()),
                ]
            )


def _save_figures(
    output_dir: Path,
    *,
    sigma: Tensor,
    precision: Tensor,
    movement: Tensor,
    perturbation_importance: Tensor,
    curvature_signal: Tensor,
    curvature_expected: Tensor,
    freezing: dict[str, float],
) -> list[str]:
    artifacts: list[str] = []

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.scatter(
        sigma.cpu().numpy(),
        movement.cpu().numpy(),
    )
    axis.set_xlabel("posterior sigma before shift")
    axis.set_ylabel("absolute mean movement after shift")
    axis.set_title("Posterior uncertainty vs parameter movement")
    path = output_dir / "movement_vs_sigma.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    artifacts.append(path.name)

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.scatter(
        precision.cpu().numpy(),
        perturbation_importance.cpu().numpy(),
    )
    axis.set_xlabel("posterior precision")
    axis.set_ylabel("loss increase under fixed perturbation")
    axis.set_title("Posterior precision vs perturbation importance")
    path = output_dir / "perturbation_vs_precision.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    artifacts.append(path.name)

    figure = plt.figure()
    axis = figure.add_subplot(111)
    labels = list(freezing)
    values = [freezing[label] for label in labels]
    axis.bar(labels, values)
    axis.set_ylabel("post-adaptation target loss")
    axis.set_title("Causal freezing counterfactual")
    path = output_dir / "freezing_counterfactual.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    artifacts.append(path.name)

    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.scatter(
        curvature_expected.cpu().numpy(),
        curvature_signal.cpu().numpy(),
    )
    low = float(
        torch.minimum(
            curvature_expected.min(),
            curvature_signal.min(),
        ).item()
    )
    high = float(
        torch.maximum(
            curvature_expected.max(),
            curvature_signal.max(),
        ).item()
    )
    axis.plot([low, high], [low, high])
    axis.set_xlabel("analytic H sigma")
    axis.set_ylabel("Monte Carlo E[g epsilon]")
    axis.set_title("BGD curvature signal validation")
    path = output_dir / "curvature_signal.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    artifacts.append(path.name)

    sorted_indices = torch.argsort(sigma)
    chunks = torch.tensor_split(sorted_indices, 4)
    movement_bins = torch.tensor([movement[index].mean() for index in chunks if index.numel() > 0])
    perturbation_bins = torch.tensor(
        [perturbation_importance[index].mean() for index in chunks if index.numel() > 0]
    )
    movement_norm = movement_bins / movement_bins.max().clamp_min(1e-12)
    perturbation_norm = perturbation_bins / perturbation_bins.max().clamp_min(1e-12)
    figure = plt.figure()
    axis = figure.add_subplot(111)
    x = list(range(1, len(movement_norm) + 1))
    axis.plot(
        x,
        movement_norm.cpu().numpy(),
        marker="o",
        label="future movement",
    )
    axis.plot(
        x,
        perturbation_norm.cpu().numpy(),
        marker="o",
        label="perturbation importance",
    )
    axis.set_xlabel("sigma quartile: low to high")
    axis.set_ylabel("within-metric normalized mean")
    axis.set_title("Uncertainty-quality diagnostic")
    axis.legend()
    path = output_dir / "uncertainty_quality.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    artifacts.append(path.name)

    return artifacts


def run_mechanistic_analysis(
    output_dir: str | Path,
    *,
    config: MechanisticAnalysisConfig | None = None,
) -> dict[str, object]:
    """Run all Phase-13 mechanistic analyses and write reproducible artifacts."""

    resolved_config = config or MechanisticAnalysisConfig()
    resolved_config.validate()
    seed_everything(
        resolved_config.seed,
        deterministic=True,
    )
    device = resolve_device(resolved_config.device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    curvature = torch.logspace(
        -0.5,
        0.9,
        resolved_config.dimension,
        device=device,
    )
    consolidation_task = diagonal_quadratic(
        resolved_config.dimension,
        optimum=0.0,
        curvature=curvature,
        name="anisotropic_consolidation",
        device=device,
    )
    _, posterior, updater = _new_posterior(
        resolved_config,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(resolved_config.seed + 1)
    consolidation_objective = _loss_objective(consolidation_task)
    for _ in range(resolved_config.consolidation_steps):
        updater.step(
            consolidation_objective,
            generator=generator,
        )

    sigma_before = posterior.stds["theta"].clone()
    precision_before = sigma_before.reciprocal().square()
    mean_before = posterior.means["theta"].clone()
    consolidated_state = updater.state_dict()

    signs = torch.where(
        torch.arange(
            resolved_config.dimension,
            device=device,
        )
        % 2
        == 0,
        1.0,
        -1.0,
    )
    shifted_task = diagonal_quadratic(
        resolved_config.dimension,
        optimum=signs,
        curvature=curvature,
        name="abrupt_shift",
        device=device,
    )
    updater.step(
        _loss_objective(shifted_task),
        generator=torch.Generator(device=device).manual_seed(resolved_config.seed + 2),
    )
    movement = (posterior.means["theta"] - mean_before).abs()

    baseline_loss = consolidation_task.loss(mean_before)
    perturbation_importance = torch.empty_like(mean_before)
    for index in range(resolved_config.dimension):
        perturbed = mean_before.clone()
        perturbed[index] += resolved_config.perturbation_size
        perturbation_importance[index] = consolidation_task.loss(perturbed) - baseline_loss

    curvature_updater = _clone_updater(
        consolidated_state,
        resolved_config,
        device=device,
    )
    frozen_posterior = curvature_updater.posterior
    epsilons = frozen_posterior.sample_epsilons(
        samples=resolved_config.curvature_samples,
        antithetic=True,
        generator=torch.Generator(device=device).manual_seed(resolved_config.seed + 3),
    )
    curvature_terms: list[Tensor] = []
    for epsilon in epsilons:
        theta = frozen_posterior.parameters_from_epsilon(epsilon)["theta"]
        gradient = consolidation_task.gradient(theta.detach())
        curvature_terms.append(gradient * epsilon["theta"])
    curvature_signal = torch.stack(curvature_terms).mean(dim=0)
    curvature_expected = curvature * sigma_before

    quartile = max(1, resolved_config.dimension // 4)
    sigma_order = torch.argsort(sigma_before)
    low_sigma = sigma_order[:quartile]
    high_sigma = sigma_order[-quartile:]
    freezing = {
        "none": _adapt_with_freeze(
            consolidated_state,
            resolved_config,
            shifted_task,
            None,
            device=device,
            generator_seed=resolved_config.seed + 10,
        ),
        "freeze_low_sigma": _adapt_with_freeze(
            consolidated_state,
            resolved_config,
            shifted_task,
            low_sigma,
            device=device,
            generator_seed=resolved_config.seed + 10,
        ),
        "freeze_high_sigma": _adapt_with_freeze(
            consolidated_state,
            resolved_config,
            shifted_task,
            high_sigma,
            device=device,
            generator_seed=resolved_config.seed + 10,
        ),
    }

    curvature_relative_error = float(
        ((curvature_signal - curvature_expected).abs() / curvature_expected.abs().clamp_min(1e-8))
        .mean()
        .item()
    )
    summary: dict[str, object] = {
        "seed": resolved_config.seed,
        "dimension": resolved_config.dimension,
        "movement_sigma_spearman": _spearman(
            sigma_before,
            movement,
        ),
        "perturbation_precision_spearman": _spearman(
            precision_before,
            perturbation_importance,
        ),
        "perturbation_sigma_spearman": _spearman(
            sigma_before,
            perturbation_importance,
        ),
        "curvature_signal_mean_relative_error": (curvature_relative_error),
        "freezing_target_loss": freezing,
    }

    parameter_csv = output / "mechanistic_parameters.csv"
    _save_parameter_csv(
        parameter_csv,
        curvature=curvature,
        sigma=sigma_before,
        precision=precision_before,
        movement=movement,
        perturbation_importance=perturbation_importance,
        curvature_signal=curvature_signal,
        curvature_expected=curvature_expected,
    )
    freezing_csv = output / "freezing_counterfactual.csv"
    with freezing_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "target_loss"])
        for condition, loss in freezing.items():
            writer.writerow([condition, loss])

    figures = _save_figures(
        output,
        sigma=sigma_before,
        precision=precision_before,
        movement=movement,
        perturbation_importance=perturbation_importance,
        curvature_signal=curvature_signal,
        curvature_expected=curvature_expected,
        freezing=freezing,
    )
    summary["artifacts"] = [
        parameter_csv.name,
        freezing_csv.name,
        *figures,
        "mechanistic_summary.json",
    ]
    summary_path = output / "mechanistic_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
