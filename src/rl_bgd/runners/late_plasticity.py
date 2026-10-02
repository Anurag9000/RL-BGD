"""Controlled late-life plasticity stress test for Bayesian Gradient Descent."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import (
    DiagonalGaussianPosterior,
    PosteriorBounds,
)
from rl_bgd.envs.synthetic.quadratic import QuadraticTask, diagonal_quadratic
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


class _PlasticityVector(nn.Module):
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


def _objective(
    task: QuadraticTask,
) -> Callable[
    [Mapping[str, Tensor]],
    Tensor,
]:
    def loss(
        params: Mapping[str, Tensor],
    ) -> Tensor:
        return task.loss(
            params["theta"]
        )

    return loss


def _normalized_trapezoid_auc(
    values: list[float],
) -> float:
    if len(values) < 2:
        raise ValueError(
            "post-shift AUC requires at least two evaluation points"
        )
    baseline = values[0]
    if baseline <= 0:
        raise ValueError(
            "post-shift initial loss must be positive"
        )
    normalized = [
        value / baseline
        for value in values
    ]
    area = sum(
        0.5
        * (
            normalized[index]
            + normalized[index + 1]
        )
        for index in range(
            len(normalized) - 1
        )
    )
    return area / float(
        len(normalized) - 1
    )


def run_late_plasticity_quadratic(
    *,
    consolidation_retention: float = 1.0,
    dimension: int = 16,
    consolidation_steps: int = 400,
    adaptation_steps: int = 32,
    eta: float = 0.15,
    prior_std: float = 0.5,
    shift: float = 1.0,
    curvature: float = 1.0,
    mc_samples: int = 16,
    seed: int = 0,
    device: str = "auto",
) -> dict[str, object]:
    """Measure adaptation after a long posterior-consolidation phase.

    The consolidation retention is applied only before the abrupt task shift.
    Adaptation always uses retention 1.0, so matched conditions differ only in
    the posterior state inherited from their pre-shift history.
    """

    if dimension < 1:
        raise ValueError(
            "dimension must be positive"
        )
    if (
        consolidation_steps < 1
        or adaptation_steps < 1
    ):
        raise ValueError(
            "plasticity phase lengths must be positive"
        )
    if eta <= 0 or prior_std <= 0:
        raise ValueError(
            "eta and prior_std must be positive"
        )
    if shift == 0:
        raise ValueError(
            "plasticity stress test requires a nonzero shift"
        )
    if curvature <= 0:
        raise ValueError(
            "curvature must be positive"
        )
    if mc_samples < 2 or mc_samples % 2:
        raise ValueError(
            "mc_samples must be an even integer >= 2"
        )
    if not 0.0 <= consolidation_retention <= 1.0:
        raise ValueError(
            "consolidation_retention must lie in [0, 1]"
        )

    seed_everything(
        seed,
        deterministic=True,
    )
    resolved_device = resolve_device(
        device
    )
    model = _PlasticityVector(
        dimension,
        device=resolved_device,
    )
    posterior = DiagonalGaussianPosterior.from_module(
        model,
        prior_std=prior_std,
        bounds=PosteriorBounds(
            sigma_min=1e-8,
            sigma_max=max(
                2.0,
                prior_std * 4.0,
            ),
        ),
    )
    updater = BGDUpdater(
        posterior,
        BGDConfig(
            eta=eta,
            mc_samples=mc_samples,
            antithetic=True,
        ),
    )
    source_task = diagonal_quadratic(
        dimension,
        optimum=0.0,
        curvature=curvature,
        name="pre_shift_consolidation",
        device=resolved_device,
    )
    target_task = diagonal_quadratic(
        dimension,
        optimum=shift,
        curvature=curvature,
        name="post_shift_target",
        device=resolved_device,
    )
    generator = torch.Generator(
        device=resolved_device
    ).manual_seed(
        seed + 31
    )

    initial_sigma_mean = float(
        posterior.stds[
            "theta"
        ].mean().item()
    )
    source_objective = _objective(
        source_task
    )
    for _ in range(
        consolidation_steps
    ):
        updater.step(
            source_objective,
            generator=generator,
            retention=consolidation_retention,
        )

    mean_before_shift = posterior.means[
        "theta"
    ].clone()
    sigma_before_shift = posterior.stds[
        "theta"
    ].clone()
    pre_shift_sigma_mean = float(
        sigma_before_shift.mean().item()
    )
    pre_shift_effective_lr_mean = float(
        (
            eta
            * sigma_before_shift.square()
        ).mean().item()
    )
    source_loss_before_shift = float(
        source_task.loss(
            mean_before_shift
        ).item()
    )
    initial_target_loss = float(
        target_task.loss(
            mean_before_shift
        ).item()
    )

    target_objective = _objective(
        target_task
    )
    loss_curve = [
        initial_target_loss
    ]
    first_step_mean_movement = 0.0
    for step in range(
        adaptation_steps
    ):
        before = posterior.means[
            "theta"
        ].clone()
        updater.step(
            target_objective,
            generator=generator,
            retention=1.0,
        )
        if step == 0:
            first_step_mean_movement = float(
                (
                    posterior.means[
                        "theta"
                    ]
                    - before
                )
                .abs()
                .mean()
                .item()
            )
        loss_curve.append(
            float(
                target_task.loss(
                    posterior.means[
                        "theta"
                    ]
                ).item()
            )
        )

    final_target_loss = loss_curve[
        -1
    ]
    post_shift_normalized_auc = (
        _normalized_trapezoid_auc(
            loss_curve
        )
    )
    post_shift_loss_timeline = [
        {
            "step": step,
            "target_loss": loss,
            "normalized_target_loss": (
                loss
                / initial_target_loss
            ),
        }
        for step, loss in enumerate(
            loss_curve
        )
    ]
    recovery_fraction = (
        (
            initial_target_loss
            - final_target_loss
        )
        / initial_target_loss
    )
    final_sigma_mean = float(
        posterior.stds[
            "theta"
        ].mean().item()
    )

    return {
        "hypothesis_id": "C",
        "dimension": dimension,
        "consolidation_steps": (
            consolidation_steps
        ),
        "adaptation_steps": (
            adaptation_steps
        ),
        "consolidation_retention": (
            consolidation_retention
        ),
        "adaptation_retention": 1.0,
        "initial_sigma_mean": (
            initial_sigma_mean
        ),
        "pre_shift_sigma_mean": (
            pre_shift_sigma_mean
        ),
        "pre_shift_effective_lr_mean": (
            pre_shift_effective_lr_mean
        ),
        "source_loss_before_shift": (
            source_loss_before_shift
        ),
        "post_shift_initial_loss": (
            initial_target_loss
        ),
        "post_shift_final_loss": (
            final_target_loss
        ),
        "post_shift_normalized_auc": (
            post_shift_normalized_auc
        ),
        "first_step_mean_movement": (
            first_step_mean_movement
        ),
        "recovery_fraction": (
            recovery_fraction
        ),
        "final_sigma_mean": (
            final_sigma_mean
        ),
        "post_shift_loss_curve": (
            loss_curve
        ),
        "post_shift_loss_timeline": (
            post_shift_loss_timeline
        ),
        "information_access": {
            "task_identity_used_by_optimizer": False,
            "task_boundary_used_by_optimizer": False,
            "task_shift_applied_externally": True,
        },
    }


def main() -> None:
    print(
        json.dumps(
            run_late_plasticity_quadratic(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
