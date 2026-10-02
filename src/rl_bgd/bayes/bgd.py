"""Reusable Bayesian Gradient Descent update engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call

from rl_bgd.bayes.diagnostics import posterior_diagnostics
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior
from rl_bgd.bayes.mc_sampling import aggregate_bgd_statistics
from rl_bgd.bayes.tempering import temper_diagonal_gaussian


@dataclass(frozen=True)
class BGDLoss:
    """Separate objectives for posterior mean learning and uncertainty evidence."""

    mean: Tensor
    uncertainty: Tensor


Objective = Callable[[Mapping[str, Tensor]], Tensor | BGDLoss]


@dataclass(frozen=True)
class BGDConfig:
    """Numerical and sampling controls for one BGD updater."""

    eta: float = 1.0
    mc_samples: int = 4
    antithetic: bool = True
    temper_retention: float = 1.0
    evidence_temperature: float = 1.0

    def validate(self) -> None:
        if self.eta <= 0:
            raise ValueError("eta must be strictly positive")
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")
        if self.antithetic and self.mc_samples % 2:
            raise ValueError("antithetic sampling requires an even mc_samples")
        if not 0.0 <= self.temper_retention <= 1.0:
            raise ValueError("temper_retention must lie in [0, 1]")
        if self.evidence_temperature <= 0:
            raise ValueError("evidence_temperature must be strictly positive")


@dataclass(frozen=True)
class BGDStepResult:
    """Observable statistics returned by a completed BGD update."""

    mean_loss: float
    uncertainty_loss: float
    diagnostics: dict[str, float]
    gradient_norm: float
    uncertainty_gradient_norm: float
    c_norm: float
    retention: float
    evidence_temperature: float


class BGDUpdater:
    """Monte Carlo BGD updates over a diagonal Gaussian posterior."""

    def __init__(
        self,
        posterior: DiagonalGaussianPosterior,
        config: BGDConfig | None = None,
    ) -> None:
        self.posterior = posterior
        self.config = config or BGDConfig()
        self.config.validate()
        self.step_count = 0

    def _temper(self, retention: float | None) -> float:
        applied = self.config.temper_retention if retention is None else retention
        if not 0.0 <= applied <= 1.0:
            raise ValueError("retention override must lie in [0, 1]")
        if applied == 1.0:
            return applied
        means, stds = temper_diagonal_gaussian(
            self.posterior.means,
            self.posterior.stds,
            self.posterior.prior_means,
            self.posterior.prior_stds,
            retention=applied,
        )
        for name in self.posterior.means:
            self.posterior.means[name].copy_(means[name])
            self.posterior.stds[name].copy_(stds[name])
        self.posterior.clamp_stds_()
        return applied

    def _evidence_temperature(
        self,
        evidence_temperature: float | None,
    ) -> float:
        applied = (
            self.config.evidence_temperature
            if evidence_temperature is None
            else float(evidence_temperature)
        )
        if applied <= 0:
            raise ValueError("evidence_temperature override must be strictly positive")
        return applied

    @staticmethod
    def _check_scalar_loss(loss: Tensor, name: str) -> None:
        if loss.ndim != 0:
            raise ValueError(f"BGD {name} objective must return a scalar loss")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite BGD {name} objective")

    def step(
        self,
        objective: Objective,
        *,
        generator: torch.Generator | None = None,
        retention: float | None = None,
        evidence_temperature: float | None = None,
    ) -> BGDStepResult:
        """Take one BGD step using distinct mean/evidence gradient channels."""

        applied_retention = self._temper(retention)
        applied_evidence_temperature = self._evidence_temperature(evidence_temperature)
        epsilons = self.posterior.sample_epsilons(
            samples=self.config.mc_samples,
            antithetic=self.config.antithetic,
            generator=generator,
        )
        gradients: list[dict[str, Tensor]] = []
        uncertainty_gradients: list[dict[str, Tensor]] = []
        losses: list[Tensor] = []
        uncertainty_losses: list[Tensor] = []
        ordered_names = list(self.posterior.means)

        for epsilon in epsilons:
            sampled = self.posterior.parameters_from_epsilon(epsilon)
            objective_value = objective(sampled)
            if isinstance(objective_value, BGDLoss):
                mean_loss = objective_value.mean
                uncertainty_loss = objective_value.uncertainty
                self._check_scalar_loss(mean_loss, "mean")
                self._check_scalar_loss(uncertainty_loss, "uncertainty")
                tempered_mean_loss = applied_evidence_temperature * mean_loss
                tempered_uncertainty_loss = applied_evidence_temperature * uncertainty_loss
                grads_tuple = torch.autograd.grad(
                    tempered_mean_loss,
                    tuple(sampled[name] for name in ordered_names),
                    allow_unused=False,
                    create_graph=False,
                    retain_graph=True,
                )
                uncertainty_grads_tuple = torch.autograd.grad(
                    tempered_uncertainty_loss,
                    tuple(sampled[name] for name in ordered_names),
                    allow_unused=False,
                    create_graph=False,
                )
            else:
                mean_loss = objective_value
                uncertainty_loss = objective_value
                self._check_scalar_loss(mean_loss, "mean")
                tempered_mean_loss = applied_evidence_temperature * mean_loss
                grads_tuple = torch.autograd.grad(
                    tempered_mean_loss,
                    tuple(sampled[name] for name in ordered_names),
                    allow_unused=False,
                    create_graph=False,
                )
                uncertainty_grads_tuple = grads_tuple

            gradients.append(
                {
                    name: grad
                    for name, grad in zip(
                        ordered_names,
                        grads_tuple,
                        strict=True,
                    )
                }
            )
            uncertainty_gradients.append(
                {
                    name: grad
                    for name, grad in zip(
                        ordered_names,
                        uncertainty_grads_tuple,
                        strict=True,
                    )
                }
            )
            losses.append(mean_loss.detach().float())
            uncertainty_losses.append(uncertainty_loss.detach().float())

        g_bar, c = aggregate_bgd_statistics(
            gradients,
            epsilons,
            uncertainty_gradients=uncertainty_gradients,
        )
        uncertainty_g_bar, _ = aggregate_bgd_statistics(
            uncertainty_gradients,
            epsilons,
        )
        with torch.no_grad():
            for name in ordered_names:
                sigma = self.posterior.stds[name]
                mean = self.posterior.means[name]
                g = g_bar[name].to(device=mean.device)
                curvature_signal = c[name].to(device=mean.device)
                mean.add_(-self.config.eta * sigma.square() * g)
                half_term = 0.5 * sigma * curvature_signal
                new_sigma = sigma * torch.sqrt(1.0 + half_term.square())
                new_sigma -= 0.5 * sigma.square() * curvature_signal
                sigma.copy_(new_sigma)

        self.posterior.clamp_stds_()
        self.posterior.assert_finite()
        self.step_count += 1

        gradient_norm = (
            torch.stack([torch.sum(value.square()) for value in g_bar.values()]).sum().sqrt().item()
        )
        uncertainty_gradient_norm = (
            torch.stack([torch.sum(value.square()) for value in uncertainty_g_bar.values()])
            .sum()
            .sqrt()
            .item()
        )
        c_norm = (
            torch.stack([torch.sum(value.square()) for value in c.values()]).sum().sqrt().item()
        )
        return BGDStepResult(
            mean_loss=torch.stack(losses).mean().item(),
            uncertainty_loss=torch.stack(uncertainty_losses).mean().item(),
            diagnostics=posterior_diagnostics(
                self.posterior.stds,
                eta=self.config.eta,
                evidence_temperature=applied_evidence_temperature,
            ),
            gradient_norm=gradient_norm,
            uncertainty_gradient_norm=uncertainty_gradient_norm,
            c_norm=c_norm,
            retention=applied_retention,
            evidence_temperature=applied_evidence_temperature,
        )

    def step_module(
        self,
        module: nn.Module,
        loss_fn: Callable[[Any], Tensor],
        *args: Any,
        uncertainty_loss_fn: Callable[[Any], Tensor] | None = None,
        buffers: Mapping[str, Tensor] | None = None,
        generator: torch.Generator | None = None,
        retention: float | None = None,
        evidence_temperature: float | None = None,
        **kwargs: Any,
    ) -> BGDStepResult:
        """Convenience wrapper for a single-module forward objective."""

        module_buffers = dict(module.named_buffers()) if buffers is None else dict(buffers)

        def objective(params: Mapping[str, Tensor]) -> Tensor | BGDLoss:
            output = functional_call(
                module,
                (dict(params), module_buffers),
                args,
                kwargs,
            )
            mean_loss = loss_fn(output)
            if uncertainty_loss_fn is None:
                return mean_loss
            return BGDLoss(
                mean=mean_loss,
                uncertainty=uncertainty_loss_fn(output),
            )

        result = self.step(
            objective,
            generator=generator,
            retention=retention,
            evidence_temperature=evidence_temperature,
        )
        self.posterior.sync_module(module)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "updater_type": "bgd",
            "version": 1,
            "step_count": self.step_count,
            "config": {
                "eta": self.config.eta,
                "mc_samples": self.config.mc_samples,
                "antithetic": self.config.antithetic,
                "temper_retention": self.config.temper_retention,
                "evidence_temperature": self.config.evidence_temperature,
            },
            "posterior": self.posterior.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("updater_type") != "bgd" or state.get("version") != 1:
            raise ValueError("incompatible BGD updater checkpoint")
        self.posterior.load_state_dict(state["posterior"])
        self.step_count = int(state["step_count"])
