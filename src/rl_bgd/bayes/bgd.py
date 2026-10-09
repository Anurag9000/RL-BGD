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
from rl_bgd.utils.checkpoint_progress import (
    checkpoint_integer,
    checkpoint_nonnegative_integer,
)
from rl_bgd.utils.config_validation import (
    config_boolean,
    config_finite_float,
    config_positive_integer,
)


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
        eta = config_finite_float(self.eta, name="BGD eta")
        samples = config_positive_integer(self.mc_samples, name="BGD mc_samples")
        antithetic = config_boolean(self.antithetic, name="BGD antithetic")
        retention = config_finite_float(self.temper_retention, name="BGD temper_retention")
        temperature = config_finite_float(
            self.evidence_temperature, name="BGD evidence_temperature"
        )
        if eta <= 0:
            raise ValueError("eta must be strictly positive")
        if antithetic and samples % 2:
            raise ValueError("antithetic sampling requires an even mc_samples")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("temper_retention must lie in [0, 1]")
        if temperature <= 0:
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
        applied = config_finite_float(
            self.config.temper_retention if retention is None else retention,
            name="BGD retention override",
        )
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
        applied = config_finite_float(
            self.config.evidence_temperature
            if evidence_temperature is None
            else evidence_temperature,
            name="BGD evidence_temperature override",
        )
        if applied <= 0:
            raise ValueError("evidence_temperature override must be strictly positive")
        return applied

    @staticmethod
    def _check_scalar_loss(loss: Tensor, name: str) -> None:
        if not isinstance(loss, Tensor):
            raise TypeError(f"BGD {name} objective must return a tensor")
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
        """Apply a BGD update atomically with respect to posterior state and progress."""

        before_means = {name: value.clone() for name, value in self.posterior.means.items()}
        before_stds = {name: value.clone() for name, value in self.posterior.stds.items()}
        before_step_count = self.step_count
        try:
            return self._step_impl(
                objective,
                generator=generator,
                retention=retention,
                evidence_temperature=evidence_temperature,
            )
        except Exception:
            with torch.no_grad():
                for name, value in before_means.items():
                    self.posterior.means[name].copy_(value)
                for name, value in before_stds.items():
                    self.posterior.stds[name].copy_(value)
            self.step_count = before_step_count
            raise

    def _step_impl(
        self,
        objective: Objective,
        *,
        generator: torch.Generator | None,
        retention: float | None,
        evidence_temperature: float | None,
    ) -> BGDStepResult:
        """Evaluate samples and commit a numerically checked BGD update."""

        applied_evidence_temperature = self._evidence_temperature(evidence_temperature)
        applied_retention = self._temper(retention)
        self.posterior.assert_finite()
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
                self._check_scalar_loss(tempered_mean_loss, "scaled mean")
                self._check_scalar_loss(tempered_uncertainty_loss, "scaled uncertainty")
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
                self._check_scalar_loss(tempered_mean_loss, "scaled mean")
                grads_tuple = torch.autograd.grad(
                    tempered_mean_loss,
                    tuple(sampled[name] for name in ordered_names),
                    allow_unused=False,
                    create_graph=False,
                )
                uncertainty_grads_tuple = grads_tuple

            for gradient in (*grads_tuple, *uncertainty_grads_tuple):
                if not torch.isfinite(gradient).all().item():
                    raise FloatingPointError("nonfinite BGD gradient")

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
        for signal in (g_bar, c, uncertainty_g_bar):
            if any(not torch.isfinite(value).all().item() for value in signal.values()):
                raise FloatingPointError("nonfinite aggregated BGD statistics")
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

        self.posterior.assert_finite()
        if any(torch.any(std <= 0).item() for std in self.posterior.stds.values()):
            raise FloatingPointError("nonpositive updated BGD standard deviation")
        self.posterior.clamp_stds_()
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

        before_means = {name: value.clone() for name, value in self.posterior.means.items()}
        before_stds = {name: value.clone() for name, value in self.posterior.stds.items()}
        before_step_count = self.step_count
        try:
            result = self.step(
                objective,
                generator=generator,
                retention=retention,
                evidence_temperature=evidence_temperature,
            )
            self.posterior.sync_module(module)
        except Exception:
            with torch.no_grad():
                for name, value in before_means.items():
                    self.posterior.means[name].copy_(value)
                for name, value in before_stds.items():
                    self.posterior.stds[name].copy_(value)
            self.step_count = before_step_count
            raise
        return result

    def _config_state(self) -> dict[str, object]:
        return {
            "eta": self.config.eta,
            "mc_samples": self.config.mc_samples,
            "antithetic": self.config.antithetic,
            "temper_retention": self.config.temper_retention,
            "evidence_temperature": self.config.evidence_temperature,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "updater_type": "bgd",
            "version": 1,
            "step_count": self.step_count,
            "config": self._config_state(),
            "posterior": self.posterior.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        version = checkpoint_integer(
            state.get("version"),
            name="BGD updater checkpoint version",
        )
        if state.get("updater_type") != "bgd" or version != 1:
            raise ValueError("incompatible BGD updater checkpoint")
        checkpoint_config = state.get("config")
        if not isinstance(checkpoint_config, Mapping):
            raise ValueError("BGD updater checkpoint is missing its configuration")
        expected_config = self._config_state()
        if set(checkpoint_config) != set(expected_config):
            raise ValueError("BGD updater checkpoint config mismatch: keys differ")
        mismatches = [
            name
            for name, expected in expected_config.items()
            if checkpoint_config[name] != expected
        ]
        if mismatches:
            details = ", ".join(
                (
                    f"{name}: checkpoint={checkpoint_config[name]!r}, "
                    f"runtime={expected_config[name]!r}"
                )
                for name in mismatches
            )
            raise ValueError(f"BGD updater checkpoint config mismatch: {details}")
        step_count = checkpoint_nonnegative_integer(
            state.get("step_count"),
            name="BGD updater checkpoint step_count",
        )
        saved_posterior = state.get("posterior")
        if not isinstance(saved_posterior, Mapping):
            raise TypeError("BGD updater checkpoint posterior must be a mapping")
        self.posterior.load_state_dict(saved_posterior)
        self.step_count = step_count
