"""Factorized Gaussian posterior for Bayesian neural-network parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.bayes.posterior import ParameterPosterior


@dataclass(frozen=True)
class PosteriorBounds:
    """Numerical bounds applied to posterior standard deviations."""

    sigma_min: float = 1e-6
    sigma_max: float = 10.0

    def validate(self) -> None:
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be strictly positive")
        if self.sigma_max < self.sigma_min:
            raise ValueError("sigma_max must be >= sigma_min")


class DiagonalGaussianPosterior(ParameterPosterior):
    """Independent Gaussian posterior over each trainable module parameter.

    Posterior means/stds and original priors are stored in FP32 even when the
    source module uses a lower precision dtype. Sampled parameters are cast
    back to each source parameter dtype for functional evaluation.
    """

    def __init__(
        self,
        means: Mapping[str, Tensor],
        stds: Mapping[str, Tensor],
        prior_means: Mapping[str, Tensor],
        prior_stds: Mapping[str, Tensor],
        parameter_dtypes: Mapping[str, torch.dtype],
        bounds: PosteriorBounds | None = None,
    ) -> None:
        self.bounds = bounds or PosteriorBounds()
        self.bounds.validate()
        keys = set(means)
        if not keys:
            raise ValueError("posterior must contain at least one parameter")
        for mapping_name, mapping in {
            "stds": stds,
            "prior_means": prior_means,
            "prior_stds": prior_stds,
            "parameter_dtypes": parameter_dtypes,
        }.items():
            if set(mapping) != keys:
                raise ValueError(f"{mapping_name} keys do not match posterior mean keys")

        self.means = {k: self._state_tensor(v) for k, v in means.items()}
        self.stds = {k: self._state_tensor(v) for k, v in stds.items()}
        self.prior_means = {k: self._state_tensor(v) for k, v in prior_means.items()}
        self.prior_stds = {k: self._state_tensor(v) for k, v in prior_stds.items()}
        self.parameter_dtypes = dict(parameter_dtypes)
        self._validate_shapes_and_values()
        self.clamp_stds_()

    @staticmethod
    def _state_tensor(value: Tensor) -> Tensor:
        return value.detach().to(dtype=torch.float32).clone()

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        prior_std: float = 0.1,
        prior_mean: float | None = None,
        bounds: PosteriorBounds | None = None,
    ) -> "DiagonalGaussianPosterior":
        if prior_std <= 0:
            raise ValueError("prior_std must be strictly positive")
        named = {name: param for name, param in module.named_parameters() if param.requires_grad}
        if not named:
            raise ValueError("module has no trainable parameters")
        means = {name: p.detach() for name, p in named.items()}
        if prior_mean is None:
            prior_means = {name: p.detach() for name, p in named.items()}
        else:
            prior_means = {name: torch.full_like(p.detach(), prior_mean) for name, p in named.items()}
        stds = {name: torch.full_like(p.detach(), prior_std) for name, p in named.items()}
        prior_stds = {name: torch.full_like(p.detach(), prior_std) for name, p in named.items()}
        dtypes = {name: p.dtype for name, p in named.items()}
        return cls(means, stds, prior_means, prior_stds, dtypes, bounds)

    @property
    def device(self) -> torch.device:
        devices = {tensor.device for tensor in self.means.values()}
        if len(devices) != 1:
            raise RuntimeError("posterior parameters span multiple devices")
        return next(iter(devices))

    def _validate_shapes_and_values(self) -> None:
        for name, mean in self.means.items():
            tensors = (self.stds[name], self.prior_means[name], self.prior_stds[name])
            if any(t.shape != mean.shape for t in tensors):
                raise ValueError(f"posterior shape mismatch for {name}")
            if not torch.isfinite(mean).all():
                raise ValueError(f"nonfinite posterior mean for {name}")
            if not torch.isfinite(self.stds[name]).all() or torch.any(self.stds[name] <= 0):
                raise ValueError(f"invalid posterior std for {name}")
            if not torch.isfinite(self.prior_stds[name]).all() or torch.any(
                self.prior_stds[name] <= 0
            ):
                raise ValueError(f"invalid prior std for {name}")

    def clamp_stds_(self) -> None:
        for name in self.stds:
            self.stds[name].clamp_(self.bounds.sigma_min, self.bounds.sigma_max)

    def assert_finite(self) -> None:
        for name in self.means:
            if not torch.isfinite(self.means[name]).all():
                raise FloatingPointError(f"nonfinite posterior mean for {name}")
            if not torch.isfinite(self.stds[name]).all():
                raise FloatingPointError(f"nonfinite posterior std for {name}")

    def mean_parameters(self) -> dict[str, Tensor]:
        return {
            name: value.to(dtype=self.parameter_dtypes[name])
            for name, value in self.means.items()
        }

    def sample_epsilons(
        self,
        *,
        samples: int,
        antithetic: bool = False,
        generator: torch.Generator | None = None,
    ) -> list[dict[str, Tensor]]:
        if samples < 1:
            raise ValueError("samples must be >= 1")
        if antithetic and samples % 2:
            raise ValueError("antithetic sampling requires an even sample count")
        base_count = samples // 2 if antithetic else samples
        base: list[dict[str, Tensor]] = []
        for _ in range(base_count):
            base.append(
                {
                    name: torch.randn(
                        mean.shape,
                        dtype=torch.float32,
                        device=mean.device,
                        generator=generator,
                    )
                    for name, mean in self.means.items()
                }
            )
        if not antithetic:
            return base
        result: list[dict[str, Tensor]] = []
        for eps in base:
            result.append(eps)
            result.append({name: -value for name, value in eps.items()})
        return result

    def parameters_from_epsilon(self, epsilon: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if set(epsilon) != set(self.means):
            raise ValueError("epsilon keys do not match posterior")
        sampled: dict[str, Tensor] = {}
        for name, mean in self.means.items():
            theta_fp32 = mean + self.stds[name] * epsilon[name].to(dtype=torch.float32)
            theta = theta_fp32.to(dtype=self.parameter_dtypes[name]).detach().requires_grad_(True)
            sampled[name] = theta
        return sampled

    def sample(
        self,
        *,
        antithetic: bool = False,
        samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> list[dict[str, Tensor]]:
        return [
            self.parameters_from_epsilon(eps)
            for eps in self.sample_epsilons(
                samples=samples, antithetic=antithetic, generator=generator
            )
        ]

    def sync_module(self, module: nn.Module) -> None:
        module_parameters = dict(module.named_parameters())
        missing = set(self.means) - set(module_parameters)
        if missing:
            raise KeyError(f"module is missing posterior parameters: {sorted(missing)}")
        with torch.no_grad():
            for name, mean in self.means.items():
                target = module_parameters[name]
                target.copy_(mean.to(device=target.device, dtype=target.dtype))

    def to(self, device: torch.device | str) -> "DiagonalGaussianPosterior":
        for mapping in (self.means, self.stds, self.prior_means, self.prior_stds):
            for name, value in mapping.items():
                mapping[name] = value.to(device=device)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "posterior_type": "diagonal_gaussian",
            "version": 1,
            "means": {k: v.detach().clone() for k, v in self.means.items()},
            "stds": {k: v.detach().clone() for k, v in self.stds.items()},
            "prior_means": {k: v.detach().clone() for k, v in self.prior_means.items()},
            "prior_stds": {k: v.detach().clone() for k, v in self.prior_stds.items()},
            "parameter_dtypes": {
                k: str(v).replace("torch.", "") for k, v in self.parameter_dtypes.items()
            },
            "bounds": {
                "sigma_min": self.bounds.sigma_min,
                "sigma_max": self.bounds.sigma_max,
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("posterior_type") != "diagonal_gaussian":
            raise ValueError("incompatible posterior type")
        if state.get("version") != 1:
            raise ValueError("unsupported posterior checkpoint version")
        for field, target in (
            ("means", self.means),
            ("stds", self.stds),
            ("prior_means", self.prior_means),
            ("prior_stds", self.prior_stds),
        ):
            incoming = state[field]
            if set(incoming) != set(target):
                raise ValueError(f"checkpoint {field} keys do not match posterior")
            for name in target:
                if incoming[name].shape != target[name].shape:
                    raise ValueError(f"checkpoint shape mismatch for {name}")
                target[name].copy_(
                    incoming[name].to(device=target[name].device, dtype=torch.float32)
                )
        self.clamp_stds_()
        self.assert_finite()
