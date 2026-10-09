"""Factorized Gaussian posterior for Bayesian neural-network parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.bayes.posterior import ParameterPosterior
from rl_bgd.utils.checkpoint_progress import checkpoint_integer


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
    ) -> DiagonalGaussianPosterior:
        if prior_std <= 0:
            raise ValueError("prior_std must be strictly positive")
        named = {name: param for name, param in module.named_parameters() if param.requires_grad}
        if not named:
            raise ValueError("module has no trainable parameters")
        means = {name: param.detach() for name, param in named.items()}
        if prior_mean is None:
            prior_means = {name: param.detach() for name, param in named.items()}
        else:
            prior_means = {
                name: torch.full_like(
                    param.detach(),
                    prior_mean,
                )
                for name, param in named.items()
            }
        stds = {
            name: torch.full_like(
                param.detach(),
                prior_std,
            )
            for name, param in named.items()
        }
        prior_stds = {
            name: torch.full_like(
                param.detach(),
                prior_std,
            )
            for name, param in named.items()
        }
        dtypes = {name: param.dtype for name, param in named.items()}
        return cls(
            means,
            stds,
            prior_means,
            prior_stds,
            dtypes,
            bounds,
        )

    @property
    def device(self) -> torch.device:
        devices = {tensor.device for tensor in self.means.values()}
        if len(devices) != 1:
            raise RuntimeError("posterior parameters span multiple devices")
        return next(iter(devices))

    def _validate_shapes_and_values(self) -> None:
        for name, mean in self.means.items():
            tensors = (
                self.stds[name],
                self.prior_means[name],
                self.prior_stds[name],
            )
            if any(tensor.shape != mean.shape for tensor in tensors):
                raise ValueError(f"posterior shape mismatch for {name}")
            if not torch.isfinite(mean).all():
                raise ValueError(f"nonfinite posterior mean for {name}")
            if not torch.isfinite(self.stds[name]).all() or torch.any(self.stds[name] <= 0):
                raise ValueError(f"invalid posterior std for {name}")
            if not torch.isfinite(self.prior_means[name]).all():
                raise ValueError(f"nonfinite prior mean for {name}")
            if not torch.isfinite(self.prior_stds[name]).all() or torch.any(
                self.prior_stds[name] <= 0
            ):
                raise ValueError(f"invalid prior std for {name}")

    def clamp_stds_(self) -> None:
        for name in self.stds:
            self.stds[name].clamp_(
                self.bounds.sigma_min,
                self.bounds.sigma_max,
            )

    def assert_finite(self) -> None:
        for name in self.means:
            if not torch.isfinite(self.means[name]).all():
                raise FloatingPointError(f"nonfinite posterior mean for {name}")
            if not torch.isfinite(self.stds[name]).all():
                raise FloatingPointError(f"nonfinite posterior std for {name}")

    def mean_parameters(self) -> dict[str, Tensor]:
        return {
            name: value.to(dtype=self.parameter_dtypes[name]) for name, value in self.means.items()
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

    def parameters_from_epsilon(
        self,
        epsilon: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
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
                samples=samples,
                antithetic=antithetic,
                generator=generator,
            )
        ]

    def sync_module(
        self,
        module: nn.Module,
    ) -> None:
        module_parameters = dict(module.named_parameters())
        missing = set(self.means) - set(module_parameters)
        if missing:
            raise KeyError(f"module is missing posterior parameters: {sorted(missing)}")
        staged: dict[str, Tensor] = {}
        for name, mean in self.means.items():
            target = module_parameters[name]
            if target.shape != mean.shape:
                raise ValueError(f"module parameter shape mismatch for {name}")
            if target.dtype != self.parameter_dtypes[name]:
                raise ValueError(f"module parameter dtype mismatch for {name}")
            candidate = mean.to(device=target.device, dtype=target.dtype)
            if not torch.isfinite(candidate).all().item():
                raise FloatingPointError(f"nonfinite module parameter after conversion for {name}")
            staged[name] = candidate

        with torch.no_grad():
            for name, candidate in staged.items():
                module_parameters[name].copy_(candidate)

    def to(
        self,
        device: torch.device | str,
    ) -> DiagonalGaussianPosterior:
        for mapping in (
            self.means,
            self.stds,
            self.prior_means,
            self.prior_stds,
        ):
            for name, value in mapping.items():
                mapping[name] = value.to(device=device)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "posterior_type": "diagonal_gaussian",
            "version": 1,
            "means": {key: value.detach().clone() for key, value in self.means.items()},
            "stds": {key: value.detach().clone() for key, value in self.stds.items()},
            "prior_means": {key: value.detach().clone() for key, value in self.prior_means.items()},
            "prior_stds": {key: value.detach().clone() for key, value in self.prior_stds.items()},
            "parameter_dtypes": {
                key: str(value).replace(
                    "torch.",
                    "",
                )
                for key, value in self.parameter_dtypes.items()
            },
            "bounds": {
                "sigma_min": self.bounds.sigma_min,
                "sigma_max": self.bounds.sigma_max,
            },
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        if state.get("posterior_type") != "diagonal_gaussian":
            raise ValueError("incompatible posterior type")
        version = checkpoint_integer(
            state.get("version"),
            name="posterior checkpoint version",
        )
        if version != 1:
            raise ValueError("unsupported posterior checkpoint version")

        saved_dtypes = state.get("parameter_dtypes")
        if not isinstance(saved_dtypes, Mapping):
            raise ValueError("posterior checkpoint is missing parameter dtypes")
        expected_dtypes = {
            key: str(value).replace("torch.", "") for key, value in self.parameter_dtypes.items()
        }
        if dict(saved_dtypes) != expected_dtypes:
            raise ValueError("posterior checkpoint parameter dtype mismatch")

        saved_bounds = state.get("bounds")
        if not isinstance(saved_bounds, Mapping):
            raise ValueError("posterior checkpoint is missing bounds")
        expected_bounds = {
            "sigma_min": self.bounds.sigma_min,
            "sigma_max": self.bounds.sigma_max,
        }
        if dict(saved_bounds) != expected_bounds:
            raise ValueError("posterior checkpoint bounds mismatch")

        prepared: dict[str, dict[str, Tensor]] = {}
        for field, target in (
            ("means", self.means),
            ("stds", self.stds),
            ("prior_means", self.prior_means),
            ("prior_stds", self.prior_stds),
        ):
            incoming = state[field]
            if not isinstance(incoming, Mapping):
                raise TypeError(f"checkpoint {field} must be a mapping")
            if set(incoming) != set(target):
                raise ValueError(f"checkpoint {field} keys do not match posterior")
            field_values: dict[str, Tensor] = {}
            for name in target:
                value = incoming[name]
                if not isinstance(value, Tensor):
                    raise TypeError(f"checkpoint {field}/{name} must be a tensor")
                if value.shape != target[name].shape:
                    raise ValueError(f"checkpoint shape mismatch for {name}")
                if value.dtype != target[name].dtype:
                    raise ValueError(f"checkpoint dtype mismatch for {field}/{name}")
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"checkpoint {field}/{name} contains non-finite values")
                if field in {"stds", "prior_stds"} and torch.any(value <= 0).item():
                    raise ValueError(f"checkpoint {field}/{name} contains non-positive std")
                if field == "stds" and (
                    torch.any(value < self.bounds.sigma_min).item()
                    or torch.any(value > self.bounds.sigma_max).item()
                ):
                    raise ValueError(f"checkpoint stds/{name} violates posterior bounds")
                field_values[name] = value.to(device=target[name].device).clone()
            prepared[field] = field_values

        for field, target in (
            ("means", self.means),
            ("stds", self.stds),
            ("prior_means", self.prior_means),
            ("prior_stds", self.prior_stds),
        ):
            for name in target:
                target[name].copy_(prepared[field][name])
        self.assert_finite()
