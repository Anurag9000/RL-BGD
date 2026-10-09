"""Bounds and priors must be finite and representable in posterior FP32 state."""

import pytest
import torch
from torch import nn

from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior, PosteriorBounds


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("sigma_min", float("nan")),
        ("sigma_min", float("inf")),
        ("sigma_min", True),
        ("sigma_min", 1e-100),
        ("sigma_max", float("nan")),
        ("sigma_max", float("inf")),
        ("sigma_max", 1e100),
        ("sigma_max", False),
    ],
)
def test_invalid_posterior_bounds_fail_before_clamping(field: str, invalid: object) -> None:
    values = {"sigma_min": 1e-6, "sigma_max": 10.0}
    values[field] = invalid
    with pytest.raises((TypeError, ValueError)):
        PosteriorBounds(**values).validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 0.0, -0.1])
def test_from_module_rejects_invalid_prior_std(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        DiagonalGaussianPosterior.from_module(nn.Linear(1, 1), prior_std=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_from_module_rejects_invalid_prior_mean(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        DiagonalGaussianPosterior.from_module(nn.Linear(1, 1), prior_mean=value)


def test_valid_small_bounds_and_priors_still_initialize() -> None:
    module = nn.Linear(1, 1)
    posterior = DiagonalGaussianPosterior.from_module(
        module, prior_std=1e-30, bounds=PosteriorBounds(sigma_min=1e-30, sigma_max=10.0)
    )
    assert torch.isfinite(posterior.stds["weight"]).all()
    assert torch.all(posterior.stds["weight"] > 0)
