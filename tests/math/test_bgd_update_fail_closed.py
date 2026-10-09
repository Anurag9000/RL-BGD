"""Failed Bayesian updates must not silently advance or poison the posterior."""

import pytest
import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior, PosteriorBounds


def make_updater(*, eta: float = 0.2, std: float = 0.5) -> BGDUpdater:
    module = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.zero_()
    posterior = DiagonalGaussianPosterior.from_module(
        module,
        prior_std=std,
        bounds=PosteriorBounds(sigma_min=1e-30, sigma_max=10.0),
    )
    return BGDUpdater(posterior, BGDConfig(eta=eta, mc_samples=2, antithetic=True))


def assert_unchanged(
    updater: BGDUpdater, means: dict[str, torch.Tensor], stds: dict[str, torch.Tensor],
    count: int,
) -> None:
    for name in means:
        torch.testing.assert_close(updater.posterior.means[name], means[name], rtol=0, atol=0)
        torch.testing.assert_close(updater.posterior.stds[name], stds[name], rtol=0, atol=0)
    assert updater.step_count == count


def snapshot(updater: BGDUpdater) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    return (
        {name: value.clone() for name, value in updater.posterior.means.items()},
        {name: value.clone() for name, value in updater.posterior.stds.items()},
        updater.step_count,
    )


def zero_loss(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum((value * 0).sum() for value in params.values())


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("eta", float("nan")),
        ("eta", float("inf")),
        ("temper_retention", float("nan")),
        ("temper_retention", float("inf")),
        ("evidence_temperature", float("nan")),
        ("evidence_temperature", float("inf")),
        ("mc_samples", True),
        ("mc_samples", 2.0),
        ("antithetic", 1),
    ],
)
def test_config_rejects_nonfinite_and_coerced_controls(kwarg: str, value: object) -> None:
    config = BGDConfig(**{kwarg: value})
    with pytest.raises((TypeError, ValueError)):
        config.validate()


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("retention", float("nan")),
        ("retention", float("inf")),
        ("evidence_temperature", float("nan")),
        ("evidence_temperature", float("inf")),
        ("evidence_temperature", 0.0),
    ],
)
def test_rejected_overrides_preserve_posterior(override: str, value: float) -> None:
    updater = make_updater()
    updater.posterior.stds["weight"].fill_(0.2)
    before = snapshot(updater)
    with pytest.raises(ValueError):
        updater.step(zero_loss, retention=0.5, **{override: value})
    assert_unchanged(updater, *before)


def test_late_objective_failure_after_tempering_rolls_back() -> None:
    updater = make_updater()
    updater.posterior.stds["weight"].fill_(0.2)
    before = snapshot(updater)
    calls = 0

    def fail_second(params: dict[str, torch.Tensor]) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            return params["weight"].sum() * float("nan")
        return zero_loss(params)

    with pytest.raises(FloatingPointError, match="nonfinite BGD mean"):
        updater.step(fail_second, retention=0.5)
    assert calls == 2
    assert_unchanged(updater, *before)


def test_finite_loss_nonfinite_gradient_does_not_advance() -> None:
    updater = make_updater()
    before = snapshot(updater)

    def bad_gradient(params: dict[str, torch.Tensor]) -> torch.Tensor:
        zero = params["weight"] - params["weight"]
        return torch.sqrt(zero).sum()

    with pytest.raises(FloatingPointError, match="nonfinite BGD gradient"):
        updater.step(bad_gradient)
    assert_unchanged(updater, *before)


def test_scaled_loss_overflow_does_not_advance() -> None:
    updater = make_updater()
    before = snapshot(updater)

    def large_finite_loss(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return (params["weight"] * 0.0).sum() + 1e30

    with pytest.raises(FloatingPointError, match="nonfinite BGD scaled mean"):
        updater.step(large_finite_loss, evidence_temperature=1e38)
    assert_unchanged(updater, *before)


def test_update_overflow_restores_values_and_count() -> None:
    updater = make_updater(eta=1e38, std=0.1)
    before = snapshot(updater)

    def large_gradient(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return (params["weight"] * 1e38).sum()

    with pytest.raises(FloatingPointError, match="nonfinite posterior mean"):
        updater.step(large_gradient)
    assert_unchanged(updater, *before)


def test_tempering_overflow_rolls_back_before_objective() -> None:
    updater = make_updater(std=1e-30)
    before = snapshot(updater)
    calls = 0

    def objective(params: dict[str, torch.Tensor]) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return zero_loss(params)

    with pytest.raises(FloatingPointError, match="nonfinite posterior"):
        updater.step(objective, retention=0.5)
    assert calls == 0
    assert_unchanged(updater, *before)


def test_successful_step_still_advances_and_is_finite() -> None:
    updater = make_updater()
    before = snapshot(updater)
    updater.step(lambda params: params["weight"].square().sum())
    assert updater.step_count == before[2] + 1
    updater.posterior.assert_finite()
