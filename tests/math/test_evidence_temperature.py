import pytest
import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior


def make_updater(
    temperature: float,
    *,
    eta: float = 0.1,
    mc_samples: int = 2,
) -> BGDUpdater:
    module = nn.Linear(
        1,
        1,
        bias=False,
    )
    with torch.no_grad():
        module.weight.fill_(1.0)
    posterior = DiagonalGaussianPosterior.from_module(
        module,
        prior_std=0.2,
    )
    return BGDUpdater(
        posterior,
        BGDConfig(
            eta=eta,
            mc_samples=mc_samples,
            antithetic=True,
            evidence_temperature=temperature,
        ),
    )


def quadratic_objective(
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    return 0.5 * params["weight"].square().sum()


def test_evidence_temperature_scales_bgd_gradient_signal() -> None:
    low = make_updater(0.25)
    high = make_updater(1.0)
    low_generator = torch.Generator().manual_seed(123)
    high_generator = torch.Generator().manual_seed(123)
    low_result = low.step(
        quadratic_objective,
        generator=low_generator,
    )
    high_result = high.step(
        quadratic_objective,
        generator=high_generator,
    )

    assert low_result.evidence_temperature == pytest.approx(0.25)
    assert high_result.evidence_temperature == pytest.approx(1.0)
    assert low_result.gradient_norm == pytest.approx(
        0.25 * high_result.gradient_norm,
        rel=1e-5,
    )
    assert low_result.uncertainty_gradient_norm == pytest.approx(
        0.25 * high_result.uncertainty_gradient_norm,
        rel=1e-5,
    )
    assert low_result.mean_loss == pytest.approx(
        high_result.mean_loss,
        rel=1e-6,
    )


def test_step_override_does_not_mutate_configured_temperature() -> None:
    updater = make_updater(1.0)
    result = updater.step(
        quadratic_objective,
        generator=torch.Generator().manual_seed(321),
        evidence_temperature=0.5,
    )
    assert result.evidence_temperature == pytest.approx(0.5)
    assert updater.config.evidence_temperature == pytest.approx(1.0)


def test_nonpositive_evidence_temperature_is_rejected() -> None:
    with pytest.raises(ValueError):
        BGDConfig(evidence_temperature=0.0).validate()


def test_evidence_temperature_is_not_equivalent_to_eta() -> None:
    eta_scaled = make_updater(
        1.0,
        eta=0.2,
        mc_samples=64,
    )
    temperature_scaled = make_updater(
        2.0,
        eta=0.1,
        mc_samples=64,
    )
    eta_generator = torch.Generator().manual_seed(777)
    temperature_generator = torch.Generator().manual_seed(777)

    eta_scaled.step(
        quadratic_objective,
        generator=eta_generator,
    )
    temperature_scaled.step(
        quadratic_objective,
        generator=temperature_generator,
    )

    torch.testing.assert_close(
        eta_scaled.posterior.means["weight"],
        temperature_scaled.posterior.means["weight"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.all(
        temperature_scaled.posterior.stds["weight"] < eta_scaled.posterior.stds["weight"]
    )
