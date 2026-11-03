"""Fail-closed numerical config and sampled-module buffer contracts."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from rl_bgd.agents.sac.bgd_agent import BGDSACConfig
from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior
from rl_bgd.bayes.generalized_bayes import GeneralizedBayesConfig


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0, True, "1"])
def test_generalized_bayes_temperature_rejects_invalid_values(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GeneralizedBayesConfig(evidence_temperature=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0, True])
def test_sac_posterior_std_rejects_invalid_values(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BGDSACConfig(posterior_std=invalid).validate()  # type: ignore[arg-type]


def _stateful() -> tuple[nn.Module, BGDUpdater, torch.Tensor]:
    torch.manual_seed(171)
    module = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3))
    module.train()
    posterior = DiagonalGaussianPosterior.from_module(module, prior_std=0.1)
    updater = BGDUpdater(posterior, BGDConfig(eta=0.1, mc_samples=2, antithetic=True))
    return module, updater, torch.randn(8, 2)


def test_monte_carlo_forward_does_not_mutate_live_batchnorm_buffers() -> None:
    module, updater, inputs = _stateful()
    previous = {name: value.clone() for name, value in module.named_buffers()}
    result = updater.step_module(module, lambda output: output.square().mean(), inputs)
    assert math.isfinite(result.mean_loss)
    assert updater.step_count == 1
    for name, value in module.named_buffers():
        torch.testing.assert_close(value, previous[name], rtol=0, atol=0)


def test_failed_sampled_forward_preserves_module_buffers_and_posterior() -> None:
    module, updater, inputs = _stateful()
    previous = {name: value.clone() for name, value in module.named_buffers()}
    means = {name: value.clone() for name, value in updater.posterior.means.items()}
    stds = {name: value.clone() for name, value in updater.posterior.stds.items()}

    def fail_after_forward(output: torch.Tensor) -> torch.Tensor:
        assert output.shape == (8, 3)
        raise RuntimeError("injected module objective failure")

    with pytest.raises(RuntimeError, match="injected module objective failure"):
        updater.step_module(module, fail_after_forward, inputs)
    assert updater.step_count == 0
    for name, value in module.named_buffers():
        torch.testing.assert_close(value, previous[name], rtol=0, atol=0)
    for name in means:
        torch.testing.assert_close(updater.posterior.means[name], means[name], rtol=0, atol=0)
        torch.testing.assert_close(updater.posterior.stds[name], stds[name], rtol=0, atol=0)


def test_partial_module_synchronization_failure_restores_local_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, updater, inputs = _stateful()
    old_parameters = {name: value.clone() for name, value in module.named_parameters()}
    old_means = {name: value.clone() for name, value in updater.posterior.means.items()}

    def partially_sync(model: nn.Module) -> None:
        with torch.no_grad():
            first = next(model.parameters())
            first.add_(17.0)
        raise RuntimeError("injected synchronization failure")

    monkeypatch.setattr(updater.posterior, "sync_module", partially_sync)
    with pytest.raises(RuntimeError, match="injected synchronization failure"):
        updater.step_module(module, lambda output: output.square().mean(), inputs)
    assert updater.step_count == 0
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter, old_parameters[name], rtol=0, atol=0)
    for name, value in updater.posterior.means.items():
        torch.testing.assert_close(value, old_means[name], rtol=0, atol=0)
