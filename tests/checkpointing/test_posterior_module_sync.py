"""Posterior/module sync must reject bad layouts and float conversion overflow."""

import pytest
import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior


class HalfScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0], dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight * x).sum()


def test_sync_rejects_late_fp16_cast_overflow_without_partial_copy() -> None:
    module = nn.Linear(1, 1, dtype=torch.float16)
    posterior = DiagonalGaussianPosterior.from_module(module, prior_std=0.2)
    original = {name: value.detach().clone() for name, value in module.named_parameters()}
    posterior.means["weight"].fill_(1.0)
    posterior.means["bias"].fill_(1e10)

    with pytest.raises(FloatingPointError, match="after conversion for bias"):
        posterior.sync_module(module)
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)


def test_sync_rejects_incompatible_shape_without_mutating_model() -> None:
    source = nn.Linear(1, 1)
    target = nn.Linear(2, 1)
    posterior = DiagonalGaussianPosterior.from_module(source)
    original = {name: value.detach().clone() for name, value in target.named_parameters()}

    with pytest.raises(ValueError, match="shape mismatch"):
        posterior.sync_module(target)
    for name, parameter in target.named_parameters():
        torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)


def test_sync_rejects_dtype_mismatch_without_mutating_model() -> None:
    source = nn.Linear(1, 1, bias=False)
    target = nn.Linear(1, 1, bias=False, dtype=torch.float16)
    posterior = DiagonalGaussianPosterior.from_module(source)
    old = target.weight.detach().clone()

    with pytest.raises(ValueError, match="dtype mismatch"):
        posterior.sync_module(target)
    torch.testing.assert_close(target.weight, old, atol=0, rtol=0)


def test_step_module_rolls_back_posterior_when_fp16_sync_overflows() -> None:
    module = HalfScale()
    posterior = DiagonalGaussianPosterior.from_module(module, prior_std=0.2)
    updater = BGDUpdater(posterior, BGDConfig(eta=1e8, mc_samples=2, antithetic=True))
    old_mean = posterior.means["weight"].clone()
    old_std = posterior.stds["weight"].clone()
    old_weight = module.weight.detach().clone()
    old_steps = updater.step_count

    with pytest.raises(FloatingPointError, match="after conversion"):
        updater.step_module(
            module,
            lambda output: output,
            torch.ones(1, dtype=torch.float16),
            generator=torch.Generator().manual_seed(21),
        )

    torch.testing.assert_close(posterior.means["weight"], old_mean, atol=0, rtol=0)
    torch.testing.assert_close(posterior.stds["weight"], old_std, atol=0, rtol=0)
    torch.testing.assert_close(module.weight, old_weight, atol=0, rtol=0)
    assert updater.step_count == old_steps
