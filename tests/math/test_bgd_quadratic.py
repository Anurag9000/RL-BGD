import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import (
    DiagonalGaussianPosterior,
    PosteriorBounds,
)


class Scalar(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([value], dtype=torch.float32))


def make_updater(
    *, value: float = 0.0, sigma: float = 0.2, eta: float = 0.2, samples: int = 400
) -> BGDUpdater:
    module = Scalar(value)
    posterior = DiagonalGaussianPosterior.from_module(
        module,
        prior_std=sigma,
        bounds=PosteriorBounds(sigma_min=1e-8, sigma_max=10.0),
    )
    return BGDUpdater(
        posterior,
        BGDConfig(eta=eta, mc_samples=samples, antithetic=True),
    )


def test_zero_gradient_does_not_change_mean_or_sigma() -> None:
    updater = make_updater(samples=8)
    before_mean = updater.posterior.means["w"].clone()
    before_std = updater.posterior.stds["w"].clone()

    def objective(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return (params["w"] * 0.0).sum()

    updater.step(objective)
    torch.testing.assert_close(updater.posterior.means["w"], before_mean)
    torch.testing.assert_close(updater.posterior.stds["w"], before_std)


def test_positive_quadratic_curvature_reduces_sigma() -> None:
    torch.manual_seed(11)
    updater = make_updater(value=0.0, sigma=0.3)
    before = updater.posterior.stds["w"].clone()

    def objective(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return 0.5 * params["w"].square().sum()

    updater.step(objective)
    assert torch.all(updater.posterior.stds["w"] < before)


def test_negative_quadratic_curvature_increases_sigma() -> None:
    torch.manual_seed(12)
    updater = make_updater(value=0.0, sigma=0.1)
    before = updater.posterior.stds["w"].clone()

    def objective(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return -0.5 * params["w"].square().sum()

    updater.step(objective)
    assert torch.all(updater.posterior.stds["w"] > before)


def test_curvature_signal_approaches_h_sigma() -> None:
    torch.manual_seed(13)
    h = 3.0
    updater = make_updater(
        value=0.0, sigma=0.05, eta=0.01, samples=4000
    )
    old_sigma = updater.posterior.stds["w"].item()
    eps = updater.posterior.sample_epsilons(samples=4000, antithetic=True)
    values = []
    for item in eps:
        theta = updater.posterior.parameters_from_epsilon(item)["w"]
        grad = h * theta.detach()
        values.append((grad.float() * item["w"]).item())
    estimate = sum(values) / len(values)
    expected = h * old_sigma
    assert abs(estimate - expected) / expected < 0.06
