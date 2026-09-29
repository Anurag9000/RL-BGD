import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import (
    DiagonalGaussianPosterior,
    PosteriorBounds,
)


class TwoWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(2))


def test_lower_sigma_produces_smaller_mean_movement() -> None:
    module = TwoWeights()
    posterior = DiagonalGaussianPosterior.from_module(
        module, prior_std=0.1
    )
    posterior.stds["w"].copy_(
        torch.tensor([0.05, 0.5])
    )
    updater = BGDUpdater(
        posterior,
        BGDConfig(
            eta=1.0, mc_samples=2, antithetic=True
        ),
    )
    before = posterior.means["w"].clone()

    def objective(
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return params["w"].sum()

    updater.step(objective)
    movement = (
        posterior.means["w"] - before
    ).abs()
    assert movement[1] > 50 * movement[0]
    torch.testing.assert_close(
        movement[1] / movement[0],
        torch.tensor(100.0),
        rtol=1e-4,
        atol=1e-4,
    )


def test_tempering_preserves_more_uncertainty() -> None:
    def run(retention: float) -> float:
        torch.manual_seed(9)
        module = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.zero_()
        posterior = DiagonalGaussianPosterior.from_module(
            module,
            prior_std=0.4,
            bounds=PosteriorBounds(
                sigma_min=1e-8, sigma_max=2.0
            ),
        )
        updater = BGDUpdater(
            posterior,
            BGDConfig(
                eta=0.1,
                mc_samples=8,
                antithetic=True,
                temper_retention=retention,
            ),
        )
        for _ in range(250):
            updater.step(
                lambda params: 0.5
                * params["weight"].square().sum()
            )
        return posterior.stds["weight"].item()

    vanilla = run(1.0)
    tempered = run(0.97)
    assert tempered > vanilla * 1.25
