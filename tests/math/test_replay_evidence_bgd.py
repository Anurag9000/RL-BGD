import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDLoss, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior


class Scalar(nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.w = nn.Parameter(
            torch.tensor(
                [value],
                dtype=torch.float32,
            )
        )


def make_updater() -> BGDUpdater:
    module = Scalar()
    posterior = (
        DiagonalGaussianPosterior.from_module(
            module,
            prior_std=0.2,
        )
    )
    return BGDUpdater(
        posterior,
        BGDConfig(
            eta=0.1,
            mc_samples=64,
            antithetic=True,
        ),
    )


def test_zero_uncertainty_evidence_preserves_sigma_but_mean_still_learns() -> None:
    torch.manual_seed(40)
    updater = make_updater()
    before_mean = (
        updater.posterior.means["w"].clone()
    )
    before_sigma = (
        updater.posterior.stds["w"].clone()
    )

    def objective(
        params: dict[str, torch.Tensor],
    ) -> BGDLoss:
        base = (
            0.5
            * params["w"]
            .square()
            .sum()
        )
        return BGDLoss(
            mean=base,
            uncertainty=base * 0.0,
        )

    updater.step(objective)
    assert torch.all(
        updater.posterior.means[
            "w"
        ].abs()
        < before_mean.abs()
    )
    torch.testing.assert_close(
        updater.posterior.stds["w"],
        before_sigma,
    )


def test_replay_downweighting_reduces_precision_accumulation() -> None:
    def run(
        inverse_reuse: bool,
    ) -> float:
        torch.manual_seed(41)
        updater = make_updater()
        for use in range(1, 21):
            weight = (
                1.0 / use
                if inverse_reuse
                else 1.0
            )
            uncertainty_weight = weight

            def objective(
                params: dict[str, torch.Tensor],
            ) -> BGDLoss:
                base = (
                    0.5
                    * params["w"]
                    .square()
                    .sum()
                )
                return BGDLoss(
                    mean=base,
                    uncertainty=(
                        base
                        * uncertainty_weight
                    ),
                )

            updater.step(objective)
        return float(
            updater.posterior.stds["w"]
            .item()
        )

    all_replay_sigma = run(False)
    corrected_sigma = run(True)
    assert (
        corrected_sigma
        > all_replay_sigma
    )
