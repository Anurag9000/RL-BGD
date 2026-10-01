import torch
from torch import nn

from rl_bgd.baselines.foo_vb import (
    FOOVBDiagonalConfig,
    foo_vb_bgd_equivalence_contract,
    make_foo_vb_diagonal_updater,
)
from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior


def _posterior(module: nn.Module) -> DiagonalGaussianPosterior:
    return DiagonalGaussianPosterior.from_module(
        module,
        prior_std=0.2,
    )


def test_foo_vb_diagonal_matches_eta_one_bgd_exactly() -> None:
    torch.manual_seed(130)
    left = nn.Linear(2, 1, bias=True)
    right = nn.Linear(2, 1, bias=True)
    right.load_state_dict(left.state_dict())

    foo_posterior = _posterior(left)
    bgd_posterior = _posterior(right)
    foo = make_foo_vb_diagonal_updater(
        foo_posterior,
        FOOVBDiagonalConfig(
            mc_samples=4,
            antithetic=True,
        ),
    )
    bgd = BGDUpdater(
        bgd_posterior,
        BGDConfig(
            eta=1.0,
            mc_samples=4,
            antithetic=True,
            temper_retention=1.0,
        ),
    )

    def objective(params: dict[str, torch.Tensor]) -> torch.Tensor:
        prediction = torch.tensor([[0.5, -1.0]]) @ params["weight"].T + params["bias"]
        return prediction.square().mean()

    foo.step(
        objective,
        generator=torch.Generator().manual_seed(131),
    )
    bgd.step(
        objective,
        generator=torch.Generator().manual_seed(131),
    )

    for name in foo_posterior.means:
        torch.testing.assert_close(
            foo_posterior.means[name],
            bgd_posterior.means[name],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            foo_posterior.stds[name],
            bgd_posterior.stds[name],
            rtol=0.0,
            atol=0.0,
        )


def test_foo_vb_equivalence_contract_is_explicit() -> None:
    contract = foo_vb_bgd_equivalence_contract()
    assert contract["mean_step_eta"] == 1.0
    assert contract["temper_retention"] == 1.0
    assert contract["same_update_engine_as_bgd"] is True
    assert contract["rl_scope"] == "generalized_bayes_surrogate"
