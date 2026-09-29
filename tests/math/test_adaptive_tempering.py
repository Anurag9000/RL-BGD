import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior


def test_retention_override_reopens_posterior_toward_prior() -> None:
    torch.manual_seed(51)
    module = nn.Linear(1, 1, bias=False)
    posterior = DiagonalGaussianPosterior.from_module(module, prior_std=1.0)
    posterior.stds["weight"].fill_(0.1)
    updater = BGDUpdater(
        posterior,
        BGDConfig(eta=0.1, mc_samples=2, antithetic=True),
    )

    def zero_objective(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return (params["weight"] * 0.0).sum()

    result = updater.step(zero_objective, retention=0.0)
    assert result.retention == 0.0
    torch.testing.assert_close(
        posterior.stds["weight"],
        torch.ones_like(posterior.stds["weight"]),
    )
