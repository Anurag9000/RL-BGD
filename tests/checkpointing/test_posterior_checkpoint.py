import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior


def test_posterior_and_updater_state_round_trip() -> None:
    torch.manual_seed(2)
    module = nn.Linear(2, 1)
    posterior = DiagonalGaussianPosterior.from_module(module, prior_std=0.2)
    updater = BGDUpdater(posterior, BGDConfig(eta=0.3, mc_samples=2, antithetic=True))

    def objective(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(value.square().sum() for value in params.values())

    updater.step(objective)
    state = updater.state_dict()

    module2 = nn.Linear(2, 1)
    posterior2 = DiagonalGaussianPosterior.from_module(module2, prior_std=0.2)
    updater2 = BGDUpdater(posterior2, BGDConfig(eta=0.3, mc_samples=2, antithetic=True))
    updater2.load_state_dict(state)

    assert updater2.step_count == updater.step_count
    for name in posterior.means:
        torch.testing.assert_close(posterior2.means[name], posterior.means[name])
        torch.testing.assert_close(posterior2.stds[name], posterior.stds[name])
