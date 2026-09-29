import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior


def test_step_module_updates_and_syncs_mean() -> None:
    torch.manual_seed(1)
    module = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    posterior = DiagonalGaussianPosterior.from_module(module, prior_std=0.1)
    updater = BGDUpdater(posterior, BGDConfig(eta=0.5, mc_samples=8, antithetic=True))
    x = torch.ones(16, 1)
    y = torch.zeros(16, 1)

    def loss_fn(output: torch.Tensor) -> torch.Tensor:
        return torch.mean((output - y) ** 2)

    before = module.weight.detach().clone()
    updater.step_module(module, loss_fn, x)
    assert not torch.equal(module.weight.detach(), before)
    torch.testing.assert_close(module.weight.detach(), posterior.means["weight"])
