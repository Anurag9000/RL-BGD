import torch
from torch import nn

from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior


def test_antithetic_epsilons_cancel_pairwise() -> None:
    module = nn.Linear(3, 2)
    posterior = DiagonalGaussianPosterior.from_module(module, prior_std=0.1)
    torch.manual_seed(7)
    eps = posterior.sample_epsilons(samples=8, antithetic=True)
    for i in range(0, 8, 2):
        for name in eps[i]:
            torch.testing.assert_close(
                eps[i][name] + eps[i + 1][name],
                torch.zeros_like(eps[i][name]),
            )
