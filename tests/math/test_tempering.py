import torch

from rl_bgd.bayes.tempering import temper_gaussian_tensor


def test_tempering_lambda_one_returns_previous_posterior() -> None:
    mean = torch.tensor([1.0, -2.0])
    std = torch.tensor([0.4, 0.7])
    prior_mean = torch.zeros(2)
    prior_std = torch.ones(2)
    out_mean, out_std = temper_gaussian_tensor(
        mean, std, prior_mean, prior_std, retention=1.0
    )
    torch.testing.assert_close(out_mean, mean)
    torch.testing.assert_close(out_std, std)


def test_tempering_lambda_zero_returns_original_prior() -> None:
    mean = torch.tensor([1.0])
    std = torch.tensor([0.2])
    prior_mean = torch.tensor([-3.0])
    prior_std = torch.tensor([1.5])
    out_mean, out_std = temper_gaussian_tensor(
        mean, std, prior_mean, prior_std, retention=0.0
    )
    torch.testing.assert_close(out_mean, prior_mean)
    torch.testing.assert_close(out_std, prior_std)


def test_tempering_matches_precision_space_formula() -> None:
    mean = torch.tensor([2.0])
    std = torch.tensor([0.5])
    prior_mean = torch.tensor([0.0])
    prior_std = torch.tensor([2.0])
    retention = 0.25
    out_mean, out_std = temper_gaussian_tensor(
        mean, std, prior_mean, prior_std, retention=retention
    )
    tau_prev = 1 / std.square()
    tau0 = 1 / prior_std.square()
    tau = retention * tau_prev + (1 - retention) * tau0
    expected_mean = (
        retention * tau_prev * mean + (1 - retention) * tau0 * prior_mean
    ) / tau
    expected_std = torch.rsqrt(tau)
    torch.testing.assert_close(out_mean, expected_mean)
    torch.testing.assert_close(out_std, expected_std)
