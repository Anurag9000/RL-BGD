import pytest
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


@pytest.mark.parametrize(
    "mismatched_config",
    [
        BGDConfig(eta=0.4, mc_samples=2, antithetic=True),
        BGDConfig(eta=0.3, mc_samples=4, antithetic=True),
        BGDConfig(eta=0.3, mc_samples=2, antithetic=False),
        BGDConfig(
            eta=0.3,
            mc_samples=2,
            antithetic=True,
            temper_retention=0.95,
        ),
        BGDConfig(
            eta=0.3,
            mc_samples=2,
            antithetic=True,
            evidence_temperature=0.5,
        ),
    ],
)
def test_updater_checkpoint_rejects_scientific_config_mismatch(
    mismatched_config: BGDConfig,
) -> None:
    module = nn.Linear(2, 1)
    updater = BGDUpdater(
        DiagonalGaussianPosterior.from_module(module, prior_std=0.2),
        BGDConfig(eta=0.3, mc_samples=2, antithetic=True),
    )
    state = updater.state_dict()

    restored_module = nn.Linear(2, 1)
    restored = BGDUpdater(
        DiagonalGaussianPosterior.from_module(restored_module, prior_std=0.2),
        mismatched_config,
    )
    with pytest.raises(ValueError, match="checkpoint config mismatch"):
        restored.load_state_dict(state)


def test_updater_checkpoint_requires_saved_config() -> None:
    module = nn.Linear(2, 1)
    updater = BGDUpdater(
        DiagonalGaussianPosterior.from_module(module, prior_std=0.2),
        BGDConfig(eta=0.3, mc_samples=2, antithetic=True),
    )
    state = updater.state_dict()
    del state["config"]

    with pytest.raises(ValueError, match="missing its configuration"):
        updater.load_state_dict(state)
