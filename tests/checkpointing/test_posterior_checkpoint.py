import pytest
import torch
from torch import nn

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior, PosteriorBounds


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


def test_posterior_checkpoint_rejects_bounds_mismatch() -> None:
    module = nn.Linear(2, 1)
    source = DiagonalGaussianPosterior.from_module(
        module,
        prior_std=0.2,
        bounds=PosteriorBounds(sigma_min=1e-6, sigma_max=10.0),
    )
    state = source.state_dict()

    restored = DiagonalGaussianPosterior.from_module(
        nn.Linear(2, 1),
        prior_std=0.2,
        bounds=PosteriorBounds(sigma_min=1e-5, sigma_max=10.0),
    )
    with pytest.raises(ValueError, match="bounds mismatch"):
        restored.load_state_dict(state)


def test_posterior_checkpoint_rejects_parameter_dtype_mismatch() -> None:
    source_module = nn.Linear(2, 1, dtype=torch.float64)
    source = DiagonalGaussianPosterior.from_module(source_module, prior_std=0.2)
    state = source.state_dict()

    restored = DiagonalGaussianPosterior.from_module(nn.Linear(2, 1), prior_std=0.2)
    with pytest.raises(ValueError, match="parameter dtype mismatch"):
        restored.load_state_dict(state)


@pytest.mark.parametrize("bad_count", [-1, True, 1.5, "2"])
def test_updater_checkpoint_rejects_invalid_step_count(bad_count: object) -> None:
    module = nn.Linear(2, 1)
    updater = BGDUpdater(
        DiagonalGaussianPosterior.from_module(module, prior_std=0.2),
        BGDConfig(eta=0.3, mc_samples=2, antithetic=True),
    )
    state = updater.state_dict()
    state["step_count"] = bad_count
    with pytest.raises(ValueError, match="step_count must be non-negative integer"):
        updater.load_state_dict(state)


def test_updater_checkpoint_rejects_extra_config_keys() -> None:
    module = nn.Linear(2, 1)
    updater = BGDUpdater(
        DiagonalGaussianPosterior.from_module(module, prior_std=0.2),
        BGDConfig(eta=0.3, mc_samples=2, antithetic=True),
    )
    state = updater.state_dict()
    saved = state["config"]
    assert isinstance(saved, dict)
    saved["unexpected"] = 1.0
    with pytest.raises(ValueError, match="config mismatch"):
        updater.load_state_dict(state)

def _posterior_state_equal(
    left: dict[str, object],
    right: dict[str, object],
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        lhs = left[key]
        rhs = right[key]
        if isinstance(lhs, dict):
            assert isinstance(rhs, dict)
            assert lhs.keys() == rhs.keys()
            for nested_key in lhs:
                nested_lhs = lhs[nested_key]
                nested_rhs = rhs[nested_key]
                if isinstance(nested_lhs, torch.Tensor):
                    assert isinstance(nested_rhs, torch.Tensor)
                    torch.testing.assert_close(nested_lhs, nested_rhs, rtol=0.0, atol=0.0)
                else:
                    assert nested_lhs == nested_rhs
        else:
            assert lhs == rhs


@pytest.mark.parametrize("field", ["means", "stds", "prior_means", "prior_stds"])
def test_posterior_checkpoint_rejection_is_non_mutating(field: str) -> None:
    source = DiagonalGaussianPosterior.from_module(nn.Linear(2, 1), prior_std=0.2)
    target = DiagonalGaussianPosterior.from_module(nn.Linear(2, 1), prior_std=0.3)
    before = target.state_dict()
    corrupt = source.state_dict()
    mapping = corrupt[field]
    assert isinstance(mapping, dict)
    name = next(iter(mapping))
    value = mapping[name]
    assert isinstance(value, torch.Tensor)
    damaged = value.clone()
    damaged.reshape(-1)[0] = float("nan")
    mapping[name] = damaged

    with pytest.raises(ValueError, match="non-finite"):
        target.load_state_dict(corrupt)

    _posterior_state_equal(before, target.state_dict())


@pytest.mark.parametrize("invalid_version", [True, 1.0, "1"])
def test_posterior_checkpoint_version_is_not_coerced(invalid_version: object) -> None:
    posterior = DiagonalGaussianPosterior.from_module(nn.Linear(2, 1), prior_std=0.2)
    state = posterior.state_dict()
    state["version"] = invalid_version
    with pytest.raises(TypeError, match="must be an integer"):
        posterior.load_state_dict(state)


def test_posterior_checkpoint_rejects_internal_dtype_coercion() -> None:
    posterior = DiagonalGaussianPosterior.from_module(nn.Linear(2, 1), prior_std=0.2)
    state = posterior.state_dict()
    means = state["means"]
    assert isinstance(means, dict)
    name = next(iter(means))
    value = means[name]
    assert isinstance(value, torch.Tensor)
    means[name] = value.to(torch.float64)

    with pytest.raises(ValueError, match="dtype mismatch"):
        posterior.load_state_dict(state)


@pytest.mark.parametrize("invalid_std", [0.0, -1.0, 1e-8, 100.0])
def test_posterior_checkpoint_rejects_invalid_saved_std(invalid_std: float) -> None:
    posterior = DiagonalGaussianPosterior.from_module(nn.Linear(2, 1), prior_std=0.2)
    state = posterior.state_dict()
    stds = state["stds"]
    assert isinstance(stds, dict)
    name = next(iter(stds))
    value = stds[name]
    assert isinstance(value, torch.Tensor)
    stds[name] = torch.full_like(value, invalid_std)

    with pytest.raises(ValueError, match="std|bounds"):
        posterior.load_state_dict(state)


@pytest.mark.parametrize("invalid_version", [True, 1.0, "1"])
def test_bgd_updater_checkpoint_version_is_not_coerced(invalid_version: object) -> None:
    updater = BGDUpdater(
        DiagonalGaussianPosterior.from_module(nn.Linear(2, 1), prior_std=0.2),
        BGDConfig(eta=0.3, mc_samples=2, antithetic=True),
    )
    state = updater.state_dict()
    state["version"] = invalid_version
    with pytest.raises(TypeError, match="must be an integer"):
        updater.load_state_dict(state)

