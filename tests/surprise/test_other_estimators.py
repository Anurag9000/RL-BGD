import pytest
import torch

from rl_bgd.surprise.ensemble import EnsembleDisagreementSurprise
from rl_bgd.surprise.predictive import (
    AdaptivePredictiveRetentionConfig,
    GaussianTransitionModel,
    PredictiveSurprise,
)


def test_ensemble_disagreement_is_zero_for_identical_members() -> None:
    estimator = EnsembleDisagreementSurprise()
    result = estimator.observe(torch.ones(3, 5, 2))
    assert result.raw == 0.0


def test_predictive_surprise_uses_mean_nll() -> None:
    estimator = PredictiveSurprise()
    result = estimator.observe_nll(torch.tensor([1.0, 2.0, 3.0]))
    assert result.raw == 2.0

@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": float("nan")},
        {"gradient_clip_norm": float("inf")},
        {"min_log_std": float("nan")},
        {"max_log_std": float("inf")},
        {"hidden_dims": (True,)},
        {"learning_rate": "0.001"},
    ],
)
def test_predictive_config_rejects_nonfinite_or_coerced_parameters(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        AdaptivePredictiveRetentionConfig(**kwargs).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_log_std": float("nan")},
        {"max_log_std": float("inf")},
        {"observation_dim": True},
        {"hidden_dims": (1.5,)},
    ],
)
def test_gaussian_transition_model_rejects_invalid_constructor_config(
    kwargs: dict[str, object],
) -> None:
    parameters: dict[str, object] = {
        "observation_dim": 2,
        "action_dim": 1,
        "hidden_dims": (8,),
    }
    parameters.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        GaussianTransitionModel(**parameters)

