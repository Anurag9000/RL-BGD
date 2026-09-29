import torch

from rl_bgd.surprise.ensemble import EnsembleDisagreementSurprise
from rl_bgd.surprise.predictive import PredictiveSurprise


def test_ensemble_disagreement_is_zero_for_identical_members() -> None:
    estimator = EnsembleDisagreementSurprise()
    result = estimator.observe(torch.ones(3, 5, 2))
    assert result.raw == 0.0


def test_predictive_surprise_uses_mean_nll() -> None:
    estimator = PredictiveSurprise()
    result = estimator.observe_nll(torch.tensor([1.0, 2.0, 3.0]))
    assert result.raw == 2.0
