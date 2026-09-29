"""Bayesian posterior and update primitives."""

from rl_bgd.bayes.bgd import BGDConfig, BGDStepResult, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior
from rl_bgd.bayes.tempering import temper_diagonal_gaussian

__all__ = [
    "BGDConfig",
    "BGDStepResult",
    "BGDUpdater",
    "DiagonalGaussianPosterior",
    "temper_diagonal_gaussian",
]
