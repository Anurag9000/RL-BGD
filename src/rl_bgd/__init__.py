"""RL-BGD research package."""

from rl_bgd.bayes.bgd import BGDConfig, BGDStepResult, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior

__all__ = [
    "BGDConfig",
    "BGDStepResult",
    "BGDUpdater",
    "DiagonalGaussianPosterior",
]
__version__ = "0.1.0"
