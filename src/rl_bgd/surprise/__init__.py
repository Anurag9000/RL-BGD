"""Task-boundary-free surprise estimators and retention mappings."""

from rl_bgd.surprise.base import (
    EMANormalizerConfig,
    EMASurpriseNormalizer,
    RetentionMappingConfig,
    SurpriseObservation,
    surprise_to_retention,
)
from rl_bgd.surprise.ensemble import EnsembleDisagreementSurprise
from rl_bgd.surprise.predictive import PredictiveSurprise
from rl_bgd.surprise.td import (
    AdaptiveTDRetentionConfig,
    TDSurprise,
    TDSurpriseConfig,
)

__all__ = [
    "AdaptiveTDRetentionConfig",
    "EMANormalizerConfig",
    "EMASurpriseNormalizer",
    "EnsembleDisagreementSurprise",
    "PredictiveSurprise",
    "RetentionMappingConfig",
    "SurpriseObservation",
    "TDSurprise",
    "TDSurpriseConfig",
    "surprise_to_retention",
]
