"""Generalized-Bayes bookkeeping primitives.

The full RL objective integration remains outside this module until SAC/PPO surrogate
semantics are wired. This prevents arbitrary RL losses from being presented as literal
negative log likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneralizedBayesConfig:
    """Configuration for generalized posterior updates."""

    evidence_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.evidence_temperature <= 0:
            raise ValueError("evidence_temperature must be strictly positive")
