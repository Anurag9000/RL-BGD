"""Generalized-Bayes bookkeeping primitives.

BGDUpdater already applies the evidence temperature to RL surrogate objectives.
The temperature is not a literal likelihood model, and BGDConfig is the optimizer authority.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_bgd.utils.config_validation import config_finite_float


@dataclass(frozen=True)
class GeneralizedBayesConfig:
    """Configuration for generalized posterior updates."""

    evidence_temperature: float = 1.0

    def __post_init__(self) -> None:
        temperature = config_finite_float(self.evidence_temperature, name="evidence_temperature")
        if temperature <= 0:
            raise ValueError("evidence_temperature must be strictly positive")
