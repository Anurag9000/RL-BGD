"""FOO-VB diagonal baseline mapped onto the reusable BGD engine.

For a diagonal Gaussian posterior, the FOO-VB fixed-point equations are

    mu_i <- m_i - v_i^2 E[g_i]

    sigma_i <- v_i * (
        sqrt(1 + (0.5 * v_i * E[g_i epsilon_i])**2)
        - 0.5 * v_i * E[g_i epsilon_i]
    )

This is the same update used by BGDUpdater when eta=1 and posterior
tempering is disabled. Keeping one numerical implementation avoids silent
drift between two algebraically identical baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior


@dataclass(frozen=True)
class FOOVBDiagonalConfig:
    """Monte-Carlo controls for the exact diagonal fixed-point step."""

    mc_samples: int = 4
    antithetic: bool = True

    def validate(self) -> None:
        BGDConfig(
            eta=1.0,
            mc_samples=self.mc_samples,
            antithetic=self.antithetic,
            temper_retention=1.0,
        ).validate()


def make_foo_vb_diagonal_updater(
    posterior: DiagonalGaussianPosterior,
    config: FOOVBDiagonalConfig | None = None,
) -> BGDUpdater:
    """Construct the exact diagonal FOO-VB update through the BGD engine."""

    resolved = config or FOOVBDiagonalConfig()
    resolved.validate()
    return BGDUpdater(
        posterior,
        BGDConfig(
            eta=1.0,
            mc_samples=resolved.mc_samples,
            antithetic=resolved.antithetic,
            temper_retention=1.0,
        ),
    )


def foo_vb_bgd_equivalence_contract() -> dict[str, object]:
    """Machine-readable statement of the exact diagonal-update mapping."""

    return {
        "posterior": "diagonal_gaussian",
        "mean_step_eta": 1.0,
        "temper_retention": 1.0,
        "same_update_engine_as_bgd": True,
        "paper_scope": "online_variational_bayes_likelihood",
        "rl_scope": "generalized_bayes_surrogate",
    }
