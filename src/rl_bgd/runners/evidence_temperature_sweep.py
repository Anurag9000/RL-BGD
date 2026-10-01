"""Matched generalized-Bayes evidence-temperature sweep on stationary LQR."""

from __future__ import annotations

import json
from collections.abc import Sequence

from rl_bgd.runners.bgd_sac_lqr import run_bgd_sac_lqr


def run_evidence_temperature_sweep(
    *,
    temperatures: Sequence[float] = (
        0.25,
        0.5,
        1.0,
        2.0,
    ),
    steps: int = 600,
    seed: int = 11,
    device: str = "auto",
    bayesianization: str = "critic_only",
) -> dict[str, object]:
    """Run matched BGD-SAC controls that vary only evidence temperature."""

    values = tuple(
        float(value)
        for value in temperatures
    )
    if not values:
        raise ValueError(
            "temperature sweep requires at least one value"
        )
    if any(value <= 0 for value in values):
        raise ValueError(
            "all evidence temperatures must be strictly positive"
        )
    if steps < 1:
        raise ValueError(
            "steps must be positive"
        )

    runs: list[dict[str, object]] = []
    for temperature in values:
        result = run_bgd_sac_lqr(
            steps=steps,
            seed=seed,
            device=device,
            bayesianization=bayesianization,
            evidence_temperature=temperature,
        )
        training = result["training"]
        if not isinstance(
            training,
            dict,
        ):
            raise TypeError(
                "BGD-SAC runner returned invalid training summary"
            )
        runs.append(
            {
                "evidence_temperature": temperature,
                "pre_return": result[
                    "pre_return"
                ],
                "post_return": result[
                    "post_return"
                ],
                "improvement": result[
                    "improvement"
                ],
                "final_10_mean_return": training[
                    "final_10_mean_return"
                ],
                "last_update_metrics": training[
                    "last_update_metrics"
                ],
            }
        )

    return {
        "study": (
            "generalized_bayes_evidence_temperature"
        ),
        "temperatures": list(values),
        "steps_per_run": steps,
        "seed": seed,
        "bayesianization": bayesianization,
        "matched_controls": {
            "same_seed": True,
            "same_environment": True,
            "same_training_budget": True,
            "same_architecture": True,
            "same_bgd_eta": True,
            "same_mc_samples": True,
            "only_temperature_varies": True,
        },
        "runs": runs,
    }


def main() -> None:
    print(
        json.dumps(
            run_evidence_temperature_sweep(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
