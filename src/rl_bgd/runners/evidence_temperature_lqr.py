"""Matched generalized-Bayes evidence-temperature sweep on LQR."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

from rl_bgd.runners.bgd_sac_lqr import run_bgd_sac_lqr


def run_evidence_temperature_sweep(
    *,
    temperatures: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    steps: int = 600,
    seed: int = 101,
    device: str = "auto",
    bayesianization: str = "critic_only",
) -> dict[str, object]:
    """Run matched BGD-SAC experiments varying only generalized-Bayes power."""

    values = tuple(float(value) for value in temperatures)
    if not values:
        raise ValueError("temperature sweep requires at least one value")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("evidence temperatures must be finite and positive")

    runs: list[dict[str, object]] = []
    for temperature in values:
        result = run_bgd_sac_lqr(
            steps=steps,
            seed=seed,
            device=device,
            bayesianization=bayesianization,
            evidence_temperature=temperature,
        )
        runs.append(result)

    return {
        "study": "generalized_bayes_evidence_temperature",
        "controlled_variables": {
            "seed": seed,
            "steps": steps,
            "bayesianization": bayesianization,
            "temperature_is_only_swept_bgd_parameter": True,
        },
        "temperatures": list(values),
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
