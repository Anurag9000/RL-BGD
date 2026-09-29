import math

import pytest

from rl_bgd.runners.ppo_continual_lqr import (
    run_ppo_recurring_lqr,
)


@pytest.mark.parametrize(
    "optimizer",
    [
        "adam",
        "bgd",
    ],
)
def test_ppo_runs_across_task_agnostic_recurring_lqr(
    optimizer: str,
) -> None:
    result = run_ppo_recurring_lqr(
        steps=128,
        seed=81,
        device="cpu",
        optimizer=optimizer,  # type: ignore[arg-type]
    )
    assert result["environment_steps"] == 128
    assert result["training"]["episodes"] >= 4
    metrics = result["training"]["last_update_metrics"]
    assert all(math.isfinite(value) for value in metrics.values())
