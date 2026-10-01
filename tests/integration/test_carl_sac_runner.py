import math

import pytest

from rl_bgd.runners.carl_pendulum_sac import run_carl_pendulum_sac


@pytest.mark.benchmark
def test_live_carl_sac_runner_is_strict_task_agnostic() -> None:
    pytest.importorskip("carl")
    result = run_carl_pendulum_sac(
        mode="recurring",
        optimizer="adam",
        steps=32,
        phase_steps=12,
        seed=160,
        device="cpu",
    )
    access = result["information_access"]
    assert access["receives_task_id"] is False
    assert access["receives_task_boundary"] is False
    assert access["receives_context"] is False
    assert access["context_hidden_by_adapter"] is True
    assert math.isfinite(
        result["training"]["last_update_metrics"]["critic_loss"]
    )
