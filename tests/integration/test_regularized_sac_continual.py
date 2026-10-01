import math

from rl_bgd.runners.regularized_sac_continual_lqr import (
    run_regularized_sac_recurring_lqr,
)


def test_regularized_sac_runner_does_not_receive_hidden_context() -> None:
    result = run_regularized_sac_recurring_lqr(
        method="online_ewc",
        steps=80,
        seed=94,
        device="cpu",
        consolidation_interval_updates=8,
    )
    access = result["information_access"]
    assert access["receives_task_id"] is False
    assert access["receives_task_boundary"] is False
    assert access["receives_context"] is False
    assert access["consolidation_trigger"] == "fixed_optimizer_update_interval"
    assert result["consolidation_count"] > 0
    metrics = result["training"]["last_update_metrics"]
    assert math.isfinite(metrics["actor_total_loss"])
    assert math.isfinite(metrics["critic_total_loss"])
