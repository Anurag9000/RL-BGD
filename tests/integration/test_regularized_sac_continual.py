import math

from rl_bgd.runners.regularized_sac_continual_lqr import (
    run_regularized_sac_recurring_lqr,
    run_sac_recurring_lqr_control,
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
    assert access["receives_environment_context"] is False
    assert access["consolidation_trigger"] == "fixed_optimizer_update_interval"
    assert result["consolidation_count"] > 0
    metrics = result["training"]["last_update_metrics"]
    assert math.isfinite(metrics["actor_total_loss"])
    assert math.isfinite(metrics["critic_total_loss"])


def test_regularized_sac_and_control_share_exact_baseline_profile() -> None:
    control = run_sac_recurring_lqr_control(
        steps=80,
        seed=95,
        device="cpu",
        phase_steps=40,
        horizon=24,
    )
    regularized = run_regularized_sac_recurring_lqr(
        method="ewc",
        steps=80,
        seed=95,
        device="cpu",
        consolidation_interval_updates=8,
        phase_steps=40,
        horizon=24,
    )

    assert (
        control["benchmark_profile"]
        == regularized["benchmark_profile"]
        == "recurring_lqr_matched_v1"
    )
    assert control["phase_steps"] == regularized["phase_steps"] == 40
    assert control["horizon"] == regularized["horizon"] == 24
    assert control["final_evaluation_context"] == regularized["final_evaluation_context"]
    control_access = control["information_access"]
    assert control_access["receives_task_id"] is False
    assert control_access["receives_task_boundary"] is False
    assert control_access["receives_environment_context"] is False
    assert control_access["consolidation_trigger"] == "none"
    assert math.isfinite(float(control["training"]["final_10_mean_return"]))
