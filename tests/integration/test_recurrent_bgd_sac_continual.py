import torch

from rl_bgd.runners.recurrent_sac_continual_lqr import (
    run_recurrent_sac_recurring_lqr,
)


def test_recurrent_bgd_sac_recurring_stream_is_finite() -> None:
    result = run_recurrent_sac_recurring_lqr(
        steps=96,
        seed=88,
        device="cpu",
        optimizer="bgd",
    )
    metrics = result["training"]["last_update_metrics"]
    assert metrics["actor_sigma_mean"] > 0.0
    assert metrics["critic1_sigma_mean"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))


def test_recurrent_adaptive_bgd_sac_recurring_stream_emits_retention() -> None:
    result = run_recurrent_sac_recurring_lqr(
        steps=96,
        seed=89,
        device="cpu",
        optimizer="adaptive_bgd",
    )
    metrics = result["training"]["last_update_metrics"]
    assert 0.6 <= metrics["retention_lambda"] <= 1.0
    assert torch.isfinite(torch.tensor(metrics["surprise_raw"]))
    assert result["information_access"]["receives_task_boundary"] is False
