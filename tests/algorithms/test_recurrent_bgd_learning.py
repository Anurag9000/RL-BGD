import pytest

from rl_bgd.runners.recurrent_bgd_stationary_lqr import (
    run_recurrent_adam_ppo_lqr,
    run_recurrent_bgd_ppo_lqr,
    run_recurrent_bgd_sac_lqr,
)


@pytest.mark.slow
def test_matched_recurrent_adam_ppo_learns_control_requiring_lqr() -> None:
    result = run_recurrent_adam_ppo_lqr(
        steps=2_000,
        seed=100,
        device="cpu",
    )
    assert result["post_return"] > result["pre_return"] + 1.0


@pytest.mark.slow
def test_recurrent_bgd_ppo_learns_control_requiring_lqr() -> None:
    result = run_recurrent_bgd_ppo_lqr(
        steps=2_000,
        seed=101,
        device="cpu",
        bayesianization="actor_only",
    )
    assert result["post_return"] > result["pre_return"] + 1.0
    metrics = result["training"]["last_update_metrics"]
    assert metrics["actor_sigma_mean"] > 0.0
    assert "value_sigma_mean" not in metrics


@pytest.mark.slow
def test_recurrent_bgd_sac_learns_stationary_lqr() -> None:
    result = run_recurrent_bgd_sac_lqr(
        steps=800,
        seed=102,
        device="cpu",
    )
    assert result["post_return"] > result["pre_return"] + 10.0
    metrics = result["training"]["last_update_metrics"]
    assert metrics["critic1_sigma_mean"] > 0.0
    assert metrics["critic2_sigma_mean"] > 0.0
