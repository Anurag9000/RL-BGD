import pytest

from rl_bgd.runners.recurrent_bgd_stationary_lqr import (
    run_recurrent_bgd_ppo_lqr,
    run_recurrent_bgd_sac_lqr,
)


@pytest.mark.slow
def test_recurrent_bgd_ppo_learns_stationary_lqr() -> None:
    result = run_recurrent_bgd_ppo_lqr(
        steps=1_000,
        seed=101,
        device="cpu",
    )
    assert result["post_return"] > result["pre_return"] + 0.1
    metrics = result["training"]["last_update_metrics"]
    assert metrics["actor_sigma_mean"] > 0.0
    assert metrics["value_sigma_mean"] > 0.0


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
