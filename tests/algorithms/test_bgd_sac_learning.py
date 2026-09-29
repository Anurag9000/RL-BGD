import pytest

from rl_bgd.runners.bgd_sac_lqr import (
    run_bgd_sac_lqr,
)


@pytest.mark.slow
def test_critic_only_bgd_sac_learns_stationary_lqr() -> None:
    result = run_bgd_sac_lqr(
        steps=600,
        seed=11,
        device="cpu",
        bayesianization="critic_only",
    )
    assert (
        result["post_return"]
        > result["pre_return"] + 50.0
    )
    metrics = result["training"][
        "last_update_metrics"
    ]
    assert (
        metrics["critic1_sigma_mean"] > 0
    )
    assert (
        metrics["critic2_sigma_mean"] > 0
    )
