from rl_bgd.runners.bgd_ppo_lqr import run_bgd_ppo_lqr


def test_bgd_ppo_learns_stationary_lqr() -> None:
    result = run_bgd_ppo_lqr(
        steps=800,
        seed=29,
        device="cpu",
        bayesianization="actor_and_value",
    )
    assert result["post_return"] > result["pre_return"] + 0.25
    metrics = result["training"]["last_update_metrics"]
    assert metrics["actor_sigma_mean"] > 0
    assert metrics["value_sigma_mean"] > 0
