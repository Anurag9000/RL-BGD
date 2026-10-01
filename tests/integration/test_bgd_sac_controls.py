from rl_bgd.runners.bgd_sac_lqr import run_bgd_sac_lqr


def test_bgd_sac_runner_exposes_independent_bayesian_controls() -> None:
    result = run_bgd_sac_lqr(
        steps=96,
        seed=111,
        device="cpu",
        bayesianization="critic_only",
        evidence_temperature=0.5,
        temper_retention=0.99,
        replay_evidence_mode="fresh_only_uncertainty",
    )
    assert result["evidence_temperature"] == 0.5
    assert result["temper_retention"] == 0.99
    assert result["replay_evidence_mode"] == "fresh_only_uncertainty"
    metrics = result["training"]["last_update_metrics"]
    assert metrics["evidence_fresh_fraction"] >= 0.0
    assert metrics["critic1_sigma_mean"] > 0.0
