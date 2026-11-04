import pytest

from rl_bgd.runners.recurrent_bgd_stationary_lqr import (
    run_recurrent_adam_ppo_lqr,
    run_recurrent_adam_sac_lqr,
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


def test_matched_recurrent_adam_sac_control_uses_native_budget() -> None:
    from unittest.mock import patch

    from rl_bgd.agents.sac.recurrent_agent import RecurrentSACAgent

    with (
        patch(
            "rl_bgd.runners.recurrent_bgd_stationary_lqr.evaluate_recurrent_sac",
            side_effect=[1.5, 2.5],
        ) as evaluation,
        patch(
            "rl_bgd.runners.recurrent_bgd_stationary_lqr.train_recurrent_sac",
            return_value={"completed": True},
        ) as training,
    ):
        result = run_recurrent_adam_sac_lqr(steps=80, seed=103, device="cpu")
    assert result["algorithm"] == "recurrent_adam_sac"
    assert result["pre_return"] == 1.5
    assert result["post_return"] == 2.5
    assert result["improvement"] == 1.0
    assert evaluation.call_count == 2
    agent = training.call_args.args[1]
    assert isinstance(agent, RecurrentSACAgent)
    assert agent.recurrent_config.recurrent_hidden_dim == 16
    assert agent.recurrent_config.encoder_hidden_dims == (16,)
    assert agent.recurrent_config.q_hidden_dims == (16,)
    config = training.call_args.kwargs["config"]
    assert config.total_steps == 80
    assert config.random_steps == 64
    assert config.sequence_batch_size == 4
    assert config.burn_in == 4
    assert config.unroll == 8
    assert config.replay_capacity == 2_000
    assert config.seed == 103
