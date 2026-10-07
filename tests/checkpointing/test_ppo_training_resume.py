from pathlib import Path

import torch

from rl_bgd.agents.ppo.agent import PPOAgent, PPOConfig
from rl_bgd.agents.ppo.train import PPOTrainConfig, train_ppo
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.utils.randomness import seed_everything


def _make() -> tuple[LinearQuadraticControlEnv, PPOAgent]:
    env = LinearQuadraticControlEnv(
        horizon=8,
        process_noise=0.01,
    )
    agent = PPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(8, 8),
        config=PPOConfig(
            actor_lr=1e-3,
            value_lr=1e-3,
            update_epochs=2,
            minibatch_size=8,
        ),
    )
    return env, agent


def test_ppo_training_checkpoint_resume_matches_uninterrupted_run(
    tmp_path: Path,
) -> None:
    config = PPOTrainConfig(
        total_steps=64,
        rollout_steps=16,
        seed=93,
    )

    seed_everything(93, deterministic=True)
    full_env, full_agent = _make()
    full_summary = train_ppo(
        full_env,
        full_agent,
        config=config,
    )

    checkpoint = tmp_path / "ppo.pt"
    seed_everything(93, deterministic=True)
    split_env, split_agent = _make()
    partial = train_ppo(
        split_env,
        split_agent,
        config=config,
        checkpoint_path=checkpoint,
        checkpoint_interval_rollouts=1,
        max_rollouts_this_call=2,
    )
    assert partial["completed"] is False
    assert partial["steps"] == 32

    resumed_env, resumed_agent = _make()
    resumed_summary = train_ppo(
        resumed_env,
        resumed_agent,
        config=config,
        resume_from=checkpoint,
    )

    assert resumed_summary["completed"] is True
    assert resumed_summary["steps"] == full_summary["steps"]
    assert resumed_summary["rollouts"] == full_summary["rollouts"]
    assert resumed_agent.update_count == full_agent.update_count
    for left, right in zip(
        resumed_agent.actor.parameters(),
        full_agent.actor.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)
    for left, right in zip(
        resumed_agent.value.parameters(),
        full_agent.value.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)
