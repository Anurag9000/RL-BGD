from pathlib import Path

import torch

from rl_bgd.agents.sac.agent import SACAgent, SACConfig
from rl_bgd.agents.sac.train import SACTrainConfig, train_sac
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.utils.randomness import seed_everything


def _make() -> tuple[LinearQuadraticControlEnv, SACAgent]:
    env = LinearQuadraticControlEnv(
        horizon=8,
        process_noise=0.01,
    )
    agent = SACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(8, 8),
        config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
    )
    return env, agent


def test_sac_training_checkpoint_resume_matches_uninterrupted_run(
    tmp_path: Path,
) -> None:
    config = SACTrainConfig(
        total_steps=48,
        random_steps=12,
        batch_size=8,
        replay_capacity=64,
        seed=91,
    )

    seed_everything(91, deterministic=True)
    full_env, full_agent = _make()
    full_summary = train_sac(
        full_env,
        full_agent,
        config=config,
    )

    checkpoint = tmp_path / "sac.pt"
    seed_everything(91, deterministic=True)
    split_env, split_agent = _make()
    partial = train_sac(
        split_env,
        split_agent,
        config=config,
        checkpoint_path=checkpoint,
        checkpoint_interval=12,
        max_steps_this_call=24,
    )
    assert partial["completed"] is False
    assert partial["steps"] == 24

    resumed_env, resumed_agent = _make()
    resumed_summary = train_sac(
        resumed_env,
        resumed_agent,
        config=config,
        resume_from=checkpoint,
    )

    assert resumed_summary["completed"] is True
    assert resumed_summary["steps"] == full_summary["steps"]
    assert resumed_summary["episodes"] == full_summary["episodes"]
    assert resumed_agent.update_count == full_agent.update_count
    for left, right in zip(
        resumed_agent.actor.parameters(),
        full_agent.actor.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)
    for left, right in zip(
        resumed_agent.critic1.parameters(),
        full_agent.critic1.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)


def test_sac_training_checkpoint_rejects_train_config_change(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "sac.pt"
    seed_everything(92, deterministic=True)
    env, agent = _make()
    train_sac(
        env,
        agent,
        config=SACTrainConfig(
            total_steps=24,
            random_steps=8,
            batch_size=8,
            replay_capacity=32,
            seed=92,
        ),
        checkpoint_path=checkpoint,
        max_steps_this_call=12,
    )

    restored_env, restored_agent = _make()
    try:
        train_sac(
            restored_env,
            restored_agent,
            config=SACTrainConfig(
                total_steps=24,
                random_steps=9,
                batch_size=8,
                replay_capacity=32,
                seed=92,
            ),
            resume_from=checkpoint,
        )
    except ValueError as exc:
        assert "configuration mismatch" in str(exc)
    else:
        raise AssertionError("resume accepted a changed SAC training config")
