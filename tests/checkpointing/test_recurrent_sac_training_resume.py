from pathlib import Path

import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.recurrent_agent import RecurrentSACAgent, RecurrentSACConfig
from rl_bgd.agents.sac.recurrent_train import (
    RecurrentSACTrainConfig,
    train_recurrent_sac,
)
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.utils.randomness import seed_everything


def _make() -> tuple[LinearQuadraticControlEnv, RecurrentSACAgent]:
    env = LinearQuadraticControlEnv(
        horizon=8,
        process_noise=0.01,
    )
    agent = RecurrentSACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        recurrent_config=RecurrentSACConfig(
            recurrent_hidden_dim=6,
            encoder_hidden_dims=(6,),
            q_hidden_dims=(6,),
        ),
    )
    return env, agent


def test_recurrent_sac_training_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    config = RecurrentSACTrainConfig(
        total_steps=48,
        random_steps=12,
        sequence_batch_size=2,
        burn_in=1,
        unroll=3,
        replay_capacity=64,
        seed=94,
    )

    seed_everything(94, deterministic=True)
    full_env, full_agent = _make()
    full = train_recurrent_sac(
        full_env,
        full_agent,
        config=config,
    )

    checkpoint = tmp_path / "recurrent_sac.pt"
    seed_everything(94, deterministic=True)
    split_env, split_agent = _make()
    partial = train_recurrent_sac(
        split_env,
        split_agent,
        config=config,
        checkpoint_path=checkpoint,
        max_steps_this_call=24,
    )
    assert partial["completed"] is False

    resumed_env, resumed_agent = _make()
    resumed = train_recurrent_sac(
        resumed_env,
        resumed_agent,
        config=config,
        resume_from=checkpoint,
    )

    assert resumed["steps"] == full["steps"]
    assert resumed_agent.update_count == full_agent.update_count
    assert resumed_agent.recurrent_reset_count == full_agent.recurrent_reset_count
    torch.testing.assert_close(
        resumed_agent.actor_hidden,
        full_agent.actor_hidden,
    )
    for left, right in zip(
        resumed_agent.actor.parameters(),
        full_agent.actor.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)
