import math

import torch

from rl_bgd.agents.sac.agent import (
    SACAgent,
    SACConfig,
)
from rl_bgd.agents.sac.train import (
    SACTrainConfig,
    train_sac,
)
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)


def test_sac_training_loop_runs_on_lqr() -> None:
    torch.manual_seed(0)
    env = LinearQuadraticControlEnv(horizon=20)
    agent = SACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(32, 32),
        config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
    )
    summary = train_sac(
        env,
        agent,
        config=SACTrainConfig(
            total_steps=160,
            random_steps=32,
            batch_size=32,
            replay_capacity=256,
            updates_per_step=1,
            seed=0,
        ),
    )
    assert summary["episodes"] >= 5
    assert summary["replay_size"] == 160
    assert math.isfinite(summary["last_update_metrics"]["critic_loss"])
