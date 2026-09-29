import math

import torch

from rl_bgd.agents.ppo.agent import (
    PPOAgent,
    PPOConfig,
)
from rl_bgd.agents.ppo.train import (
    PPOTrainConfig,
    train_ppo,
)
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)


def test_ppo_training_loop_runs_on_lqr() -> None:
    torch.manual_seed(54)
    env = LinearQuadraticControlEnv(
        horizon=20
    )
    agent = PPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(32, 32),
        config=PPOConfig(
            actor_lr=1e-3,
            value_lr=1e-3,
            update_epochs=2,
            minibatch_size=32,
        ),
    )
    summary = train_ppo(
        env,
        agent,
        config=PPOTrainConfig(
            total_steps=160,
            rollout_steps=64,
            seed=54,
        ),
    )
    assert summary["episodes"] >= 5
    assert math.isfinite(
        summary["last_update_metrics"][
            "policy_loss"
        ]
    )
