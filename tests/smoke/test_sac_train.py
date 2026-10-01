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


def test_sac_post_step_observer_runs_after_each_environment_step() -> None:
    torch.manual_seed(1)
    env = LinearQuadraticControlEnv(horizon=10)
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
    observed_steps: list[int] = []
    observed_updates: list[int] = []

    def observe_step(
        completed_steps: int,
        current_agent: SACAgent,
    ) -> None:
        observed_steps.append(completed_steps)
        observed_updates.append(current_agent.update_count)

    train_sac(
        env,
        agent,
        config=SACTrainConfig(
            total_steps=8,
            random_steps=2,
            batch_size=2,
            replay_capacity=16,
            seed=1,
        ),
        post_step_observer=observe_step,
    )
    assert observed_steps == list(range(1, 9))
    assert observed_updates[-1] == agent.update_count
    assert observed_updates[-1] > 0
