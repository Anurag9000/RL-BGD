import math

import torch

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.train import PPOTrainConfig, train_ppo
from rl_bgd.agents.ppo.ucl_agent import UCLPPOAgent
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv


def test_ucl_ppo_updates_and_checkpoint_round_trip() -> None:
    torch.manual_seed(142)
    env = LinearQuadraticControlEnv(horizon=16)
    agent = UCLPPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(8, 8),
        ppo_config=PPOConfig(
            update_epochs=2,
            minibatch_size=16,
        ),
    )
    train_ppo(
        env,
        agent,
        config=PPOTrainConfig(
            total_steps=64,
            rollout_steps=32,
            seed=142,
        ),
    )
    agent.consolidate_boundary()
    summary = train_ppo(
        env,
        agent,
        config=PPOTrainConfig(
            total_steps=64,
            rollout_steps=32,
            seed=143,
        ),
    )
    metrics = summary["last_update_metrics"]
    assert math.isfinite(metrics["actor_ucl_penalty"])
    assert math.isfinite(metrics["value_ucl_penalty"])
    assert metrics["boundary_count"] == 1.0
    assert metrics["actor_sigma_mean"] > 0.0

    state = agent.state_dict()
    restored = UCLPPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(8, 8),
        ppo_config=PPOConfig(
            update_epochs=2,
            minibatch_size=16,
        ),
    )
    restored.load_state_dict(state)
    assert restored.boundary_count == 1
    observation, _ = env.reset(seed=999)
    torch.testing.assert_close(
        restored.act(observation, deterministic=True),
        agent.act(observation, deterministic=True),
    )
