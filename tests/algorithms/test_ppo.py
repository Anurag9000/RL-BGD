import math

import torch

from rl_bgd.agents.ppo.agent import (
    PPOAgent,
    PPOConfig,
)
from rl_bgd.agents.ppo.rollout import RolloutBuffer


def make_rollout(
    agent: PPOAgent,
    *,
    size: int = 32,
) -> RolloutBuffer:
    torch.manual_seed(51)
    rollout = RolloutBuffer(
        size,
        2,
        1,
        device=agent.device,
    )
    observation = torch.tensor(
        [0.3, -0.2],
        device=agent.device,
    )
    for index in range(size):
        action, log_prob, value = agent.sample_action(
            observation
        )
        next_observation = torch.randn(
            2,
            device=agent.device,
        )
        next_value = agent.value_of(
            next_observation
        )
        rollout.add(
            observation,
            action,
            reward=-float(
                observation.square().sum().item()
            ),
            terminated=False,
            truncated=(index % 8 == 7),
            value=value,
            next_value=next_value,
            log_prob=log_prob,
        )
        observation = next_observation
    return rollout


def test_ppo_action_log_prob_and_update_are_finite() -> None:
    torch.manual_seed(52)
    agent = PPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
        config=PPOConfig(
            update_epochs=2,
            minibatch_size=8,
        ),
    )
    observations = torch.randn(13, 2)
    actions, log_prob, _ = agent.actor.sample(
        observations
    )
    evaluated, _ = agent.actor.evaluate_actions(
        observations,
        actions,
    )
    torch.testing.assert_close(
        evaluated,
        log_prob,
        atol=2e-5,
        rtol=2e-5,
    )

    metrics = agent.update(
        make_rollout(agent)
    )
    assert all(
        math.isfinite(value)
        for value in metrics.values()
    )


def test_gae_bootstraps_truncation_but_stops_episode_trace() -> None:
    rollout = RolloutBuffer(
        2,
        1,
        1,
    )
    zero = torch.zeros(1)
    rollout.add(
        zero,
        zero,
        0.0,
        terminated=False,
        truncated=True,
        value=torch.tensor([1.0]),
        next_value=torch.tensor([2.0]),
        log_prob=zero,
    )
    rollout.add(
        zero,
        zero,
        0.0,
        terminated=True,
        truncated=False,
        value=torch.tensor([1.0]),
        next_value=torch.tensor([7.0]),
        log_prob=zero,
    )
    rollout.compute_gae(
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
    )
    assert rollout.advantages is not None
    torch.testing.assert_close(
        rollout.advantages,
        torch.tensor(
            [
                [1.0],
                [-1.0],
            ]
        ),
    )


def test_ppo_checkpoint_round_trip_preserves_deterministic_action() -> None:
    torch.manual_seed(53)
    kwargs = dict(
        observation_dim=2,
        action_dim=1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
    )
    agent = PPOAgent(**kwargs)
    agent.update(
        make_rollout(agent, size=16)
    )
    observation = torch.tensor([0.4, -0.6])
    expected = agent.act(
        observation,
        deterministic=True,
    )

    restored = PPOAgent(**kwargs)
    restored.load_state_dict(
        agent.state_dict()
    )
    torch.testing.assert_close(
        restored.act(
            observation,
            deterministic=True,
        ),
        expected,
    )


def test_ppo_rollout_checkpoint_round_trip() -> None:
    torch.manual_seed(54)
    agent = PPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
    )
    rollout = make_rollout(
        agent,
        size=16,
    )
    rollout.compute_gae(
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False,
    )
    restored = RolloutBuffer(
        16,
        2,
        1,
    )
    restored.load_state_dict(
        rollout.state_dict()
    )

    assert restored.size == rollout.size
    for name in (
        "observations",
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "values",
        "next_values",
        "log_probs",
    ):
        torch.testing.assert_close(
            getattr(restored, name)[: restored.size],
            getattr(rollout, name)[: rollout.size],
        )
    assert restored.advantages is not None
    assert rollout.advantages is not None
    assert restored.returns is not None
    assert rollout.returns is not None
    torch.testing.assert_close(
        restored.advantages,
        rollout.advantages,
    )
    torch.testing.assert_close(
        restored.returns,
        rollout.returns,
    )
