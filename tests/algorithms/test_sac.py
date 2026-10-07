import math

import pytest
import torch

from rl_bgd.agents.sac.agent import (
    SACAgent,
    SACConfig,
)
from rl_bgd.replay.buffer import ReplayBatch


def make_batch(
    batch_size: int = 32,
) -> ReplayBatch:
    torch.manual_seed(0)
    return ReplayBatch(
        observations=torch.randn(batch_size, 2),
        actions=(torch.rand(batch_size, 1) * 2 - 1),
        rewards=torch.randn(batch_size, 1),
        next_observations=torch.randn(batch_size, 2),
        terminated=torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
        ),
        truncated=torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
        ),
        transition_ids=torch.arange(batch_size).view(-1, 1),
        insertion_steps=torch.arange(batch_size).view(-1, 1),
        usage_counts=torch.ones(
            batch_size,
            1,
            dtype=torch.long,
        ),
        fresh=torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
        ),
    )


def test_sac_action_bounds_and_update_finite() -> None:
    torch.manual_seed(3)
    agent = SACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(32, 32),
        config=SACConfig(),
    )
    obs = torch.tensor([0.2, -0.4])
    for deterministic in (True, False):
        action = agent.act(
            obs,
            deterministic=deterministic,
        )
        assert action.shape == (1,)
        assert torch.all(action <= 1.0)
        assert torch.all(action >= -1.0)
    before = [p.detach().clone() for p in agent.critic1.parameters()]
    metrics = agent.update(make_batch())
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            before,
            agent.critic1.parameters(),
            strict=True,
        )
    )


def test_sac_checkpoint_round_trip_preserves_deterministic_action() -> None:
    torch.manual_seed(4)
    kwargs = dict(
        observation_dim=2,
        action_dim=1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
    )
    agent = SACAgent(**kwargs)
    agent.update(make_batch(batch_size=8))
    obs = torch.tensor([0.1, 0.7])
    expected = agent.act(obs, deterministic=True)
    restored = SACAgent(**kwargs)
    restored.load_state_dict(agent.state_dict())
    torch.testing.assert_close(
        restored.act(obs, deterministic=True),
        expected,
    )


def test_sac_checkpoint_rejects_config_mismatch() -> None:
    kwargs = dict(
        observation_dim=2,
        action_dim=1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
    )
    state = SACAgent(**kwargs, config=SACConfig(gamma=0.95)).state_dict()
    restored = SACAgent(**kwargs, config=SACConfig(gamma=0.99))
    with pytest.raises(ValueError, match="configuration mismatch"):
        restored.load_state_dict(state)
