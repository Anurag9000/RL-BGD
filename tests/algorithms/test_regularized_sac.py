import pytest
import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.regularized_agent import (
    RegularizedSACAgent,
    RegularizedSACConfig,
)
from rl_bgd.replay.buffer import ReplayBatch


def make_batch(batch_size: int = 8) -> ReplayBatch:
    torch.manual_seed(82)
    return ReplayBatch(
        observations=torch.randn(batch_size, 2),
        actions=torch.rand(batch_size, 1) * 2 - 1,
        rewards=torch.randn(batch_size, 1),
        next_observations=torch.randn(batch_size, 2),
        terminated=torch.zeros(batch_size, 1, dtype=torch.bool),
        truncated=torch.zeros(batch_size, 1, dtype=torch.bool),
        transition_ids=torch.arange(batch_size).view(-1, 1),
        insertion_steps=torch.arange(batch_size).view(-1, 1),
        usage_counts=torch.ones(batch_size, 1, dtype=torch.long),
        fresh=torch.ones(batch_size, 1, dtype=torch.bool),
    )


@pytest.mark.parametrize(
    "method",
    ["ewc", "online_ewc", "mas", "si"],
)
def test_regularized_sac_consolidates_and_updates(method: str) -> None:
    torch.manual_seed(83)
    agent = RegularizedSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8, 8),
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        regularizer_config=RegularizedSACConfig(
            method=method,  # type: ignore[arg-type]
            strength=0.5,
            importance_samples=4,
        ),
    )
    batch = make_batch()
    if method == "si":
        agent.update(batch)
    consolidation = agent.consolidate(batch)
    assert consolidation["consolidation_count"] == 1.0
    metrics = agent.update(batch)
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))
    assert metrics["critic_regularizer_penalty"] >= 0.0


def test_regularized_sac_checkpoint_round_trip() -> None:
    torch.manual_seed(84)
    config = RegularizedSACConfig(
        method="online_ewc",
        strength=0.7,
        importance_samples=3,
    )
    agent = RegularizedSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8, 8),
        regularizer_config=config,
    )
    agent.consolidate(make_batch())
    state = agent.state_dict()
    restored = RegularizedSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8, 8),
        regularizer_config=config,
    )
    restored.load_state_dict(state)
    assert restored.consolidation_count == agent.consolidation_count
