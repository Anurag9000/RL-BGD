import math

import pytest
import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.bgd_agent import (
    BGDSACAgent,
    BGDSACConfig,
)
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.replay.buffer import ReplayBatch


def make_batch(
    batch_size: int = 16,
) -> ReplayBatch:
    torch.manual_seed(21)
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


def make_agent(
    mode: str,
) -> BGDSACAgent:
    return BGDSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
        ),
        bgd_config=BGDSACConfig(
            bayesianization=mode,  # type: ignore[arg-type]
            posterior_std=0.1,
            actor_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
            critic_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
        ),
    )


@pytest.mark.parametrize(
    "mode",
    [
        "critic_only",
        "actor_only",
        "actor_and_critic",
    ],
)
def test_bgd_sac_modes_update_without_nonfinite_values(
    mode: str,
) -> None:
    torch.manual_seed(22)
    agent = make_agent(mode)
    metrics = agent.update(make_batch())
    assert all(math.isfinite(value) for value in metrics.values())
    if "critic" in mode:
        assert metrics["critic1_sigma_mean"] > 0
        assert metrics["critic2_sigma_mean"] > 0
    if "actor" in mode:
        assert metrics["actor_sigma_mean"] > 0


def test_bgd_sac_checkpoint_round_trip() -> None:
    torch.manual_seed(23)
    agent = make_agent("actor_and_critic")
    agent.update(make_batch())
    state = agent.state_dict()
    observation = torch.tensor([0.3, -0.1])
    expected = agent.act(
        observation,
        deterministic=True,
    )
    restored = make_agent("actor_and_critic")
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.act(
            observation,
            deterministic=True,
        ),
        expected,
    )
    assert restored.actor_posterior is not None
    assert agent.actor_posterior is not None
    for name in agent.actor_posterior.stds:
        torch.testing.assert_close(
            restored.actor_posterior.stds[name],
            agent.actor_posterior.stds[name],
        )


def test_bgd_sac_checkpoint_rejects_bayesian_config_mismatch() -> None:
    agent = make_agent("actor_and_critic")
    state = agent.state_dict()
    restored = BGDSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
        ),
        bgd_config=BGDSACConfig(
            bayesianization="actor_and_critic",
            posterior_std=0.2,
            actor_bgd=BGDConfig(eta=0.1, mc_samples=2, antithetic=True),
            critic_bgd=BGDConfig(eta=0.1, mc_samples=2, antithetic=True),
        ),
    )
    with pytest.raises(ValueError, match="BGD-SAC checkpoint configuration mismatch"):
        restored.load_state_dict(state)
