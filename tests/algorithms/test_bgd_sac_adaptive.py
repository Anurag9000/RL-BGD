import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.bgd_agent import BGDSACAgent, BGDSACConfig
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.replay.buffer import ReplayBatch
from rl_bgd.surprise.base import EMANormalizerConfig, RetentionMappingConfig
from rl_bgd.surprise.td import AdaptiveTDRetentionConfig, TDSurpriseConfig


def make_batch(batch_size: int = 16) -> ReplayBatch:
    torch.manual_seed(52)
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


def make_agent() -> BGDSACAgent:
    adaptive = AdaptiveTDRetentionConfig(
        surprise=TDSurpriseConfig(
            normalizer=EMANormalizerConfig(
                decay=0.9,
                smoothing_decay=0.5,
                initial_variance=0.1,
            )
        ),
        mapping=RetentionMappingConfig(lambda_min=0.4, kappa=1.5),
    )
    return BGDSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
        sac_config=SACConfig(),
        bgd_config=BGDSACConfig(
            bayesianization="critic_only",
            posterior_std=0.1,
            adaptive_td_retention=adaptive,
            critic_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
        ),
    )


def test_adaptive_td_retention_logs_boundary_free_surprise() -> None:
    agent = make_agent()
    first = agent.update(make_batch())
    second_batch = make_batch()
    second_batch.rewards.add_(10.0)
    second = agent.update(second_batch)
    assert 0.4 <= first["retention_lambda"] <= 1.0
    assert 0.4 <= second["retention_lambda"] <= 1.0
    assert second["surprise_normalized"] >= 0.0
    assert agent.td_surprise is not None
    assert agent.td_surprise.normalizer.count == 2


def test_adaptive_td_surprise_checkpoint_round_trip() -> None:
    agent = make_agent()
    agent.update(make_batch())
    state = agent.state_dict()
    restored = make_agent()
    restored.load_state_dict(state)
    assert restored.td_surprise is not None
    assert agent.td_surprise is not None
    assert restored.td_surprise.normalizer.count == agent.td_surprise.normalizer.count
