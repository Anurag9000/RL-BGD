import math

import pytest
import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.recurrent_agent import RecurrentSACConfig
from rl_bgd.agents.sac.recurrent_bgd_agent import (
    RecurrentBGDSACAgent,
    RecurrentBGDSACConfig,
)
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.replay.evidence_accounting import ReplayEvidenceConfig
from rl_bgd.replay.sequence_buffer import SequenceReplayBuffer
from rl_bgd.surprise.base import RetentionMappingConfig
from rl_bgd.surprise.td import AdaptiveTDRetentionConfig


def make_agent(
    mode: str = "actor_and_critic",
    *,
    adaptive: bool = False,
) -> RecurrentBGDSACAgent:
    return RecurrentBGDSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        recurrent_config=RecurrentSACConfig(
            recurrent_hidden_dim=8,
            encoder_hidden_dims=(8,),
            q_hidden_dims=(8,),
        ),
        bgd_config=RecurrentBGDSACConfig(
            bayesianization=mode,  # type: ignore[arg-type]
            posterior_std=0.1,
            replay_evidence=ReplayEvidenceConfig(
                mode="inverse_reuse_weight"
            ),
            adaptive_td_retention=(
                AdaptiveTDRetentionConfig(
                    mapping=RetentionMappingConfig(
                        lambda_min=0.6,
                        kappa=1.0,
                    )
                )
                if adaptive
                else None
            ),
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


def make_batch(agent: RecurrentBGDSACAgent):
    buffer = SequenceReplayBuffer(
        64,
        2,
        1,
        storage_device=agent.device,
    )
    generator = torch.Generator().manual_seed(201)
    for index in range(24):
        observation = torch.randn(
            2,
            generator=generator,
        )
        next_observation = torch.randn(
            2,
            generator=generator,
        )
        buffer.add(
            observation,
            torch.zeros(1),
            reward=-float(observation.square().sum().item()),
            next_observation=next_observation,
            terminated=False,
            truncated=(index in {11, 23}),
            episode_start=(index in {0, 12}),
            insertion_step=index,
        )
    return buffer.sample_sequences(
        2,
        burn_in=2,
        unroll=4,
        generator=torch.Generator().manual_seed(202),
    )


@pytest.mark.parametrize(
    "mode",
    [
        "actor_only",
        "critic_only",
        "actor_and_critic",
    ],
)
def test_recurrent_bgd_sac_modes_are_finite(mode: str) -> None:
    torch.manual_seed(203)
    agent = make_agent(mode)
    metrics = agent.update(make_batch(agent))
    assert all(
        math.isfinite(value)
        for value in metrics.values()
    )
    if mode in {"actor_only", "actor_and_critic"}:
        assert metrics["actor_sigma_mean"] > 0
    if mode in {"critic_only", "actor_and_critic"}:
        assert metrics["critic1_sigma_mean"] > 0
        assert metrics["critic2_sigma_mean"] > 0
    assert metrics["evidence_mean_usage_count"] >= 1.0


def test_recurrent_adaptive_bgd_sac_emits_retention() -> None:
    torch.manual_seed(204)
    agent = make_agent(adaptive=True)
    metrics = agent.update(make_batch(agent))
    assert 0.6 <= metrics["retention_lambda"] <= 1.0
    assert math.isfinite(metrics["surprise_raw"])


def test_recurrent_bgd_sac_checkpoint_round_trip() -> None:
    torch.manual_seed(205)
    agent = make_agent(adaptive=True)
    agent.update(make_batch(agent))
    state = agent.state_dict()

    restored = make_agent(adaptive=True)
    restored.load_state_dict(state)
    observation = torch.tensor([0.2, -0.3])
    agent.reset_recurrent_state()
    restored.reset_recurrent_state()
    torch.testing.assert_close(
        restored.act_recurrent(
            observation,
            deterministic=True,
        ),
        agent.act_recurrent(
            observation,
            deterministic=True,
        ),
    )
    assert restored.actor_posterior is not None
    assert agent.actor_posterior is not None
    for name in agent.actor_posterior.stds:
        torch.testing.assert_close(
            restored.actor_posterior.stds[name],
            agent.actor_posterior.stds[name],
        )
