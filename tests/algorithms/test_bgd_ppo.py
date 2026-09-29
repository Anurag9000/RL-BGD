import math

import pytest
import torch

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.bgd_agent import (
    BGDPPOAgent,
    BGDPPOConfig,
)
from rl_bgd.agents.ppo.rollout import RolloutBuffer
from rl_bgd.bayes.bgd import BGDConfig


def make_agent(
    mode: str,
    *,
    evidence_mode: str = "first_epoch_only",
) -> BGDPPOAgent:
    return BGDPPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(16, 16),
        ppo_config=PPOConfig(
            update_epochs=2,
            minibatch_size=8,
        ),
        bgd_config=BGDPPOConfig(
            bayesianization=mode,  # type: ignore[arg-type]
            posterior_std=0.1,
            evidence_mode=evidence_mode,  # type: ignore[arg-type]
            actor_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
            value_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
        ),
    )


def make_rollout(
    agent: BGDPPOAgent,
    *,
    size: int = 16,
) -> RolloutBuffer:
    torch.manual_seed(71)
    rollout = RolloutBuffer(
        size,
        2,
        1,
        device=agent.device,
    )
    observation = torch.tensor(
        [0.2, -0.4],
        device=agent.device,
    )
    for index in range(size):
        action, log_prob, value = agent.sample_action(observation)
        next_observation = torch.randn(
            2,
            device=agent.device,
        )
        rollout.add(
            observation,
            action,
            reward=-float(observation.square().sum().item()),
            terminated=False,
            truncated=(index % 8 == 7),
            value=value,
            next_value=agent.value_of(next_observation),
            log_prob=log_prob,
        )
        observation = next_observation
    return rollout


@pytest.mark.parametrize(
    "mode",
    [
        "actor_only",
        "value_only",
        "actor_and_value",
    ],
)
def test_bgd_ppo_modes_update_without_nonfinite_values(
    mode: str,
) -> None:
    torch.manual_seed(72)
    agent = make_agent(mode)
    metrics = agent.update(make_rollout(agent))
    assert all(math.isfinite(value) for value in metrics.values())
    if mode in {
        "actor_only",
        "actor_and_value",
    }:
        assert metrics["actor_sigma_mean"] > 0
    if mode in {
        "value_only",
        "actor_and_value",
    }:
        assert metrics["value_sigma_mean"] > 0


def test_bgd_ppo_checkpoint_round_trip() -> None:
    torch.manual_seed(73)
    agent = make_agent("actor_and_value")
    agent.update(make_rollout(agent))
    state = agent.state_dict()
    observation = torch.tensor([0.3, -0.1])
    expected = agent.act(
        observation,
        deterministic=True,
    )

    restored = make_agent("actor_and_value")
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


@pytest.mark.parametrize(
    (
        "evidence_mode",
        "expected_weight",
    ),
    [
        ("all_epochs", 1.0),
        ("first_epoch_only", 0.5),
        ("normalized_epochs", 0.5),
    ],
)
def test_bgd_ppo_evidence_reuse_weight_is_explicit(
    evidence_mode: str,
    expected_weight: float,
) -> None:
    torch.manual_seed(74)
    agent = make_agent(
        "actor_only",
        evidence_mode=evidence_mode,
    )
    metrics = agent.update(make_rollout(agent))
    assert metrics["uncertainty_evidence_weight_mean"] == pytest.approx(expected_weight)
