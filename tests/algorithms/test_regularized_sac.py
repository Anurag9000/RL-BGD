import math

import pytest
import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.regularized_agent import (
    RegularizedSACAgent,
    RegularizedSACConfig,
)
from rl_bgd.replay.buffer import ReplayBuffer


def _batch() -> object:
    replay = ReplayBuffer(
        32,
        1,
        1,
    )
    for index in range(16):
        observation = torch.tensor([0.05 * index])
        replay.add(
            observation,
            torch.tensor([0.1]),
            reward=-float(index) / 10.0,
            next_observation=observation + 0.01,
            terminated=False,
            truncated=False,
            insertion_step=index,
        )
    return replay.sample(
        8,
        generator=torch.Generator().manual_seed(3),
    )


@pytest.mark.parametrize(
    "method",
    ["ewc", "online_ewc", "mas", "si"],
)
def test_regularized_sac_fixed_update_consolidation_is_finite(
    method: str,
) -> None:
    torch.manual_seed(91)
    batch = _batch()
    agent = RegularizedSACAgent(
        1,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8, 8),
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        regularization_config=RegularizedSACConfig(
            method=method,  # type: ignore[arg-type]
            target="actor_and_critic",
            strength=0.2,
            consolidation_interval_updates=2,
            importance_samples=3,
        ),
    )
    metrics: dict[str, float] = {}
    for _ in range(4):
        metrics = agent.update(batch)
    assert agent.consolidation_count == 2
    assert metrics["consolidation_count"] == 2.0
    assert all(math.isfinite(value) for value in metrics.values())


def test_regularized_sac_checkpoint_round_trip() -> None:
    torch.manual_seed(92)
    batch = _batch()
    config = RegularizedSACConfig(
        method="online_ewc",
        target="critic_only",
        strength=0.4,
        consolidation_interval_updates=2,
        importance_samples=2,
        online_ewc_decay=0.8,
    )
    agent = RegularizedSACAgent(
        1,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8,),
        regularization_config=config,
    )
    agent.update(batch)
    agent.update(batch)
    state = agent.state_dict()

    restored = RegularizedSACAgent(
        1,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8,),
        regularization_config=config,
    )
    restored.load_state_dict(state)
    assert restored.consolidation_count == 1
    assert restored.update_count == agent.update_count
    observation = torch.tensor([0.25])
    torch.testing.assert_close(
        restored.act(observation, deterministic=True),
        agent.act(observation, deterministic=True),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "unsupported"},
        {"target": "unsupported"},
        {"target": None},
        {"strength": float("nan")},
        {"strength": float("inf")},
        {"strength": True},
        {"strength": "0.1"},
        {"strength": -0.1},
        {"consolidation_interval_updates": 1.5},
        {"consolidation_interval_updates": True},
        {"consolidation_interval_updates": 0},
        {"importance_samples": "4"},
        {"importance_samples": 0},
        {"online_ewc_decay": float("nan")},
        {"online_ewc_decay": -0.1},
        {"online_ewc_decay": 1.1},
        {"si_damping": float("inf")},
        {"si_damping": float("nan")},
        {"si_damping": False},
        {"si_damping": 0},
    ],
)
def test_regularized_sac_config_rejects_invalid_scientific_controls(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RegularizedSACConfig(**overrides).validate()


@pytest.mark.parametrize(
    "method",
    ["ewc", "online_ewc", "si", "mas"],
)
@pytest.mark.parametrize("target", ["actor_only", "critic_only", "actor_and_critic"])
def test_regularized_sac_config_accepts_supported_method_target_pairs(
    method: str,
    target: str,
) -> None:
    RegularizedSACConfig(method=method, target=target).validate()
