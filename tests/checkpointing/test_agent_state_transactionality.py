from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch

from rl_bgd.agents.ppo.agent import PPOAgent
from rl_bgd.agents.ppo.recurrent_agent import RecurrentPPOAgent, RecurrentPPOConfig
from rl_bgd.agents.sac.agent import SACAgent
from rl_bgd.agents.sac.recurrent_agent import RecurrentSACAgent, RecurrentSACConfig
from rl_bgd.agents.sac.task_aware_agent import TaskAwareSACAgent


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def _ppo() -> PPOAgent:
    return PPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8,),
    )


def _recurrent_ppo() -> RecurrentPPOAgent:
    return RecurrentPPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        recurrent_config=RecurrentPPOConfig(
            recurrent_hidden_dim=8,
            sequence_length=4,
            encoder_hidden_dims=(8,),
        ),
    )


def _sac() -> SACAgent:
    return SACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8,),
    )


def _recurrent_sac() -> RecurrentSACAgent:
    return RecurrentSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        recurrent_config=RecurrentSACConfig(
            recurrent_hidden_dim=8,
            encoder_hidden_dims=(8,),
            q_hidden_dims=(8,),
        ),
    )


def _task_aware_sac() -> TaskAwareSACAgent:
    return TaskAwareSACAgent(
        4,
        1,
        num_tasks=2,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8,),
    )


@pytest.mark.parametrize(
    ("factory", "late_optimizer_key"),
    [
        (_ppo, "value_optimizer"),
        (_recurrent_ppo, "value_optimizer"),
        (_sac, "alpha_optimizer"),
        (_recurrent_sac, "alpha_optimizer"),
        (_task_aware_sac, "alpha_optimizer"),
    ],
)
def test_agent_checkpoint_rejection_restores_exact_prior_state(
    factory: Any,
    late_optimizer_key: str,
) -> None:
    torch.manual_seed(1701)
    source = factory()
    payload = deepcopy(source.state_dict())
    payload[late_optimizer_key]["param_groups"] = []

    torch.manual_seed(1702)
    target = factory()
    before = deepcopy(target.state_dict())

    with pytest.raises(ValueError, match="parameter group"):
        target.load_state_dict(payload)

    _assert_nested_equal(target.state_dict(), before)
