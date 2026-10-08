from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pytest
import torch

from rl_bgd.agents.ppo.bgd_agent import BGDPPOAgent, BGDPPOConfig
from rl_bgd.agents.ppo.recurrent_agent import RecurrentPPOConfig
from rl_bgd.agents.ppo.recurrent_bgd_agent import BGDRecurrentPPOAgent
from rl_bgd.agents.ppo.ucl_agent import UCLPPOAgent
from rl_bgd.agents.sac.bgd_agent import BGDSACAgent, BGDSACConfig
from rl_bgd.agents.sac.recurrent_agent import RecurrentSACConfig
from rl_bgd.agents.sac.recurrent_bgd_agent import BGDRecurrentSACAgent
from rl_bgd.agents.sac.regularized_agent import RegularizedSACAgent, RegularizedSACConfig
from rl_bgd.bayes.bgd import BGDConfig


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


def _bgd_config_ppo() -> BGDPPOConfig:
    return BGDPPOConfig(
        bayesianization="actor_only",
        actor_bgd=BGDConfig(eta=0.1, mc_samples=2, antithetic=True),
    )


def _bgd_config_sac() -> BGDSACConfig:
    return BGDSACConfig(
        bayesianization="actor_only",
        actor_bgd=BGDConfig(eta=0.1, mc_samples=2, antithetic=True),
    )


def _bgd_ppo() -> BGDPPOAgent:
    return BGDPPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8,),
        bgd_config=_bgd_config_ppo(),
    )


def _recurrent_bgd_ppo() -> BGDRecurrentPPOAgent:
    return BGDRecurrentPPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        recurrent_config=RecurrentPPOConfig(
            recurrent_hidden_dim=8,
            sequence_length=4,
            encoder_hidden_dims=(8,),
        ),
        bgd_config=_bgd_config_ppo(),
    )


def _bgd_sac() -> BGDSACAgent:
    return BGDSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8,),
        bgd_config=_bgd_config_sac(),
    )


def _recurrent_bgd_sac() -> BGDRecurrentSACAgent:
    return BGDRecurrentSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        recurrent_config=RecurrentSACConfig(
            recurrent_hidden_dim=8,
            encoder_hidden_dims=(8,),
            q_hidden_dims=(8,),
        ),
        bgd_config=_bgd_config_sac(),
    )


def _regularized_sac() -> RegularizedSACAgent:
    return RegularizedSACAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8,),
        regularization_config=RegularizedSACConfig(
            method="ewc",
            target="actor_only",
        ),
    )


def _ucl_ppo() -> UCLPPOAgent:
    return UCLPPOAgent(
        2,
        1,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        hidden_dims=(8,),
    )


def _missing_actor_bgd(payload: dict[str, Any]) -> None:
    del payload["actor_bgd"]


def _bad_regularizer(payload: dict[str, Any]) -> None:
    payload["regularized_sac"]["actor_regularizer"] = []


def _bad_ucl_snapshot(payload: dict[str, Any]) -> None:
    payload["value_snapshot"] = [{}]


@pytest.mark.parametrize(
    ("factory", "corrupt", "error_type"),
    [
        (_bgd_ppo, _missing_actor_bgd, KeyError),
        (_recurrent_bgd_ppo, _missing_actor_bgd, KeyError),
        (_bgd_sac, _missing_actor_bgd, KeyError),
        (_recurrent_bgd_sac, _missing_actor_bgd, KeyError),
        (_regularized_sac, _bad_regularizer, TypeError),
        (_ucl_ppo, _bad_ucl_snapshot, KeyError),
    ],
)
def test_derived_agent_checkpoint_rejection_restores_exact_prior_state(
    factory: Callable[[], Any],
    corrupt: Callable[[dict[str, Any]], None],
    error_type: type[Exception],
) -> None:
    torch.manual_seed(1801)
    source = factory()
    payload = deepcopy(source.state_dict())
    corrupt(payload)

    torch.manual_seed(1802)
    target = factory()
    before = deepcopy(target.state_dict())

    with pytest.raises(error_type):
        target.load_state_dict(payload)

    _assert_nested_equal(target.state_dict(), before)
