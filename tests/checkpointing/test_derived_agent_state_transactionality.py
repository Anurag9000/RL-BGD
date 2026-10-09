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
        (_ucl_ppo, _bad_ucl_snapshot, ValueError),
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


@pytest.mark.parametrize(
    ("factory", "version_key"),
    [
        (_recurrent_bgd_ppo, "bgd_recurrent_ppo_version"),
        (_bgd_sac, "bgd_sac_version"),
        (_recurrent_bgd_sac, "bgd_recurrent_sac_version"),
        (_ucl_ppo, "version"),
    ],
)
@pytest.mark.parametrize("invalid_version", [True, 2.0, "2"])
def test_derived_checkpoint_versions_are_not_coerced(
    factory: Callable[[], Any],
    version_key: str,
    invalid_version: object,
) -> None:
    agent = factory()
    payload = deepcopy(agent.state_dict())
    expected = payload[version_key]
    if expected == 1:
        invalid_version = (
            True
            if invalid_version is True
            else (1.0 if isinstance(invalid_version, float) else "1")
        )
    payload[version_key] = invalid_version

    with pytest.raises(TypeError, match="must be an integer"):
        agent.load_state_dict(payload)

@pytest.mark.parametrize(
    "flag_name",
    [
        "adaptive_td_retention",
        "adaptive_ensemble_retention",
        "adaptive_predictive_retention",
    ],
)
def test_bgd_sac_adaptive_flags_require_real_booleans(flag_name: str) -> None:
    agent = _bgd_sac()
    payload = deepcopy(agent.state_dict())
    payload[flag_name] = 0

    with pytest.raises(TypeError, match="must be a boolean"):
        agent.load_state_dict(payload)


def test_recurrent_bgd_sac_adaptive_flag_requires_real_boolean() -> None:
    agent = _recurrent_bgd_sac()
    payload = deepcopy(agent.state_dict())
    payload["adaptive_td_retention"] = "false"

    with pytest.raises(TypeError, match="must be a boolean"):
        agent.load_state_dict(payload)


@pytest.mark.parametrize("counter_name", ["boundary_count", "update_count"])
@pytest.mark.parametrize("invalid_count", [True, 1.5, "1", -1])
def test_ucl_checkpoint_counters_are_strict(
    counter_name: str,
    invalid_count: object,
) -> None:
    agent = _ucl_ppo()
    payload = deepcopy(agent.state_dict())
    payload[counter_name] = invalid_count

    with pytest.raises((TypeError, ValueError)):
        agent.load_state_dict(payload)


@pytest.mark.parametrize("invalid_count", [True, 1.5, "1", -1])
def test_regularized_sac_consolidation_count_is_strict(invalid_count: object) -> None:
    agent = _regularized_sac()
    payload = deepcopy(agent.state_dict())
    payload["regularized_sac"]["consolidation_count"] = invalid_count

    with pytest.raises((TypeError, ValueError)):
        agent.load_state_dict(payload)


@pytest.mark.parametrize("invalid_version", [True, 1.0, "1"])
def test_regularized_sac_version_is_not_coerced(invalid_version: object) -> None:
    agent = _regularized_sac()
    payload = deepcopy(agent.state_dict())
    payload["regularized_sac"]["version"] = invalid_version

    with pytest.raises(TypeError, match="must be an integer"):
        agent.load_state_dict(payload)



def test_ucl_checkpoint_rejects_nonfinite_snapshot_without_mutation() -> None:
    torch.manual_seed(2001)
    source = _ucl_ppo()
    payload = deepcopy(source.state_dict())
    actor_snapshot = payload["actor_snapshot"]
    assert isinstance(actor_snapshot, list)
    weight_mu = actor_snapshot[0]["weight_mu"]
    assert isinstance(weight_mu, torch.Tensor)
    damaged = weight_mu.clone()
    damaged.reshape(-1)[0] = float("nan")
    actor_snapshot[0]["weight_mu"] = damaged

    torch.manual_seed(2002)
    target = _ucl_ppo()
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="non-finite"):
        target.load_state_dict(payload)

    _assert_nested_equal(target.state_dict(), before)


def test_ucl_checkpoint_rejects_nonpositive_snapshot_sigma() -> None:
    agent = _ucl_ppo()
    payload = deepcopy(agent.state_dict())
    value_snapshot = payload["value_snapshot"]
    assert isinstance(value_snapshot, list)
    sigma = value_snapshot[0]["weight_sigma"]
    assert isinstance(sigma, torch.Tensor)
    value_snapshot[0]["weight_sigma"] = torch.zeros_like(sigma)

    with pytest.raises(ValueError, match="strictly positive"):
        agent.load_state_dict(payload)


def test_ucl_checkpoint_rejects_snapshot_layer_count_mismatch() -> None:
    agent = _ucl_ppo()
    payload = deepcopy(agent.state_dict())
    actor_snapshot = payload["actor_snapshot"]
    assert isinstance(actor_snapshot, list)
    actor_snapshot.clear()

    with pytest.raises(ValueError, match="layer count mismatch"):
        agent.load_state_dict(payload)
