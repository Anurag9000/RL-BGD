"""Regularizer checkpoints validate completely before mutating live state."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pytest
import torch
from torch import nn

from rl_bgd.baselines.ewc import EWCRegularizer, OnlineEWCRegularizer
from rl_bgd.baselines.mas import MASRegularizer
from rl_bgd.baselines.si import SynapticIntelligence


def _si(strength: float = 1.0, damping: float = 0.1) -> SynapticIntelligence:
    return SynapticIntelligence(nn.Linear(2, 1), strength=strength, damping=damping)


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (EWCRegularizer, "strength"),
        (OnlineEWCRegularizer, "strength"),
        (OnlineEWCRegularizer, "decay"),
        (MASRegularizer, "strength"),
        (_si, "strength"),
        (_si, "damping"),
    ],
)
@pytest.mark.parametrize("invalid", [True, "1.0", float("nan"), float("inf")])
def test_regularizer_checkpoint_hyperparameters_are_strict(
    factory: Callable[..., Any],
    field: str,
    invalid: object,
) -> None:
    owner = factory()
    payload = deepcopy(owner.state_dict())
    payload[field] = invalid
    before = deepcopy(owner.state_dict())

    with pytest.raises((TypeError, ValueError)):
        owner.load_state_dict(payload)

    _assert_nested_equal(owner.state_dict(), before)


def test_ewc_late_state_failure_does_not_mutate_strength() -> None:
    target = EWCRegularizer(strength=7.0)
    payload = EWCRegularizer(strength=2.0).state_dict()
    payload["states"] = [{"anchor": [], "importance": {}}]
    before = deepcopy(target.state_dict())

    with pytest.raises(TypeError):
        target.load_state_dict(payload)

    _assert_nested_equal(target.state_dict(), before)


def test_online_ewc_incomplete_state_does_not_mutate_hyperparameters() -> None:
    target = OnlineEWCRegularizer(strength=7.0, decay=0.7)
    payload = OnlineEWCRegularizer(strength=2.0, decay=0.2).state_dict()
    payload["anchor"] = {"weight": torch.ones(1)}
    payload["importance"] = None
    before = deepcopy(target.state_dict())

    with pytest.raises(ValueError, match="incomplete"):
        target.load_state_dict(payload)

    _assert_nested_equal(target.state_dict(), before)


def test_mas_incomplete_state_does_not_mutate_strength() -> None:
    target = MASRegularizer(strength=7.0)
    payload = MASRegularizer(strength=2.0).state_dict()
    payload["anchor"] = {"weight": torch.ones(1)}
    payload["importance"] = None
    before = deepcopy(target.state_dict())

    with pytest.raises(ValueError, match="incomplete"):
        target.load_state_dict(payload)

    _assert_nested_equal(target.state_dict(), before)


def test_si_late_field_failure_does_not_mutate_hyperparameters() -> None:
    target = _si(strength=7.0, damping=0.7)
    payload = _si(strength=2.0, damping=0.2).state_dict()
    payload["importance"] = []
    before = deepcopy(target.state_dict())

    with pytest.raises(TypeError, match="importance"):
        target.load_state_dict(payload)

    _assert_nested_equal(target.state_dict(), before)


def _consolidated_ewc() -> EWCRegularizer:
    owner = EWCRegularizer()
    model = nn.Linear(2, 1)
    importance = {
        name: torch.ones_like(parameter, dtype=torch.float32)
        for name, parameter in model.named_parameters()
    }
    owner.consolidate(model, importance)
    return owner


def _consolidated_online_ewc() -> OnlineEWCRegularizer:
    owner = OnlineEWCRegularizer()
    model = nn.Linear(2, 1)
    importance = {
        name: torch.ones_like(parameter, dtype=torch.float32)
        for name, parameter in model.named_parameters()
    }
    owner.consolidate(model, importance)
    return owner


def _consolidated_mas() -> MASRegularizer:
    owner = MASRegularizer()
    model = nn.Linear(2, 1)
    importance = {
        name: torch.ones_like(parameter, dtype=torch.float32)
        for name, parameter in model.named_parameters()
    }
    owner.consolidate(model, importance)
    return owner


@pytest.mark.parametrize(
    ("factory", "path"),
    [
        (_consolidated_ewc, ("states", 0, "importance")),
        (_consolidated_online_ewc, ("importance",)),
        (_consolidated_mas, ("importance",)),
        (_si, ("importance",)),
    ],
)
def test_regularizer_checkpoint_rejects_nonfinite_tensor_payload(
    factory: Callable[..., Any],
    path: tuple[object, ...],
) -> None:
    owner = factory()
    before = deepcopy(owner.state_dict())
    corrupt: Any = deepcopy(before)
    nested: Any = corrupt
    for component in path:
        nested = nested[component]
    assert isinstance(nested, dict)
    name = next(iter(nested))
    tensor = nested[name]
    assert isinstance(tensor, torch.Tensor)
    damaged = tensor.clone()
    damaged.reshape(-1)[0] = float("nan")
    nested[name] = damaged

    with pytest.raises(ValueError, match="non-finite"):
        owner.load_state_dict(corrupt)

    _assert_nested_equal(owner.state_dict(), before)
@pytest.mark.parametrize(
    ("factory", "path"),
    [
        (_consolidated_ewc, ("states", 0, "importance")),
        (_consolidated_online_ewc, ("importance",)),
        (_consolidated_mas, ("importance",)),
        (_si, ("importance",)),
    ],
)
def test_regularizer_checkpoint_rejects_negative_importance(
    factory: Callable[..., Any],
    path: tuple[object, ...],
) -> None:
    owner = factory()
    before = deepcopy(owner.state_dict())
    corrupt: Any = deepcopy(before)
    nested: Any = corrupt
    for component in path:
        nested = nested[component]
    assert isinstance(nested, dict)
    name = next(iter(nested))
    tensor = nested[name]
    assert isinstance(tensor, torch.Tensor)
    nested[name] = torch.full_like(tensor, -1.0)

    with pytest.raises(ValueError, match="non-negative"):
        owner.load_state_dict(corrupt)

    _assert_nested_equal(owner.state_dict(), before)


def test_online_ewc_checkpoint_rejects_layout_mismatch_without_mutation() -> None:
    owner = _consolidated_online_ewc()
    before = deepcopy(owner.state_dict())
    corrupt = deepcopy(before)
    importance = corrupt["importance"]
    assert isinstance(importance, dict)
    importance.pop(next(iter(importance)))

    with pytest.raises(ValueError, match="keys do not match"):
        owner.load_state_dict(corrupt)

    _assert_nested_equal(owner.state_dict(), before)


def test_si_checkpoint_rejects_tensor_dtype_coercion_without_mutation() -> None:
    owner = _si()
    before = deepcopy(owner.state_dict())
    corrupt = deepcopy(before)
    importance = corrupt["importance"]
    assert isinstance(importance, dict)
    name = next(iter(importance))
    tensor = importance[name]
    assert isinstance(tensor, torch.Tensor)
    importance[name] = tensor.to(torch.float64)

    with pytest.raises(ValueError, match="dtype mismatch"):
        owner.load_state_dict(corrupt)

    _assert_nested_equal(owner.state_dict(), before)


