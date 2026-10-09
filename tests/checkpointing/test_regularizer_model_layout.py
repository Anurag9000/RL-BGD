"""Reject incompatible regularizer model layouts without partial mutation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from torch import nn

from rl_bgd.baselines.ewc import EWCRegularizer, OnlineEWCRegularizer
from rl_bgd.baselines.mas import MASRegularizer
from rl_bgd.baselines.si import SynapticIntelligence


def _importance(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.ones_like(parameter, dtype=torch.float32)
        for name, parameter in module.named_parameters()
    }


def _assert_same(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            _assert_same(lhs, rhs)
    else:
        assert left == right


def test_si_restore_rejects_foreign_model_layout_without_mutation() -> None:
    target = SynapticIntelligence(nn.Linear(1, 1), strength=7.0)
    source = SynapticIntelligence(nn.Linear(1, 2), strength=2.0)
    before = deepcopy(target.state_dict())

    with pytest.raises(ValueError, match="SI checkpoint live model.*shape mismatch"):
        target.load_state_dict(source.state_dict())

    _assert_same(target.state_dict(), before)


def test_ewc_consolidate_rejects_changed_layout_without_mutation() -> None:
    first = nn.Linear(1, 1)
    other = nn.Linear(1, 2)
    owner = EWCRegularizer()
    owner.consolidate(first, _importance(first))
    before = deepcopy(owner.state_dict())

    with pytest.raises(ValueError, match="EWC consolidation.*shape mismatch"):
        owner.consolidate(other, _importance(other))

    _assert_same(owner.state_dict(), before)


def test_ewc_restore_rejects_inconsistent_consolidation_layouts() -> None:
    first = nn.Linear(1, 1)
    other = nn.Linear(1, 2)
    source = EWCRegularizer()
    source.consolidate(first, _importance(first))
    different = EWCRegularizer()
    different.consolidate(other, _importance(other))
    payload = deepcopy(source.state_dict())
    payload["states"].extend(different.state_dict()["states"])

    target = EWCRegularizer(strength=7.0)
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="EWC checkpoint consolidation.*shape mismatch"):
        target.load_state_dict(payload)

    _assert_same(target.state_dict(), before)


@pytest.mark.parametrize("regularizer_type", [OnlineEWCRegularizer, MASRegularizer])
def test_importance_accumulation_rejects_broadcastable_layout_change(
    regularizer_type: type[OnlineEWCRegularizer] | type[MASRegularizer],
) -> None:
    first = nn.Linear(1, 1)
    other = nn.Linear(1, 2)
    owner = regularizer_type()
    owner.consolidate(first, _importance(first))
    before = deepcopy(owner.state_dict())

    with pytest.raises(ValueError, match="shape mismatch"):
        owner.consolidate(other, _importance(other))

    _assert_same(owner.state_dict(), before)


@pytest.mark.parametrize("regularizer_type", [EWCRegularizer, OnlineEWCRegularizer, MASRegularizer])
def test_unchanged_model_layout_still_allows_repeated_consolidation(
    regularizer_type: type[EWCRegularizer]
    | type[OnlineEWCRegularizer]
    | type[MASRegularizer],
) -> None:
    model = nn.Linear(1, 2)
    owner = regularizer_type()
    owner.consolidate(model, _importance(model))
    owner.consolidate(model, _importance(model))

    restored = regularizer_type()
    restored.load_state_dict(owner.state_dict())
    _assert_same(restored.state_dict(), owner.state_dict())
