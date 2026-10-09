"""SI updates are atomic across every trainable parameter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from torch import nn

from rl_bgd.baselines.si import SynapticIntelligence


def _assert_equal(a: Any, b: Any) -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, dict):
        assert isinstance(b, dict)
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal(a[key], b[key])
    else:
        assert a == b


@pytest.mark.parametrize("corruption", ["shape", "nan"])
def test_si_accumulation_late_invalid_gradient_is_transactional(
    corruption: str,
) -> None:
    module = nn.Linear(2, 1)
    si = SynapticIntelligence(module)
    gradient = {name: torch.ones_like(p) for name, p in module.named_parameters()}
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(0.1)

    if corruption == "shape":
        gradient["bias"] = torch.ones(2)
    else:
        gradient["bias"] = torch.full_like(gradient["bias"], float("nan"))

    before = deepcopy(si.state_dict())
    with pytest.raises((ValueError, FloatingPointError)):
        si.accumulate_after_step(module, gradient)
    _assert_equal(si.state_dict(), before)


def test_si_accumulation_rejects_nonfinite_arithmetic_result() -> None:
    module = nn.Linear(2, 1)
    si = SynapticIntelligence(module)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(1e38)
    gradient = {
        name: torch.full_like(parameter, 1e38)
        for name, parameter in module.named_parameters()
    }
    before = deepcopy(si.state_dict())

    with pytest.raises(FloatingPointError, match="nonfinite SI accumulated path"):
        si.accumulate_after_step(module, gradient)

    _assert_equal(si.state_dict(), before)


@pytest.mark.parametrize("corruption", ["shape", "nan"])
def test_si_consolidation_late_corruption_is_transactional(
    corruption: str,
) -> None:
    module = nn.Linear(2, 1)
    si = SynapticIntelligence(module)
    si.path_integral["weight"] = torch.ones_like(si.path_integral["weight"])
    if corruption == "shape":
        si.path_integral["bias"] = torch.zeros(2)
    else:
        si.path_integral["bias"] = torch.full_like(
            si.path_integral["bias"], float("nan")
        )
    before = deepcopy(si.state_dict())

    with pytest.raises((ValueError, FloatingPointError)):
        si.consolidate(module)

    _assert_equal(si.state_dict(), before)


def test_si_valid_accumulation_and_consolidation_preserve_nonnegative_importance() -> None:
    module = nn.Linear(2, 1)
    si = SynapticIntelligence(module)
    gradient = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.sub_(0.1 * gradient[name])
    si.accumulate_after_step(module, gradient)
    si.consolidate(module)

    for value in si.importance.values():
        assert torch.isfinite(value).all()
        assert (value >= 0).all()
    for value in si.path_integral.values():
        assert torch.count_nonzero(value) == 0
