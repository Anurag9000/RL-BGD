"""Checkpoint version metadata must never be numerically coerced."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from rl_bgd.baselines.ewc import EWCRegularizer, OnlineEWCRegularizer
from rl_bgd.baselines.mas import MASRegularizer
from rl_bgd.baselines.si import SynapticIntelligence
from rl_bgd.bayes.bgd import BGDConfig, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import DiagonalGaussianPosterior
from rl_bgd.surprise.base import EMASurpriseNormalizer
from rl_bgd.surprise.ensemble import EnsembleDisagreementSurprise
from rl_bgd.surprise.predictive import PredictiveSurprise
from rl_bgd.surprise.td import TDSurprise


def _bgd() -> BGDUpdater:
    module = nn.Linear(2, 1)
    posterior = DiagonalGaussianPosterior.from_module(module, prior_std=0.1)
    return BGDUpdater(posterior, BGDConfig(mc_samples=2))


def _si() -> SynapticIntelligence:
    return SynapticIntelligence(nn.Linear(2, 1))


@pytest.mark.parametrize(
    ("factory", "expected_version"),
    [
        (_bgd, 1),
        (EMASurpriseNormalizer, 2),
        (TDSurprise, 2),
        (PredictiveSurprise, 1),
        (EnsembleDisagreementSurprise, 1),
        (EWCRegularizer, 1),
        (OnlineEWCRegularizer, 1),
        (MASRegularizer, 1),
        (_si, 1),
    ],
)
@pytest.mark.parametrize("kind", ["float", "string", "bool"])
def test_primitive_checkpoint_versions_reject_coercion(
    factory: Callable[[], Any],
    expected_version: int,
    kind: str,
) -> None:
    owner = factory()
    payload = owner.state_dict()
    invalid: object
    if kind == "float":
        invalid = float(expected_version)
    elif kind == "string":
        invalid = str(expected_version)
    else:
        invalid = True
    payload["version"] = invalid

    with pytest.raises(TypeError, match="must be an integer"):
        owner.load_state_dict(payload)


@pytest.mark.parametrize("invalid", [True, 1.5, "1", -1])
def test_bgd_step_count_requires_nonnegative_integer(invalid: object) -> None:
    owner = _bgd()
    payload = owner.state_dict()
    payload["step_count"] = invalid

    with pytest.raises((TypeError, ValueError)):
        owner.load_state_dict(payload)
