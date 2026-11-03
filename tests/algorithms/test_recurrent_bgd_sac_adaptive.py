"""Regression contracts for recurrent BGD-SAC adaptive retention parity."""

from __future__ import annotations

import math
from copy import deepcopy

import pytest
import torch

from rl_bgd.agents.sac.bgd_agent import BGDSACConfig
from rl_bgd.agents.sac.recurrent_agent import RecurrentSACConfig
from rl_bgd.agents.sac.recurrent_bgd_agent import BGDRecurrentSACAgent
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.replay.sequence_buffer import SequenceReplayBatch
from rl_bgd.surprise.ensemble import AdaptiveEnsembleRetentionConfig
from rl_bgd.surprise.predictive import AdaptivePredictiveRetentionConfig
from rl_bgd.surprise.td import AdaptiveTDRetentionConfig


def _agent(source: str) -> BGDRecurrentSACAgent:
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
        bgd_config=BGDSACConfig(
            bayesianization="critic_only",
            critic_bgd=BGDConfig(eta=0.1, mc_samples=2, antithetic=True),
            adaptive_td_retention=AdaptiveTDRetentionConfig() if source == "td" else None,
            adaptive_ensemble_retention=(
                AdaptiveEnsembleRetentionConfig() if source == "ensemble" else None
            ),
            adaptive_predictive_retention=(
                AdaptivePredictiveRetentionConfig(hidden_dims=(8,))
                if source == "predictive"
                else None
            ),
        ),
    )


def _batch() -> SequenceReplayBatch:
    torch.manual_seed(117)
    count, width, burn_in = 2, 5, 2
    observation = torch.randn(count, width, 2)
    actions = torch.tanh(torch.randn(count, width, 1))
    flags = torch.zeros(count, width, 1, dtype=torch.bool)
    usage = torch.ones(count, width - burn_in, 1, dtype=torch.long)
    return SequenceReplayBatch(
        observations=observation,
        actions=actions,
        rewards=torch.randn(count, width, 1),
        next_observations=observation + 0.1,
        terminated=flags,
        truncated=flags,
        episode_starts=flags,
        transition_ids=torch.arange(count * width).reshape(count, width, 1),
        insertion_steps=torch.arange(count * width).reshape(count, width, 1),
        usage_counts=usage,
        fresh=torch.ones_like(usage, dtype=torch.bool),
        burn_in=burn_in,
    )


@pytest.mark.parametrize("source", ["none", "td", "ensemble", "predictive"])
def test_recurrent_sources_are_active_and_finite(source: str) -> None:
    torch.manual_seed(120)
    agent = _agent(source)
    metrics = agent.update(_batch())
    assert all(math.isfinite(value) for value in metrics.values())
    if source == "none":
        assert "retention_lambda" not in metrics
        assert "predictive_model_loss" not in metrics
    else:
        assert 0.0 < metrics["retention_lambda"] <= 1.0
        assert "surprise_raw" in metrics
        assert "surprise_smoothed" in metrics
    if source == "ensemble":
        assert agent.ensemble_surprise is not None
        assert agent.ensemble_surprise.normalizer.count == 1
    if source == "predictive":
        assert agent.predictive_surprise is not None
        assert agent.predictive_surprise.normalizer.count == 1
        assert math.isfinite(metrics["predictive_model_loss"])
        assert agent.predictive_optimizer is not None
        assert len(agent.predictive_optimizer.state) > 0


def test_ensemble_uses_pre_update_optimized_critic_predictions() -> None:
    torch.manual_seed(121)
    agent, batch = _agent("ensemble"), _batch()
    assert agent.ensemble_surprise is not None
    with torch.no_grad():
        q1, q2 = agent._critic_predictions(batch)
        expected = torch.stack((q1, q2)).var(dim=0, unbiased=False).mean().item()
    _, observation = agent._adaptive_retention(batch, torch.zeros_like(q1))
    assert observation is not None
    assert observation.raw == pytest.approx(expected, abs=1e-6)


def test_predictive_scores_only_unroll_and_trains_after_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, batch = _agent("predictive"), _batch()
    assert agent.predictive_model is not None
    assert agent.predictive_optimizer is not None
    observed: list[tuple[str, int]] = []
    old_nll = agent.predictive_model.negative_log_likelihood
    old_step = agent.predictive_optimizer.step

    def score(*args: torch.Tensor) -> torch.Tensor:
        observed.append(("nll", args[0].shape[0]))
        return old_nll(*args)

    def optimizer_step(*args: object, **kwargs: object) -> object:
        observed.append(("optimizer", 0))
        return old_step(*args, **kwargs)

    monkeypatch.setattr(agent.predictive_model, "negative_log_likelihood", score)
    monkeypatch.setattr(agent.predictive_optimizer, "step", optimizer_step)
    agent.update(batch)
    assert observed == [("nll", 6), ("nll", 6), ("optimizer", 0)]


def test_predictive_burn_in_does_not_enter_transition_nll() -> None:
    agent, batch = _agent("predictive"), _batch()
    first = agent._predictive_transitions(batch)
    assert first[0].shape == (6, 2)
    assert first[1].shape == (6, 1)
    assert first[2].shape == (6, 2)
    assert first[3].shape == (6, 1)
    torch.testing.assert_close(first[0], batch.observations[:, 2:].reshape(6, 2))


@pytest.mark.parametrize("source", ["td", "ensemble", "predictive"])
def test_adaptive_checkpoint_round_trip(source: str) -> None:
    torch.manual_seed(122)
    agent = _agent(source)
    agent.update(_batch())
    checkpoint = deepcopy(agent.state_dict())
    restored = _agent(source)
    restored.load_state_dict(checkpoint)
    if source == "td":
        assert restored.td_surprise is not None and agent.td_surprise is not None
        assert restored.td_surprise.state_dict() == agent.td_surprise.state_dict()
    if source == "ensemble":
        assert restored.ensemble_surprise is not None
        assert agent.ensemble_surprise is not None
        assert restored.ensemble_surprise.state_dict() == agent.ensemble_surprise.state_dict()
    if source == "predictive":
        assert restored.predictive_surprise is not None
        assert agent.predictive_surprise is not None
        assert restored.predictive_surprise.state_dict() == agent.predictive_surprise.state_dict()
        assert restored.predictive_model is not None and agent.predictive_model is not None
        for current, expected in zip(
            restored.predictive_model.parameters(),
            agent.predictive_model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(current, expected)
        assert restored.predictive_optimizer is not None
        assert agent.predictive_optimizer is not None
        assert len(restored.predictive_optimizer.state) == len(agent.predictive_optimizer.state)


@pytest.mark.parametrize("source", ["ensemble", "predictive"])
def test_adaptive_checkpoint_rejects_missing_state_atomically(source: str) -> None:
    agent = _agent(source)
    agent.update(_batch())
    checkpoint = deepcopy(agent.state_dict())
    checkpoint.pop("ensemble_surprise" if source == "ensemble" else "predictive_optimizer")
    restored = _agent(source)
    old_actor = {k: v.clone() for k, v in restored.actor.state_dict().items()}
    with pytest.raises(KeyError):
        restored.load_state_dict(checkpoint)
    for name, tensor in old_actor.items():
        torch.testing.assert_close(restored.actor.state_dict()[name], tensor)


def test_recurrent_rejects_conflicting_surprise_sources() -> None:
    with pytest.raises(ValueError, match="at most one adaptive"):
        BGDSACConfig(
            adaptive_td_retention=AdaptiveTDRetentionConfig(),
            adaptive_ensemble_retention=AdaptiveEnsembleRetentionConfig(),
        ).validate()


def test_legacy_v2_td_checkpoint_is_readable() -> None:
    original = _agent("td")
    checkpoint = deepcopy(original.state_dict())
    checkpoint["bgd_recurrent_sac_version"] = 2
    checkpoint.pop("adaptive_ensemble_retention")
    checkpoint.pop("adaptive_predictive_retention")
    restored = _agent("td")
    restored.load_state_dict(checkpoint)
    assert restored.td_surprise is not None
