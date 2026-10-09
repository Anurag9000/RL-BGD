import pytest
import torch

from rl_bgd.replay.buffer import ReplayBatch
from rl_bgd.replay.evidence_accounting import (
    ReplayEvidenceConfig,
    replay_evidence_weights,
    weighted_evidence_mean,
)


def make_batch(
    usage: list[int],
    fresh: list[bool],
) -> ReplayBatch:
    batch_size = len(usage)
    zeros = torch.zeros(batch_size, 1)
    bool_zeros = torch.zeros(batch_size, 1, dtype=torch.bool)
    return ReplayBatch(
        observations=zeros,
        actions=zeros,
        rewards=zeros,
        next_observations=zeros,
        terminated=bool_zeros,
        truncated=bool_zeros,
        transition_ids=torch.arange(batch_size).view(-1, 1),
        insertion_steps=torch.arange(batch_size).view(-1, 1),
        usage_counts=torch.tensor(usage).view(-1, 1),
        fresh=torch.tensor(fresh).view(-1, 1),
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("all_replay", [1.0, 1.0, 1.0]),
        ("fresh_only_uncertainty", [1.0, 0.0, 0.0]),
        ("inverse_reuse_weight", [1.0, 0.5, 0.25]),
    ],
)
def test_replay_evidence_weights(
    mode: str,
    expected: list[float],
) -> None:
    batch = make_batch([1, 2, 4], [True, False, False])
    summary = replay_evidence_weights(
        batch,
        ReplayEvidenceConfig(mode=mode),  # type: ignore[arg-type]
    )
    torch.testing.assert_close(
        summary.weights,
        torch.tensor(expected).view(-1, 1),
    )


def test_normalized_batch_evidence_uses_uniform_inverse_reuse_scale() -> None:
    batch = make_batch([1, 2, 4], [True, False, False])
    summary = replay_evidence_weights(
        batch,
        ReplayEvidenceConfig(mode="normalized_batch_evidence"),
    )
    expected_scale = torch.tensor([1.0, 0.5, 0.25]).mean()
    torch.testing.assert_close(
        summary.weights,
        torch.full((3, 1), expected_scale),
    )
    assert summary.effective_sample_size == pytest.approx(3.0)


def test_weighted_evidence_mean_does_not_renormalize_weight_mass() -> None:
    loss = torch.tensor([[1.0], [3.0]])
    weights = torch.tensor([[1.0], [0.0]])
    assert weighted_evidence_mean(loss, weights).item() == pytest.approx(0.5)


@pytest.mark.parametrize("mode", ["all_replay", "inverse_reuse_weight"])
def test_replay_evidence_rejects_invalid_metadata_even_in_all_replay(mode: str) -> None:
    good = make_batch([1, 2], [True, False])
    for usage, fresh, error in (
        (torch.tensor([[1.0], [2.0]]), good.fresh, "integer tensors"),
        (torch.tensor([[1], [0]]), good.fresh, ">= 1"),
        (good.usage_counts, torch.tensor([[1], [0]]), "boolean tensors"),
        (good.usage_counts, torch.tensor([[False], [False]]), "freshness disagrees"),
        (torch.empty(0, 1, dtype=torch.long), torch.empty(0, 1, dtype=torch.bool), "non-empty"),
    ):
        batch = ReplayBatch(
            observations=good.observations,
            actions=good.actions,
            rewards=good.rewards,
            next_observations=good.next_observations,
            terminated=good.terminated,
            truncated=good.truncated,
            transition_ids=good.transition_ids,
            insertion_steps=good.insertion_steps,
            usage_counts=usage,
            fresh=fresh,
        )
        with pytest.raises((TypeError, ValueError), match=error):
            replay_evidence_weights(
                batch,
                ReplayEvidenceConfig(mode=mode),  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    ("losses", "weights", "message"),
    [
        (torch.empty(0, 1), torch.empty(0, 1), "non-empty"),
        (torch.tensor([[float("nan")]]), torch.ones(1, 1), "non-finite"),
        (torch.ones(1, 1), torch.tensor([[float("inf")]]), "invalid"),
        (torch.ones(1, 1), torch.tensor([[-1.0]]), "invalid"),
    ],
)
def test_evidence_reduction_rejects_corrupt_input(
    losses: torch.Tensor,
    weights: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((ValueError, FloatingPointError), match=message):
        weighted_evidence_mean(losses, weights)
