import pytest
import torch

from rl_bgd.replay.evidence_accounting import (
    ReplayEvidenceConfig,
)
from rl_bgd.replay.sequence_buffer import (
    SequenceReplayBatch,
)
from rl_bgd.replay.sequence_evidence import (
    sequence_replay_evidence_weights,
    weighted_sequence_evidence_mean,
)


def make_batch() -> SequenceReplayBatch:
    batch = 2
    window = 4
    zeros = torch.zeros(
        batch,
        window,
        1,
    )
    bool_zeros = torch.zeros(
        batch,
        window,
        1,
        dtype=torch.bool,
    )
    return SequenceReplayBatch(
        observations=zeros,
        actions=zeros,
        rewards=zeros,
        next_observations=zeros,
        terminated=bool_zeros,
        truncated=bool_zeros,
        episode_starts=bool_zeros,
        transition_ids=torch.zeros(
            batch,
            window,
            1,
            dtype=torch.long,
        ),
        insertion_steps=torch.zeros(
            batch,
            window,
            1,
            dtype=torch.long,
        ),
        usage_counts=torch.tensor(
            [
                [[1], [2], [4]],
                [[1], [1], [2]],
            ]
        ),
        fresh=torch.tensor(
            [
                [[True], [False], [False]],
                [[True], [True], [False]],
            ]
        ),
        burn_in=1,
    )


def test_inverse_sequence_evidence_weights_follow_usage_counts() -> None:
    summary = sequence_replay_evidence_weights(
        make_batch(),
        ReplayEvidenceConfig(mode="inverse_reuse_weight"),
    )
    expected = torch.tensor(
        [
            [[1.0], [0.5], [0.25]],
            [[1.0], [1.0], [0.5]],
        ]
    )
    torch.testing.assert_close(
        summary.weights,
        expected,
    )


def test_fresh_sequence_evidence_only_counts_first_uses() -> None:
    summary = sequence_replay_evidence_weights(
        make_batch(),
        ReplayEvidenceConfig(mode="fresh_only_uncertainty"),
    )
    assert summary.weights.sum().item() == 3.0


def test_weighted_sequence_loss_does_not_renormalize() -> None:
    losses = torch.tensor(
        [
            [[2.0], [4.0]],
        ]
    )
    weights = torch.tensor(
        [
            [[1.0], [0.0]],
        ]
    )
    assert (
        weighted_sequence_evidence_mean(
            losses,
            weights,
        ).item()
        == 1.0
    )


@pytest.mark.parametrize("mode", ["all_replay", "inverse_reuse_weight"])
def test_sequence_evidence_rejects_invalid_metadata(mode: str) -> None:
    batch = make_batch()
    for usage, fresh, message in (
        (batch.usage_counts.float(), batch.fresh, "integer tensors"),
        (torch.zeros_like(batch.usage_counts), batch.fresh, ">= 1"),
        (batch.usage_counts, batch.fresh.long(), "boolean tensors"),
        (batch.usage_counts, torch.zeros_like(batch.fresh), "freshness disagrees"),
    ):
        invalid = SequenceReplayBatch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            next_observations=batch.next_observations,
            terminated=batch.terminated,
            truncated=batch.truncated,
            episode_starts=batch.episode_starts,
            transition_ids=batch.transition_ids,
            insertion_steps=batch.insertion_steps,
            usage_counts=usage,
            fresh=fresh,
            burn_in=batch.burn_in,
        )
        with pytest.raises((TypeError, ValueError), match=message):
            sequence_replay_evidence_weights(
                invalid,
                ReplayEvidenceConfig(mode=mode),  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    ("losses", "weights", "message"),
    [
        (torch.empty(0, 1, 1), torch.empty(0, 1, 1), "non-empty"),
        (torch.tensor([[[float("nan")]]]), torch.ones(1, 1, 1), "non-finite"),
        (torch.ones(1, 1, 1), torch.tensor([[[float("inf")]]]), "invalid"),
        (torch.ones(1, 1, 1), torch.tensor([[[-1.0]]]), "invalid"),
    ],
)
def test_sequence_evidence_reduction_rejects_corrupt_input(
    losses: torch.Tensor,
    weights: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((ValueError, FloatingPointError), match=message):
        weighted_sequence_evidence_mean(losses, weights)
