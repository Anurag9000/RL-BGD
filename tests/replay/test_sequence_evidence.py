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
    summary = (
        sequence_replay_evidence_weights(
            make_batch(),
            ReplayEvidenceConfig(
                mode="inverse_reuse_weight"
            ),
        )
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
    summary = (
        sequence_replay_evidence_weights(
            make_batch(),
            ReplayEvidenceConfig(
                mode="fresh_only_uncertainty"
            ),
        )
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
