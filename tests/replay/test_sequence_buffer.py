import pytest
import torch

from rl_bgd.replay.sequence_buffer import (
    SequenceReplayBuffer,
)


def add_transition(
    buffer: SequenceReplayBuffer,
    index: int,
    *,
    episode_start: bool = False,
    terminated: bool = False,
    truncated: bool = False,
) -> None:
    value = torch.tensor([float(index)])
    buffer.add(
        value,
        torch.tensor([0.0]),
        float(index),
        value + 1.0,
        terminated=terminated,
        truncated=truncated,
        episode_start=episode_start,
        insertion_step=index,
    )


def test_sequence_replay_survives_ring_wrap_in_chronological_order() -> None:
    buffer = SequenceReplayBuffer(
        5,
        1,
        1,
    )
    for index in range(8):
        add_transition(
            buffer,
            index,
            episode_start=(index in {0, 4}),
        )
    assert buffer.logical_transition_ids().reshape(-1).tolist() == [
        3,
        4,
        5,
        6,
        7,
    ]

    batch = buffer.sample_sequences(
        1,
        burn_in=1,
        unroll=3,
        generator=torch.Generator().manual_seed(0),
    )
    ids = batch.transition_ids[
        0,
        :,
        0,
    ]
    assert torch.all(ids[1:] - ids[:-1] == 1)


def test_burn_in_does_not_count_as_bayesian_evidence() -> None:
    buffer = SequenceReplayBuffer(
        8,
        1,
        1,
    )
    for index in range(8):
        add_transition(
            buffer,
            index,
            episode_start=(index == 0),
        )
    batch = buffer.sample_sequences(
        1,
        burn_in=2,
        unroll=3,
        generator=torch.Generator().manual_seed(1),
    )
    assert batch.usage_counts.shape == (
        1,
        3,
        1,
    )
    assert torch.all(batch.usage_counts == 1)
    burn_ids = batch.transition_ids[
        0,
        :2,
        0,
    ]
    physical_ids = buffer.transition_ids[
        :,
        0,
    ]
    for transition_id in burn_ids:
        index = torch.nonzero(
            physical_ids == transition_id,
            as_tuple=False,
        ).reshape(-1)
        assert (
            buffer.usage_counts[
                index,
                0,
            ].item()
            == 0
        )


def test_sequence_bootstrap_mask_preserves_truncation() -> None:
    buffer = SequenceReplayBuffer(
        6,
        1,
        1,
    )
    for index in range(6):
        add_transition(
            buffer,
            index,
            episode_start=(index == 0),
            terminated=(index == 4),
            truncated=(index == 3),
        )
    batch = buffer.sample_sequences(
        1,
        burn_in=0,
        unroll=6,
        generator=torch.Generator().manual_seed(2),
    )
    assert (
        batch.bootstrap_mask[
            0,
            3,
            0,
        ].item()
        == 1.0
    )
    assert (
        batch.bootstrap_mask[
            0,
            4,
            0,
        ].item()
        == 0.0
    )


def test_sequence_replay_checkpoint_round_trip() -> None:
    buffer = SequenceReplayBuffer(
        6,
        1,
        1,
    )
    for index in range(6):
        add_transition(
            buffer,
            index,
            episode_start=(index in {0, 3}),
            truncated=(index == 2),
        )
    buffer.sample_sequences(
        1,
        burn_in=1,
        unroll=2,
        generator=torch.Generator().manual_seed(3),
    )
    state = buffer.state_dict()
    restored = SequenceReplayBuffer(
        6,
        1,
        1,
    )
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.logical_transition_ids(),
        buffer.logical_transition_ids(),
    )
    torch.testing.assert_close(
        restored.usage_counts,
        buffer.usage_counts,
    )


def test_sequence_replay_checkpoint_rejects_inconsistent_provenance() -> None:
    buffer = SequenceReplayBuffer(6, 1, 1)
    for index in range(3):
        add_transition(
            buffer,
            index,
            episode_start=(index == 0),
        )
    state = buffer.state_dict()
    state["next_transition_id"] = 9
    with pytest.raises(ValueError, match="next transition ID is inconsistent"):
        SequenceReplayBuffer(6, 1, 1).load_state_dict(state)


def test_sequence_replay_checkpoint_rejects_freshness_usage_disagreement() -> None:
    buffer = SequenceReplayBuffer(6, 1, 1)
    for index in range(3):
        add_transition(
            buffer,
            index,
            episode_start=(index == 0),
        )
    state = buffer.state_dict()
    usage = state["usage_counts"]
    assert isinstance(usage, torch.Tensor)
    usage[0, 0] = 1
    with pytest.raises(ValueError, match="freshness disagrees"):
        SequenceReplayBuffer(6, 1, 1).load_state_dict(state)


def test_sequence_replay_checkpoint_rejects_nonchronological_ring_ids() -> None:
    source = SequenceReplayBuffer(4, 1, 1)
    for index in range(7):
        add_transition(source, index, episode_start=(index == 0))
    state = source.state_dict()
    ids = state["transition_ids"]
    assert isinstance(ids, torch.Tensor)
    ids[0, 0], ids[1, 0] = ids[1, 0].item(), ids[0, 0].item()
    with pytest.raises(ValueError, match="chronological transition IDs"):
        SequenceReplayBuffer(4, 1, 1).load_state_dict(state)


def test_sequence_replay_checkpoint_preserves_wrapped_ring_and_future_samples() -> None:
    source = SequenceReplayBuffer(4, 1, 1)
    for index in range(7):
        add_transition(
            source,
            index,
            episode_start=(index in {0, 4}),
            truncated=(index == 3),
        )
    source.sample_sequences(
        1,
        burn_in=1,
        unroll=2,
        generator=torch.Generator().manual_seed(4),
    )
    restored = SequenceReplayBuffer(4, 1, 1)
    restored.load_state_dict(source.state_dict())
    torch.testing.assert_close(
        source.logical_transition_ids(),
        restored.logical_transition_ids(),
    )
    for field in ("observations", "transition_ids", "usage_counts", "fresh"):
        torch.testing.assert_close(
            source.state_dict()[field],
            restored.state_dict()[field],
        )
    add_transition(source, 7)
    add_transition(restored, 7)
    original = source.sample_sequences(
        1,
        burn_in=1,
        unroll=2,
        generator=torch.Generator().manual_seed(9),
    )
    resumed = restored.sample_sequences(
        1,
        burn_in=1,
        unroll=2,
        generator=torch.Generator().manual_seed(9),
    )
    for field in ("transition_ids", "usage_counts", "fresh", "observations"):
        torch.testing.assert_close(getattr(original, field), getattr(resumed, field))


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("capacity", 4.0),
        ("observation_dim", True),
        ("action_dim", "1"),
        ("size", 0.5),
        ("size", False),
    ],
)
def test_sequence_replay_checkpoint_rejects_coerced_integer_metadata(
    field: str,
    invalid: object,
) -> None:
    state = SequenceReplayBuffer(4, 1, 1).state_dict()
    state[field] = invalid
    with pytest.raises(TypeError, match=f"{field} must be an integer"):
        SequenceReplayBuffer(4, 1, 1).load_state_dict(state)


def test_sequence_replay_checkpoint_rejects_tensor_dtype_drift() -> None:
    source = SequenceReplayBuffer(4, 1, 1)
    add_transition(source, 0, episode_start=True)
    state = source.state_dict()
    observations = state["observations"]
    assert isinstance(observations, torch.Tensor)
    state["observations"] = observations.double()
    with pytest.raises(ValueError, match="dtype mismatch for observations"):
        SequenceReplayBuffer(4, 1, 1).load_state_dict(state)


def test_sequence_replay_rejected_checkpoint_does_not_mutate_live_buffer() -> None:
    source = SequenceReplayBuffer(4, 1, 1)
    target = SequenceReplayBuffer(4, 1, 1)
    add_transition(source, 1, episode_start=True)
    add_transition(target, 99, episode_start=True)
    before = target.state_dict()
    invalid = source.state_dict()
    invalid["next_transition_id"] = 999
    with pytest.raises(ValueError, match="next transition ID is inconsistent"):
        target.load_state_dict(invalid)
    after = target.state_dict()
    for field in ("observations", "transition_ids", "usage_counts", "fresh"):
        torch.testing.assert_close(after[field], before[field])
    for field in ("size", "position", "next_transition_id"):
        assert after[field] == before[field]

@pytest.mark.parametrize(
    ("field", "invalid", "error", "message"),
    [
        ("observation", torch.tensor([0.0, 1.0]), ValueError, "shape mismatch"),
        ("next_observation", torch.tensor([float("inf")]), ValueError, "non-finite"),
        ("reward", torch.tensor([1.0, 2.0]), ValueError, "exactly one value"),
        ("episode_start", 1, TypeError, "episode_start must be a boolean"),
        ("insertion_step", True, TypeError, "insertion_step must be an integer"),
    ],
)
def test_sequence_replay_add_rejects_invalid_transition_without_mutation(
    field: str,
    invalid: object,
    error: type[Exception],
    message: str,
) -> None:
    buffer = SequenceReplayBuffer(4, 1, 1)
    payload: dict[str, object] = {
        "observation": torch.tensor([1.0]),
        "action": torch.tensor([0.0]),
        "reward": 1.0,
        "next_observation": torch.tensor([2.0]),
        "terminated": False,
        "truncated": False,
        "episode_start": True,
        "insertion_step": 0,
    }
    payload[field] = invalid

    with pytest.raises(error, match=message):
        buffer.add(**payload)  # type: ignore[arg-type]

    state = buffer.state_dict()
    assert state["size"] == 0
    assert state["position"] == 0
    assert state["next_transition_id"] == 0

