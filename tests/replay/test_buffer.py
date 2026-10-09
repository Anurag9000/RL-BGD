import pytest
import torch

from rl_bgd.replay.buffer import ReplayBuffer


def test_replay_tracks_usage_freshness_and_truncation_bootstrap() -> None:
    buffer = ReplayBuffer(4, 1, 1)
    for i in range(4):
        buffer.add(
            torch.tensor([float(i)]),
            torch.tensor([0.0]),
            1.0,
            torch.tensor([float(i + 1)]),
            terminated=(i == 0),
            truncated=(i == 1),
            insertion_step=i,
        )
    generator = torch.Generator().manual_seed(0)
    batch = buffer.sample(4, generator=generator)
    assert torch.all(batch.usage_counts >= 1)
    assert torch.all(batch.fresh)
    mask = batch.truncated & ~batch.terminated
    assert torch.all(batch.bootstrap_mask[mask] == 1)
    assert torch.all(batch.bootstrap_mask[batch.terminated] == 0)


def test_replay_checkpoint_round_trip() -> None:
    source = ReplayBuffer(4, 1, 1)
    source.add(
        torch.tensor([1.0]),
        torch.tensor([0.25]),
        2.0,
        torch.tensor([1.5]),
        terminated=False,
        truncated=True,
        insertion_step=7,
    )
    state = source.state_dict()
    restored = ReplayBuffer(4, 1, 1)
    restored.load_state_dict(state)
    assert len(restored) == 1
    torch.testing.assert_close(
        restored.observations[0],
        torch.tensor([1.0]),
    )
    torch.testing.assert_close(
        restored.actions[0],
        torch.tensor([0.25]),
    )
    assert restored.truncated[0, 0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("position", 3, "position is inconsistent"),
        ("next_transition_id", 4, "next transition ID is inconsistent"),
    ],
)
def test_replay_checkpoint_rejects_inconsistent_provenance(
    field: str,
    value: int,
    message: str,
) -> None:
    source = ReplayBuffer(4, 1, 1)
    source.add(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        1.0,
        torch.tensor([2.0]),
        terminated=False,
        truncated=False,
        insertion_step=0,
    )
    state = source.state_dict()
    state[field] = value
    with pytest.raises(ValueError, match=message):
        ReplayBuffer(4, 1, 1).load_state_dict(state)


def test_replay_checkpoint_rejects_freshness_usage_disagreement() -> None:
    source = ReplayBuffer(4, 1, 1)
    source.add(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        1.0,
        torch.tensor([2.0]),
        terminated=False,
        truncated=False,
        insertion_step=0,
    )
    state = source.state_dict()
    fresh = state["fresh"]
    assert isinstance(fresh, torch.Tensor)
    fresh[0, 0] = False
    with pytest.raises(ValueError, match="freshness disagrees"):
        ReplayBuffer(4, 1, 1).load_state_dict(state)


def test_replay_checkpoint_rejects_nonchronological_ring_ids() -> None:
    source = ReplayBuffer(4, 1, 1)
    for index in range(7):
        source.add(
            torch.tensor([float(index)]),
            torch.tensor([0.0]),
            0.0,
            torch.tensor([float(index + 1)]),
            terminated=False,
            truncated=False,
            insertion_step=index,
        )
    state = source.state_dict()
    ids = state["transition_ids"]
    assert isinstance(ids, torch.Tensor)
    ids[0, 0], ids[1, 0] = ids[1, 0].item(), ids[0, 0].item()
    with pytest.raises(ValueError, match="chronological transition IDs"):
        ReplayBuffer(4, 1, 1).load_state_dict(state)


def test_replay_checkpoint_preserves_wrapped_ring_and_future_samples() -> None:
    source = ReplayBuffer(4, 1, 1)
    for index in range(7):
        source.add(
            torch.tensor([float(index)]),
            torch.tensor([0.0]),
            float(index),
            torch.tensor([float(index + 1)]),
            terminated=False,
            truncated=False,
            insertion_step=index,
        )
    source.sample(2, generator=torch.Generator().manual_seed(4))
    restored = ReplayBuffer(4, 1, 1)
    restored.load_state_dict(source.state_dict())
    assert len(restored) == 4
    for field in ("observations", "transition_ids", "usage_counts", "fresh"):
        torch.testing.assert_close(source.state_dict()[field], restored.state_dict()[field])
    for buffer in (source, restored):
        transition_id = buffer.add(
            torch.tensor([7.0]),
            torch.tensor([0.0]),
            7.0,
            torch.tensor([8.0]),
            terminated=False,
            truncated=False,
            insertion_step=7,
        )
        assert transition_id == 7
    original = source.sample(3, generator=torch.Generator().manual_seed(9))
    resumed = restored.sample(3, generator=torch.Generator().manual_seed(9))
    for field in ("transition_ids", "usage_counts", "fresh", "observations"):
        torch.testing.assert_close(getattr(original, field), getattr(resumed, field))


def test_replay_checkpoint_rejects_tensor_dtype_drift() -> None:
    source = ReplayBuffer(4, 1, 1)
    source.add(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        1.0,
        torch.tensor([2.0]),
        terminated=False,
        truncated=False,
        insertion_step=0,
    )
    state = source.state_dict()
    observations = state["observations"]
    assert isinstance(observations, torch.Tensor)
    state["observations"] = observations.double()
    with pytest.raises(ValueError, match="dtype mismatch for observations"):
        ReplayBuffer(4, 1, 1).load_state_dict(state)


def test_replay_rejected_checkpoint_does_not_mutate_live_buffer() -> None:
    source = ReplayBuffer(4, 1, 1)
    target = ReplayBuffer(4, 1, 1)
    for buffer, observation in ((source, 1.0), (target, 99.0)):
        buffer.add(
            torch.tensor([observation]),
            torch.tensor([0.0]),
            0.0,
            torch.tensor([observation + 1.0]),
            terminated=False,
            truncated=False,
            insertion_step=0,
        )
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
        ("action", torch.tensor([float("nan")]), ValueError, "non-finite"),
        ("reward", float("inf"), ValueError, "reward must be finite"),
        ("terminated", 1, TypeError, "terminated must be a boolean"),
        ("truncated", 0, TypeError, "truncated must be a boolean"),
        ("insertion_step", -1, ValueError, "insertion_step must be non-negative"),
    ],
)
def test_replay_add_rejects_invalid_transition_without_mutation(
    field: str,
    invalid: object,
    error: type[Exception],
    message: str,
) -> None:
    buffer = ReplayBuffer(4, 1, 1)
    payload: dict[str, object] = {
        "observation": torch.tensor([1.0]),
        "action": torch.tensor([0.0]),
        "reward": 1.0,
        "next_observation": torch.tensor([2.0]),
        "terminated": False,
        "truncated": False,
        "insertion_step": 0,
    }
    payload[field] = invalid

    with pytest.raises(error, match=message):
        buffer.add(**payload)  # type: ignore[arg-type]

    state = buffer.state_dict()
    assert state["size"] == 0
    assert state["position"] == 0
    assert state["next_transition_id"] == 0

def test_replay_checkpoint_rejects_negative_insertion_step() -> None:
    source = ReplayBuffer(4, 1, 1)
    source.add(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        1.0,
        torch.tensor([2.0]),
        terminated=False,
        truncated=False,
        insertion_step=0,
    )
    state = source.state_dict()
    insertion_steps = state["insertion_steps"]
    assert isinstance(insertion_steps, torch.Tensor)
    insertion_steps[0, 0] = -1

    with pytest.raises(ValueError, match="negative insertion steps"):
        ReplayBuffer(4, 1, 1).load_state_dict(state)

