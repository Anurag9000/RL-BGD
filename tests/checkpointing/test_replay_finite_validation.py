"""Fail-closed numerical integrity of stationary and recurrent SAC replay."""

import pytest
import torch

from rl_bgd.replay.buffer import ReplayBuffer
from rl_bgd.replay.sequence_buffer import SequenceReplayBuffer


def _new_buffer(recurrent: bool) -> ReplayBuffer | SequenceReplayBuffer:
    if recurrent:
        return SequenceReplayBuffer(4, 1, 1)
    return ReplayBuffer(4, 1, 1)


def _populate(buffer: ReplayBuffer | SequenceReplayBuffer, values: tuple[float, ...]) -> None:
    for index, value in enumerate(values):
        kwargs = {
            "terminated": False,
            "truncated": False,
            "insertion_step": index,
        }
        if isinstance(buffer, SequenceReplayBuffer):
            buffer.add(
                torch.tensor([value]),
                torch.tensor([0.25]),
                value,
                torch.tensor([value + 1.0]),
                episode_start=index == 0,
                **kwargs,
            )
        else:
            buffer.add(
                torch.tensor([value]),
                torch.tensor([0.25]),
                value,
                torch.tensor([value + 1.0]),
                **kwargs,
            )


def _assert_same_state(left: dict[str, object], right: dict[str, object]) -> None:
    assert left.keys() == right.keys()
    for name, original in left.items():
        restored = right[name]
        if isinstance(original, torch.Tensor):
            assert isinstance(restored, torch.Tensor)
            torch.testing.assert_close(original, restored)
        else:
            assert restored == original, name


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("field", ["observations", "actions", "rewards", "next_observations"])
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_corrupt_replay_floats_fail_without_partial_mutation(
    recurrent: bool,
    field: str,
    invalid_value: float,
) -> None:
    source = _new_buffer(recurrent)
    _populate(source, (1.0, 2.0))
    target = _new_buffer(recurrent)
    _populate(target, (99.0,))
    before = target.state_dict()

    corrupt = source.state_dict()
    values = corrupt[field]
    assert isinstance(values, torch.Tensor)
    damaged = values.clone()
    damaged[1, 0] = invalid_value
    corrupt[field] = damaged

    with pytest.raises(ValueError, match="non-finite"):
        target.load_state_dict(corrupt)
    _assert_same_state(before, target.state_dict())


@pytest.mark.parametrize("recurrent", [False, True])
def test_replay_rejects_boolean_version_without_mutating_state(recurrent: bool) -> None:
    source = _new_buffer(recurrent)
    _populate(source, (1.0, 2.0))
    target = _new_buffer(recurrent)
    _populate(target, (99.0,))
    before = target.state_dict()

    corrupt = source.state_dict()
    corrupt["version"] = True
    with pytest.raises(TypeError, match="must be an integer"):
        target.load_state_dict(corrupt)
    _assert_same_state(before, target.state_dict())
