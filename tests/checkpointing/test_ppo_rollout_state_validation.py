"""Checkpoint invariants for feed-forward and recurrent PPO rollout buffers."""

import pytest
import torch

from rl_bgd.agents.ppo.recurrent_rollout import RecurrentRolloutBuffer
from rl_bgd.agents.ppo.rollout import RolloutBuffer


def _new_buffer(recurrent: bool) -> RolloutBuffer | RecurrentRolloutBuffer:
    if recurrent:
        return RecurrentRolloutBuffer(4, 1, 1, 2, 2)
    return RolloutBuffer(4, 1, 1)


def _fill(
    buffer: RolloutBuffer | RecurrentRolloutBuffer,
    observations: tuple[float, ...],
    *,
    gae: bool,
) -> None:
    for index, obs in enumerate(observations):
        shared = {
            "terminated": False,
            "truncated": False,
            "value": torch.tensor([1.0]),
            "next_value": torch.tensor([1.5]),
            "log_prob": torch.tensor([-0.2]),
        }
        if isinstance(buffer, RecurrentRolloutBuffer):
            buffer.add(
                torch.tensor([obs]),
                torch.tensor([0.1]),
                1.0,
                episode_start=index == 0,
                actor_hidden=torch.tensor([obs, 0.0]),
                value_hidden=torch.tensor([obs, 1.0]),
                **shared,
            )
        else:
            buffer.add(
                torch.tensor([obs]),
                torch.tensor([0.1]),
                1.0,
                **shared,
            )
    if gae:
        buffer.compute_gae(gamma=0.9, gae_lambda=0.95)


def _assert_equal_state(left: dict[str, object], right: dict[str, object]) -> None:
    assert left.keys() == right.keys()
    for name in left:
        if isinstance(left[name], torch.Tensor):
            assert isinstance(right[name], torch.Tensor)
            torch.testing.assert_close(left[name], right[name])
        else:
            assert left[name] == right[name], name


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("gae", [False, True])
def test_ppo_rollout_checkpoint_roundtrip_retains_behavior_state(
    recurrent: bool,
    gae: bool,
) -> None:
    source = _new_buffer(recurrent)
    _fill(source, (1.0, 2.0), gae=gae)
    restored = _new_buffer(recurrent)
    _fill(restored, (99.0,), gae=True)
    restored.load_state_dict(source.state_dict())
    _assert_equal_state(source.state_dict(), restored.state_dict())


@pytest.mark.parametrize("recurrent", [False, True])
def test_ppo_rollout_rejects_invalid_payload_without_mutating_live_state(
    recurrent: bool,
) -> None:
    source = _new_buffer(recurrent)
    _fill(source, (1.0, 2.0), gae=True)
    target = _new_buffer(recurrent)
    _fill(target, (99.0,), gae=True)
    saved = source.state_dict()
    before = target.state_dict()

    observations = saved["observations"]
    returns = saved["returns"]
    assert isinstance(observations, torch.Tensor)
    assert isinstance(returns, torch.Tensor)
    corruptions = (
        ("version", True, "must be an integer"),
        ("capacity", True, "must be an integer"),
        ("size", 2.0, "must be an integer"),
        ("observations", observations.double(), "dtype mismatch"),
        ("returns", torch.zeros(3, 1), "shape mismatch"),
        ("returns", returns.double(), "dtype mismatch"),
        ("advantages", None, "both advantages and returns or neither"),
        ("observations", torch.tensor([[1.0], [float("nan")]]), "non-finite"),
        ("rewards", torch.tensor([[float("inf")], [1.0]]), "non-finite"),
        ("log_probs", torch.tensor([[-0.2], [float("-inf")]]), "non-finite"),
        ("advantages", torch.full((2, 1), float("nan")), "non-finite"),
        ("returns", torch.full((2, 1), float("inf")), "non-finite"),
    )
    if recurrent:
        actor_hiddens = saved["actor_hiddens"]
        assert isinstance(actor_hiddens, torch.Tensor)
        bad_hiddens = actor_hiddens.clone()
        bad_hiddens[1, 0] = float("nan")
        corruptions += (("actor_hiddens", bad_hiddens, "non-finite"),)
    for name, value, message in corruptions:
        corrupt = dict(saved)
        corrupt[name] = value
        with pytest.raises((TypeError, ValueError), match=message):
            target.load_state_dict(corrupt)
        _assert_equal_state(before, target.state_dict())
