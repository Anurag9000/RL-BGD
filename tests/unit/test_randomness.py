import random

import numpy as np
import pytest
import torch

from rl_bgd.utils.randomness import (
    load_random_state_dict,
    preserved_random_state,
    random_state_dict,
    seed_everything,
)


def _draw() -> tuple[float, float, torch.Tensor]:
    return (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )


def test_preserved_random_state_restores_all_cpu_rng_streams() -> None:
    seed_everything(123)
    with preserved_random_state():
        _draw()
        _draw()
    actual = _draw()

    seed_everything(123)
    expected = _draw()

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_preserved_random_state_restores_after_exception() -> None:
    seed_everything(456)
    with pytest.raises(RuntimeError), preserved_random_state():
        _draw()
        raise RuntimeError("evaluation failed")
    actual = _draw()

    seed_everything(456)
    expected = _draw()
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_random_state_checkpoint_round_trip() -> None:
    seed_everything(789)
    state = random_state_dict()
    expected = _draw()
    _draw()
    load_random_state_dict(state)
    actual = _draw()

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])

@pytest.mark.parametrize(
    "corruption",
    ["numpy", "torch_cpu", "torch_cuda"],
)
def test_random_state_checkpoint_rejection_is_non_mutating(corruption: str) -> None:
    seed_everything(999)
    state = random_state_dict()
    if corruption == "numpy":
        state["numpy"] = "invalid"
    elif corruption == "torch_cpu":
        state["torch_cpu"] = torch.ones(8, dtype=torch.float32)
    else:
        state["torch_cuda"] = "invalid"

    seed_everything(321)
    expected = _draw()
    seed_everything(321)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        load_random_state_dict(state)
    actual = _draw()

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])

