import random

import numpy as np
import pytest
import torch

from rl_bgd.utils.randomness import preserved_random_state, seed_everything


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
    with pytest.raises(RuntimeError):
        with preserved_random_state():
            _draw()
            raise RuntimeError("evaluation failed")
    actual = _draw()

    seed_everything(456)
    expected = _draw()
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])
