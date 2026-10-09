"""Reject finite high-precision values that overflow checkpoint storage precision."""

import pytest
import torch

from rl_bgd.replay.transition_validation import replay_vector
from rl_bgd.utils.checkpoint_payload import checkpoint_observation


@pytest.mark.parametrize("magnitude", [1e100, -1e100])
def test_replay_vector_rejects_downcast_overflow(magnitude: float) -> None:
    saved = torch.tensor([magnitude], dtype=torch.float64)
    target = torch.zeros(1, dtype=torch.float32)

    with pytest.raises(ValueError, match="overflows replay storage dtype"):
        replay_vector(saved, name="observation", reference=target)


@pytest.mark.parametrize("magnitude", [1e100, -1e100])
def test_checkpoint_observation_rejects_downcast_overflow(magnitude: float) -> None:
    saved = torch.tensor([magnitude], dtype=torch.float64)

    with pytest.raises(ValueError, match="after conversion"):
        checkpoint_observation(
            saved,
            name="resumed observation",
            device=torch.device("cpu"),
            expected_shape=(1,),
        )


def test_valid_float64_values_still_convert_to_float32() -> None:
    saved = torch.tensor([0.125], dtype=torch.float64)
    reference = torch.zeros(1, dtype=torch.float32)
    torch.testing.assert_close(
        replay_vector(saved, name="observation", reference=reference),
        torch.tensor([0.125], dtype=torch.float32),
    )
    torch.testing.assert_close(
        checkpoint_observation(
            saved,
            name="resumed observation",
            device=torch.device("cpu"),
            expected_shape=(1,),
        ),
        torch.tensor([0.125], dtype=torch.float32),
    )
