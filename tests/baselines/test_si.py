import pytest
import torch
from torch import nn

from rl_bgd.baselines.si import (
    SynapticIntelligence,
)


class Scalar(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0]))


def test_si_tracks_descent_path_and_consolidates_positive_importance() -> None:
    model = Scalar()
    regularizer = SynapticIntelligence(
        model,
        strength=2.0,
        damping=0.01,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    loss = 0.5 * model.w.square().sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = regularizer.capture_gradients(model)
    optimizer.step()
    regularizer.accumulate_after_step(
        model,
        gradients,
    )
    regularizer.consolidate(model)
    assert regularizer.importance["w"].item() > 0.0
    assert regularizer.penalty(model).item() == 0.0
    model.w.data.add_(0.25)
    assert regularizer.penalty(model).item() > 0.0


def test_si_checkpoint_round_trip() -> None:
    model = Scalar()
    regularizer = SynapticIntelligence(
        model,
        strength=3.0,
        damping=0.2,
    )
    state = regularizer.state_dict()
    restored = SynapticIntelligence(model)
    restored.load_state_dict(state)
    assert restored.strength == pytest.approx(3.0)
    assert restored.damping == pytest.approx(0.2)
    torch.testing.assert_close(
        restored.anchor["w"],
        regularizer.anchor["w"],
    )
