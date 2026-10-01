import pytest
import torch
from torch import nn

from rl_bgd.baselines.importance import (
    mas_importance,
)
from rl_bgd.baselines.mas import (
    MASRegularizer,
)


class Scalar(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0]))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.w * x


def test_mas_importance_matches_output_norm_gradient() -> None:
    model = Scalar()
    x = torch.tensor([3.0])

    def closure() -> torch.Tensor:
        return model(x)

    importance = mas_importance(
        model,
        [
            closure,
        ],
    )
    assert importance["w"].item() == pytest.approx(18.0)


def test_mas_accumulates_importance_and_penalizes_drift() -> None:
    model = Scalar()
    regularizer = MASRegularizer(strength=2.0)
    regularizer.consolidate(
        model,
        {"w": torch.tensor([1.0])},
    )
    model.w.data.fill_(3.0)
    assert regularizer.penalty(model).item() == pytest.approx(1.0)
    regularizer.consolidate(
        model,
        {"w": torch.tensor([2.0])},
    )
    assert regularizer.importance is not None
    assert regularizer.importance["w"].item() == pytest.approx(3.0)
