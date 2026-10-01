import pytest
import torch
from torch import nn

from rl_bgd.baselines.ewc import (
    EWCRegularizer,
    OnlineEWCRegularizer,
)
from rl_bgd.baselines.importance import (
    empirical_fisher_diagonal,
)


class Scalar(nn.Module):
    def __init__(
        self,
        value: float = 1.0,
    ) -> None:
        super().__init__()
        self.w = nn.Parameter(
            torch.tensor(
                [value],
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.w
            * x
        )


def test_ewc_penalty_is_zero_at_anchor_and_quadratic_after_move() -> None:
    model = Scalar(
        1.0
    )
    regularizer = EWCRegularizer(
        strength=2.0
    )
    regularizer.consolidate(
        model,
        {
            "w": torch.tensor(
                [3.0]
            )
        },
    )
    assert (
        regularizer.penalty(
            model
        ).item()
        == 0.0
    )
    model.w.data.fill_(
        2.0
    )
    assert (
        regularizer.penalty(
            model
        ).item()
        == pytest.approx(
            3.0
        )
    )


def test_online_ewc_decays_old_importance_before_adding_new() -> None:
    model = Scalar(
        1.0
    )
    regularizer = OnlineEWCRegularizer(
        strength=1.0,
        decay=0.5,
    )
    regularizer.consolidate(
        model,
        {
            "w": torch.tensor(
                [2.0]
            )
        },
    )
    model.w.data.fill_(
        2.0
    )
    regularizer.consolidate(
        model,
        {
            "w": torch.tensor(
                [4.0]
            )
        },
    )
    assert (
        regularizer.importance
        is not None
    )
    assert regularizer.importance[
        "w"
    ].item() == pytest.approx(
        5.0
    )
    model.w.data.fill_(
        3.0
    )
    assert regularizer.penalty(
        model
    ).item() == pytest.approx(
        2.5
    )


def test_empirical_fisher_matches_scalar_gradient_square() -> None:
    model = Scalar(
        1.0
    )
    x = torch.tensor(
        [1.0]
    )

    def closure() -> torch.Tensor:
        prediction = model(
            x
        )
        return (
            0.5
            * prediction.square().sum()
        )

    fisher = empirical_fisher_diagonal(
        model,
        [
            closure,
            closure,
        ],
    )
    assert fisher[
        "w"
    ].item() == pytest.approx(
        1.0
    )
