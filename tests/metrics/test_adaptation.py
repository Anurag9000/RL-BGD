import pytest

from rl_bgd.metrics.adaptation import (
    post_change_auc,
    recurrence_metrics,
    time_to_fraction,
)


def test_time_to_fraction_and_post_change_auc() -> None:
    steps = [0, 10, 20, 30, 40]
    values = [1.0, 0.0, 4.0, 8.0, 10.0]
    assert time_to_fraction(
        steps,
        values,
        switch_step=10,
        reference=10.0,
        fraction=0.8,
    ) == pytest.approx(20.0)
    assert post_change_auc(
        steps,
        values,
        switch_step=10,
        window_steps=20,
    ) == pytest.approx(4.0)


def test_recurrence_metrics() -> None:
    result = recurrence_metrics(
        [0, 10, 20, 30],
        [10.0, 3.0, 8.0, 9.0],
        revisit_step=10,
        reference=10.0,
        asymptotic_window=2,
    )
    assert result.zero_shot_return == 3.0
    assert result.reacquisition_steps == 10.0
    assert result.recovered_performance == pytest.approx(8.5)
