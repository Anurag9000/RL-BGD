import pytest

from rl_bgd.metrics.continual import (
    backward_transfer,
    final_average_performance,
    forgetting,
    forward_transfer,
    lifetime_auc,
    plasticity_retention,
)


def test_basic_continual_metrics() -> None:
    matrix = [
        [10.0, 0.0, 0.0],
        [8.0, 12.0, 0.0],
        [7.0, 11.0, 15.0],
    ]
    values, mean_forgetting = forgetting(matrix)
    assert values.tolist() == [3.0, 1.0]
    assert mean_forgetting == pytest.approx(2.0)
    assert backward_transfer(matrix) == pytest.approx(-2.0)
    assert final_average_performance(matrix[-1]) == pytest.approx(11.0)


def test_forward_transfer_and_lifetime_auc() -> None:
    assert forward_transfer([2.0, 4.0], [1.0, 1.0]) == pytest.approx(2.0)
    assert lifetime_auc([0.0, 10.0], [0.0, 2.0]) == pytest.approx(1.0)
    assert plasticity_retention(4.0, 5.0) == pytest.approx(0.8)
