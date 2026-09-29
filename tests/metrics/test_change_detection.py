import pytest

from rl_bgd.metrics.change_detection import (
    binary_auroc,
    change_detection_metrics,
    change_event_labels,
)


def test_event_matching_reports_delay_and_false_alarm() -> None:
    result = change_detection_metrics(
        [100, 300],
        [10, 105, 320],
        tolerance_steps=30,
        total_steps=500,
    )
    assert result.matched_events == 2
    assert result.mean_detection_delay == pytest.approx(12.5)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == 1.0
    assert result.false_alarms_per_million_steps == pytest.approx(2000.0)


def test_change_labels_and_auroc() -> None:
    labels = change_event_labels(
        8,
        [2, 6],
        positive_window=1,
    )
    assert labels.tolist() == [0, 0, 1, 0, 0, 0, 1, 0]
    scores = [0.0, 0.1, 1.0, 0.2, 0.3, 0.4, 0.9, 0.2]
    assert binary_auroc(scores, labels) == pytest.approx(1.0)
