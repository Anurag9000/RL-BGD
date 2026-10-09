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



@pytest.mark.parametrize(
    ("steps", "values"),
    [
        ([0, 10, 10], [0.0, 5.0, 10.0]),
        ([0, 20, 10], [0.0, 5.0, 10.0]),
        ([0, float("nan"), 20], [0.0, 5.0, 10.0]),
        ([0, 10, 20], [0.0, float("inf"), 10.0]),
    ],
)
def test_adaptation_metrics_reject_unordered_or_nonfinite_traces(
    steps: list[float],
    values: list[float],
) -> None:
    with pytest.raises(ValueError, match="finite|strictly increasing"):
        time_to_fraction(steps, values, switch_step=10, reference=10)
    with pytest.raises(ValueError, match="finite|strictly increasing"):
        post_change_auc(steps, values, switch_step=0, window_steps=10)
    with pytest.raises(ValueError, match="finite|strictly increasing"):
        recurrence_metrics(steps, values, revisit_step=10, reference=10)


@pytest.mark.parametrize("field", ["switch_step", "reference", "baseline"])
def test_adaptation_threshold_rejects_nonfinite_parameters(field: str) -> None:
    kwargs = {"switch_step": 10.0, "reference": 10.0, "baseline": 0.0}
    kwargs[field] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        time_to_fraction([0, 10, 20], [0.0, 5.0, 10.0], **kwargs)


def test_post_change_auc_interpolates_exact_window_boundaries() -> None:
    steps = [0.0, 10.0, 20.0]
    values = [0.0, 10.0, 20.0]
    assert post_change_auc(
        steps, values, switch_step=5.0, window_steps=10.0
    ) == pytest.approx(10.0)
    assert post_change_auc(
        steps,
        values,
        switch_step=5.0,
        window_steps=10.0,
        normalize_by_duration=False,
    ) == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("switch_step", "window_steps"),
    [
        (10.0, 15.0),
        (-1.0, 10.0),
        (0.0, float("inf")),
        (float("nan"), 10.0),
    ],
)
def test_post_change_auc_rejects_incomplete_or_nonfinite_windows(
    switch_step: float,
    window_steps: float,
) -> None:
    with pytest.raises(ValueError, match="finite|covered"):
        post_change_auc(
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 20.0],
            switch_step=switch_step,
            window_steps=window_steps,
        )


def test_recurrence_rejects_nonfinite_reference_and_boolean_window() -> None:
    with pytest.raises(ValueError, match="finite"):
        recurrence_metrics([0, 10], [1.0, 2.0], revisit_step=10, reference=float("nan"))
    with pytest.raises(TypeError, match="integer"):
        recurrence_metrics(
            [0, 10], [1.0, 2.0], revisit_step=10, reference=2.0, asymptotic_window=True
        )
