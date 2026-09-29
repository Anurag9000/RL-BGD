import math

import pytest

from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig


def test_abrupt_schedule_stops_at_last_anchor() -> None:
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="abrupt",
            anchors=({"g": 1.0}, {"g": 2.0}, {"g": 3.0}),
            phase_steps=5,
        )
    )
    assert schedule.context_at(0)["g"] == 1.0
    assert schedule.context_at(5)["g"] == 2.0
    assert schedule.context_at(500)["g"] == 3.0


def test_recurring_schedule_cycles() -> None:
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="recurring",
            anchors=({"g": 1.0}, {"g": 2.0}, {"g": 3.0}),
            phase_steps=2,
        )
    )
    values = [schedule.context_at(step)["g"] for step in (0, 2, 4, 6)]
    assert values == [1.0, 2.0, 3.0, 1.0]


def test_smooth_schedule_interpolates_multiple_dimensions() -> None:
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="smooth",
            anchors=(
                {"g": 0.0, "m": 2.0},
                {"g": 10.0, "m": 4.0},
            ),
            phase_steps=10,
        )
    )
    assert schedule.context_at(5) == {"g": 5.0, "m": 3.0}
    assert schedule.context_at(10) == {"g": 10.0, "m": 4.0}


def test_periodic_schedule_uses_center_and_amplitude() -> None:
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="periodic",
            anchors=({"g": 10.0}, {"g": 2.0}),
            period_steps=100,
        )
    )
    assert schedule.context_at(0)["g"] == pytest.approx(10.0)
    assert schedule.context_at(25)["g"] == pytest.approx(12.0)
    assert schedule.context_at(75)["g"] == pytest.approx(8.0)
    assert math.isfinite(schedule.context_at(123)["g"])


def test_random_walk_is_seeded_bounded_and_order_stable() -> None:
    config = ContextScheduleConfig(
        mode="random_walk",
        anchors=({"g": 0.0},),
        random_walk_std=1.0,
        seed=7,
        bounds={"g": (-0.5, 0.5)},
    )
    left = ContextSchedule(config)
    right = ContextSchedule(config)
    values_left = [left.context_at(i)["g"] for i in range(10)]
    values_right = [right.context_at(i)["g"] for i in range(10)]
    assert values_left == values_right
    assert all(-0.5 <= value <= 0.5 for value in values_left)
