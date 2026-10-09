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


@pytest.mark.parametrize("mode", ["unknown", "random-walk", ""])
def test_context_schedule_rejects_invalid_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="unsupported context schedule mode"):
        ContextSchedule(
            ContextScheduleConfig(
                mode=mode,  # type: ignore[arg-type]
                anchors=({"g": 0.0},),
            )
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_context_schedule_rejects_nonfinite_anchors(bad_value: float) -> None:
    with pytest.raises(ValueError, match="anchors must contain finite"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="abrupt",
                anchors=({"g": bad_value},),
            )
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_context_schedule_rejects_nonfinite_bounds(bad_value: float) -> None:
    with pytest.raises(ValueError, match="bounds must be finite"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="random_walk",
                anchors=({"g": 0.0},),
                bounds={"g": (bad_value, 1.0)},
            )
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_context_schedule_rejects_nonfinite_walk_std(bad_value: float) -> None:
    with pytest.raises(ValueError, match="random_walk_std must be finite"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="random_walk",
                anchors=({"g": 0.0},),
                random_walk_std=bad_value,
            )
        )


@pytest.mark.parametrize("step", [-1, 1.5, True])
def test_context_schedule_rejects_invalid_step(step: object) -> None:
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="recurring",
            anchors=({"g": 1.0},),
        )
    )
    with pytest.raises(ValueError, match="step must be a non-negative integer"):
        schedule.context_at(step)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", [True, "1.0", object()])
def test_context_schedule_rejects_non_numeric_anchor_values(bad_value: object) -> None:
    with pytest.raises(TypeError, match="anchor values must be real numbers"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="abrupt",
                anchors=({"g": bad_value},),  # type: ignore[dict-item]
            )
        )


@pytest.mark.parametrize("bad_seed", [True, 1.5, "7"])
def test_context_schedule_rejects_non_integer_seed(bad_seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="abrupt",
                anchors=({"g": 0.0},),
                seed=bad_seed,  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize("bad_std", [True, "0.1"])
def test_context_schedule_rejects_non_numeric_walk_std(bad_std: object) -> None:
    with pytest.raises(TypeError, match="random_walk_std must be a real number"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="random_walk",
                anchors=({"g": 0.0},),
                random_walk_std=bad_std,  # type: ignore[arg-type]
            )
        )


def test_context_schedule_rejects_non_string_anchor_key() -> None:
    with pytest.raises(ValueError, match="keys must be non-empty strings"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="abrupt",
                anchors=({1: 0.0},),  # type: ignore[dict-item]
            )
        )


@pytest.mark.parametrize(
    "bad_bounds",
    [
        {"g": [0.0, 1.0]},
        {"g": (False, 1.0)},
        {"g": ("0.0", 1.0)},
    ],
)
def test_context_schedule_rejects_malformed_bounds(bad_bounds: object) -> None:
    with pytest.raises(TypeError, match="context bounds"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="random_walk",
                anchors=({"g": 0.0},),
                bounds=bad_bounds,  # type: ignore[arg-type]
            )
        )

def test_random_walk_rejects_extra_ignored_anchors() -> None:
    with pytest.raises(ValueError, match="exactly one starting anchor"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="random_walk",
                anchors=({"g": 0.0}, {"g": 1.0}),
            )
        )


def test_non_random_walk_rejects_ignored_bounds() -> None:
    with pytest.raises(ValueError, match="only supported for random-walk"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="abrupt",
                anchors=({"g": 0.0},),
                bounds={"g": (-1.0, 1.0)},
            )
        )


def test_random_walk_rejects_start_outside_declared_bounds() -> None:
    with pytest.raises(ValueError, match="starting anchor must lie within bounds"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="random_walk",
                anchors=({"g": 2.0},),
                bounds={"g": (-1.0, 1.0)},
            )
        )


def test_context_schedule_rejects_non_mapping_anchor() -> None:
    with pytest.raises(TypeError, match="anchors must be dictionaries"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="abrupt",
                anchors=(("g", 0.0),),  # type: ignore[arg-type]
            )
        )


def test_context_schedule_rejects_non_mapping_bounds() -> None:
    with pytest.raises(TypeError, match="bounds must be a dictionary"):
        ContextSchedule(
            ContextScheduleConfig(
                mode="random_walk",
                anchors=({"g": 0.0},),
                bounds=(("g", (-1.0, 1.0)),),  # type: ignore[arg-type]
            )
        )

