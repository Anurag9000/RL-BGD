import pytest

from rl_bgd.runners.adaptive_bgd_lqr_stream import (
    run_adaptive_bgd_lqr_stream,
)


@pytest.mark.slow
def test_adaptive_stream_runner_produces_surprise_timeline() -> None:
    result = run_adaptive_bgd_lqr_stream(
        total_steps=160,
        phase_steps=80,
        seed=3,
        device="cpu",
    )
    assert result["surprise_timeline"]
    assert result["true_change_steps"] == [80]
    access = result["information_access"]
    assert access["receives_task_id"] is False
    assert access["receives_task_boundary"] is False


@pytest.mark.slow
@pytest.mark.parametrize(
    ("source", "expects_timeline"),
    (
        ("none", False),
        ("td", True),
        ("ensemble", True),
        ("predictive", True),
    ),
)
def test_adaptive_stream_runner_supports_all_surprise_sources(
    source: str,
    expects_timeline: bool,
) -> None:
    result = run_adaptive_bgd_lqr_stream(
        total_steps=128,
        phase_steps=64,
        seed=4,
        device="cpu",
        surprise_source=source,  # type: ignore[arg-type]
    )
    assert result["surprise_source"] == source
    assert bool(
        result["surprise_timeline"]
    ) is expects_timeline
    assert result["true_change_steps"] == [
        64
    ]
    detection = result[
        "change_detection"
    ]
    assert isinstance(
        detection,
        dict,
    )
    if source == "none":
        assert result["detected_steps"] == []
        assert detection["recall"] == 0.0
