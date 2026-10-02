import pytest

from rl_bgd.runners.adaptive_bgd_lqr_stream import (
    SurpriseSource,
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
    (
        "source",
        "fixed_retention",
        "expects_timeline",
    ),
    (
        ("none", 1.0, False),
        ("none", 0.97, False),
        ("td", 1.0, True),
        ("ensemble", 1.0, True),
        ("predictive", 1.0, True),
    ),
)
def test_adaptive_stream_runner_supports_all_retention_policies(
    source: SurpriseSource,
    fixed_retention: float,
    expects_timeline: bool,
) -> None:
    result = run_adaptive_bgd_lqr_stream(
        total_steps=128,
        phase_steps=64,
        seed=4,
        device="cpu",
        surprise_source=source,
        fixed_retention=fixed_retention,
    )
    assert result["surprise_source"] == source
    assert result["fixed_retention"] == pytest.approx(
        fixed_retention
    )
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
        assert result["surprise_auroc"] is None
    else:
        assert isinstance(
            result["surprise_auroc"],
            float,
        )
