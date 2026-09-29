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
