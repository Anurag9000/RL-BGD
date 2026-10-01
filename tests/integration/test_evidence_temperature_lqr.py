import math

from rl_bgd.runners.evidence_temperature_lqr import (
    run_evidence_temperature_sweep,
)


def test_evidence_temperature_sweep_reports_matched_runs() -> None:
    result = run_evidence_temperature_sweep(
        temperatures=(0.5, 1.0),
        steps=96,
        seed=102,
        device="cpu",
    )
    assert result["temperatures"] == [
        0.5,
        1.0,
    ]
    controlled = result["controlled_variables"]
    assert controlled["temperature_is_only_swept_bgd_parameter"] is True
    runs = result["runs"]
    assert len(runs) == 2
    assert runs[0]["evidence_temperature"] == 0.5
    assert runs[1]["evidence_temperature"] == 1.0
    assert all(math.isfinite(run["post_return"]) for run in runs)
