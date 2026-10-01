import json
from pathlib import Path

from rl_bgd.experiments.suites import (
    SUITES,
    materialize_suite,
    validate_suite_registry,
)


def test_required_paper_suites_validate_against_real_targets() -> None:
    validate_suite_registry()
    assert set(SUITES) == {
        "smoke",
        "dev",
        "carl_core",
        "cw10_core",
        "cw20_final",
        "task_agnostic_final",
        "ablation_core",
        "uncertainty_analysis",
        "mechanism_analysis",
        "compute_analysis",
    }


def test_suite_manifest_contains_complete_job_metadata(
    tmp_path: Path,
) -> None:
    manifest = materialize_suite(
        "smoke",
        tmp_path,
    )
    path = Path(manifest["manifest_path"])
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["suite"] == "smoke"
    assert saved["jobs"]
    for job in saved["jobs"]:
        assert job["run_id"]
        assert job["hypothesis_id"]
        assert job["target"]
        assert job["algorithm"]
        assert job["environment"]
        assert job["protocol"]
        assert job["primary_metric"]
        assert job["command"]
        assert job["run_dir"]
