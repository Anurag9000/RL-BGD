import json
from pathlib import Path

import pytest

from rl_bgd.artifacts import load_run_directory
from rl_bgd.experiments.suites import (
    SUITES,
    ExperimentJob,
    ExperimentSuite,
    execute_suite,
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


def test_comparison_and_ablation_jobs_are_atomic() -> None:
    hidden_jobs = [
        job
        for job in SUITES["dev"].jobs
        if job.job_id.startswith(
            "dev_hidden_"
        )
    ]
    assert len(hidden_jobs) == 5
    assert all(
        job.target.endswith(
            ":run_hidden_context_sac_variant"
        )
        for job in hidden_jobs
    )
    assert len(
        {
            job.kwargs["variant"]
            for job in hidden_jobs
        }
    ) == 5

    ablations = SUITES[
        "ablation_core"
    ].jobs
    replay_jobs = [
        job
        for job in ablations
        if job.hypothesis_id == "F"
    ]
    assert {
        job.kwargs[
            "replay_evidence_mode"
        ]
        for job in replay_jobs
    } == {
        "all_replay",
        "fresh_only_uncertainty",
        "inverse_reuse_weight",
        "normalized_batch_evidence",
    }
    assert {
        job.hypothesis_id
        for job in ablations
    } >= {
        "D",
        "F",
        "G",
        "GB-T",
    }


def _tiny_suite(
    primary_metric: str,
) -> ExperimentSuite:
    return ExperimentSuite(
        name="tiny_strict",
        description=(
            "strict-artifact execution test"
        ),
        jobs=(
            ExperimentJob(
                job_id="tiny_smoke",
                hypothesis_id="TEST",
                target=(
                    "rl_bgd.runners.smoke:"
                    "run_smoke"
                ),
                kwargs={
                    "steps": 4,
                    "device": "cpu",
                },
                seeds=(0,),
                algorithm="BGD-smoke",
                environment="quadratic",
                protocol="stationary",
                config_path=None,
                primary_metric=primary_metric,
                secondary_metrics=(
                    "initial_abs_mean",
                ),
                runtime_class="smoke",
            ),
        ),
    )


def test_execute_suite_writes_strict_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = _tiny_suite(
        "final_abs_mean"
    )
    monkeypatch.setitem(
        SUITES,
        suite.name,
        suite,
    )
    result = execute_suite(
        suite.name,
        tmp_path,
    )
    assert result[
        "status"
    ] == "success"
    run_dir = (
        tmp_path
        / suite.name
        / "tiny_smoke__seed_0"
    )
    loaded = load_run_directory(
        run_dir
    )
    assert loaded.manifest.method == (
        "BGD-smoke"
    )
    assert loaded.summary.metrics[
        "final_abs_mean"
    ] >= 0.0
    metadata = json.loads(
        (
            run_dir
            / "run_metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert metadata[
        "strict_artifacts"
    ] is True
    assert metadata[
        "job_id"
    ] == "tiny_smoke"


def test_execute_suite_fails_closed_on_missing_primary_metric(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = _tiny_suite(
        "does_not_exist"
    )
    monkeypatch.setitem(
        SUITES,
        suite.name,
        suite,
    )
    result = execute_suite(
        suite.name,
        tmp_path,
    )
    assert result[
        "status"
    ] == "failed"
    run_dir = (
        tmp_path
        / suite.name
        / "tiny_smoke__seed_0"
    )
    metadata = json.loads(
        (
            run_dir
            / "run_metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert metadata[
        "strict_artifacts"
    ] is False
    assert (
        "declared primary metric"
        in metadata[
            "artifact_error"
        ]
    )
    failed = load_run_directory(
        run_dir,
        require_completed=False,
    )
    assert failed.manifest.status == "failed"
    with pytest.raises(
        ValueError,
        match="refuses incomplete",
    ):
        load_run_directory(
            run_dir
        )
