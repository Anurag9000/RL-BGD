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
        "stationary_core",
        "dev",
        "baseline_core",
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
            "hidden_context_"
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
        "C",
        "D",
        "F",
        "G",
        "GB-T",
    }

    plasticity_jobs = [
        job
        for job in ablations
        if job.hypothesis_id == "C"
    ]
    assert {
        job.kwargs[
            "consolidation_retention"
        ]
        for job in plasticity_jobs
    } == {
        1.0,
        0.97,
    }
    assert all(
        job.primary_metric
        == "post_shift_normalized_auc"
        for job in plasticity_jobs
    )


def test_paper_suites_cover_every_execution_hypothesis() -> None:
    hypothesis_ids = {
        job.hypothesis_id
        for suite in SUITES.values()
        for job in suite.jobs
    }
    assert {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "GB-T",
        "H",
        "CW-CAN10",
        "CW-CAN20",
        "CW-TA10",
        "CW-TA20",
        "UCL",
    } <= hypothesis_ids
    assert "I-J" in hypothesis_ids


def test_stationary_and_mechanism_suites_have_replicate_coverage() -> None:
    stationary = SUITES[
        "stationary_core"
    ].jobs
    assert {
        job.job_id
        for job in stationary
    } == {
        "stationary_sac_adam",
        "stationary_sac_bgd",
        "stationary_ppo_adam",
        "stationary_ppo_bgd",
    }
    assert all(
        job.hypothesis_id
        == "A"
        for job in stationary
    )
    assert all(
        job.seeds
        == (
            0,
            1,
            2,
            3,
            4,
        )
        for job in stationary
    )

    mechanisms = SUITES[
        "mechanism_analysis"
    ].jobs
    assert len(
        mechanisms
    ) == 1
    assert mechanisms[
        0
    ].seeds == (
        150,
        151,
        152,
        153,
        154,
    )
    assert mechanisms[
        0
    ].seed_kwarg == "seed"


def test_external_baseline_suite_has_full_method_coverage() -> None:
    jobs = SUITES[
        "baseline_core"
    ].jobs
    regularized = [
        job
        for job in jobs
        if job.job_id.startswith(
            "baseline_"
        )
        and job.hypothesis_id
        == "B"
    ]
    assert {
        job.kwargs[
            "method"
        ]
        for job in regularized
    } == {
        "ewc",
        "online_ewc",
        "si",
        "mas",
    }
    assert all(
        job.seeds
        == (
            0,
            1,
            2,
            3,
            4,
        )
        for job in jobs
    )

    ucl = [
        job
        for job in jobs
        if job.hypothesis_id
        == "UCL"
    ]
    assert len(
        ucl
    ) == 1
    assert ucl[
        0
    ].protocol == (
        "oracle_boundary"
    )


def _tiny_suite(
    primary_metric: str,
) -> ExperimentSuite:
    return ExperimentSuite(
        name="smoke",
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
        "smoke",
        suite,
    )
    result = execute_suite(
        "smoke",
        tmp_path,
    )
    assert result[
        "status"
    ] == "success"
    run_dir = (
        tmp_path
        / "smoke"
        / "smoke__tiny_smoke__seed_0"
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
        / "smoke__tiny_smoke__seed_0"
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
    failed_manifest = json.loads(
        (
            run_dir
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert failed_manifest[
        "status"
    ] == "failed"
    with pytest.raises(
        ValueError,
        match="refuses incomplete",
    ):
        load_run_directory(
            run_dir
        )
