import json
from pathlib import Path

from rl_bgd.analysis.artifacts import (
    BootstrapConfig,
    bootstrap_mean_ci,
    build_paper_artifacts,
)
from rl_bgd.artifacts import (
    RunManifest,
    RunSummary,
    write_run_artifacts,
)


def _fake_run(
    root: Path,
    *,
    seed: int,
    value: float,
) -> None:
    run_id = f"demo__seed_{seed}"
    run = root / "smoke" / run_id
    manifest = RunManifest(
        run_id=run_id,
        method="demo",
        setting="stationary",
        benchmark="synthetic",
        seed=seed,
        git_commit="abc123",
        metadata={
            "suite": "smoke",
            "job_id": "demo",
            "hypothesis_id": "A",
            "primary_metric": "post_return",
            "secondary_metrics": [
                "final_10_mean_return",
            ],
            "source_config_path": None,
        },
    )
    summary = RunSummary(
        run_id=run_id,
        metrics={
            "post_return": value,
            "final_10_mean_return": value - 0.5,
        },
        resources={
            "duration_seconds": 1.5 + seed,
        },
    )
    write_run_artifacts(
        run,
        manifest=manifest,
        summary=summary,
        resolved_config={
            "seed": seed,
        },
        metrics_rows=[
            {
                "post_return": value,
            }
        ],
    )


def test_bootstrap_mean_ci_is_deterministic() -> None:
    config = BootstrapConfig(
        samples=500,
        confidence=0.9,
        seed=9,
    )
    left = bootstrap_mean_ci(
        [1.0, 2.0, 3.0],
        config=config,
    )
    right = bootstrap_mean_ci(
        [1.0, 2.0, 3.0],
        config=config,
    )
    assert left == right
    assert left["ci_low"] <= left["mean"] <= left["ci_high"]


def test_paper_artifact_pipeline_requires_no_manual_transcription(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    for seed, value in enumerate((1.0, 2.0, 4.0)):
        _fake_run(
            run_root,
            seed=seed,
            value=value,
        )
    suite_dir = run_root / "smoke"
    (suite_dir / "suite_manifest.json").write_text(
        json.dumps(
            {
                "suite": "smoke",
                "suite_revision": 1,
                "git_commit": "abc123",
                "jobs": [1, 2, 3],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "paper"
    report = build_paper_artifacts(
        run_root,
        output,
        bootstrap=BootstrapConfig(
            samples=500,
            seed=10,
        ),
    )
    assert report["schema_version"] == 2
    assert report["runs_aggregated"] == 3
    assert report["bootstrap_groups"] == 1
    assert report["manifests_indexed"] == 1
    for name in report["artifacts"]:
        path = output / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_failed_run_is_excluded_but_completed_missing_metric_fails(
    tmp_path: Path,
) -> None:
    failed = tmp_path / "runs" / "smoke" / "failed__seed_0"
    failed.mkdir(parents=True)
    failed_manifest = RunManifest(
        run_id="failed__seed_0",
        method="demo",
        setting="stationary",
        benchmark="synthetic",
        seed=0,
        git_commit="abc123",
        status="failed",
        metadata={
            "suite": "smoke",
            "job_id": "failed",
            "hypothesis_id": "A",
            "primary_metric": "score",
        },
    )
    (failed / "manifest.json").write_text(
        json.dumps(failed_manifest.to_dict()),
        encoding="utf-8",
    )

    completed = tmp_path / "runs" / "smoke" / "bad__seed_1"
    write_run_artifacts(
        completed,
        manifest=RunManifest(
            run_id="bad__seed_1",
            method="demo",
            setting="stationary",
            benchmark="synthetic",
            seed=1,
            git_commit="abc123",
            metadata={
                "suite": "smoke",
                "job_id": "bad",
                "hypothesis_id": "A",
                "primary_metric": "missing_score",
            },
        ),
        summary=RunSummary(
            run_id="bad__seed_1",
            metrics={
                "other_score": 1.0,
            },
        ),
        resolved_config={
            "seed": 1,
        },
        metrics_rows=[
            {
                "other_score": 1.0,
            }
        ],
    )

    output = tmp_path / "paper"
    try:
        build_paper_artifacts(
            tmp_path / "runs",
            output,
            bootstrap=BootstrapConfig(
                samples=500,
            ),
        )
    except ValueError as exc:
        assert "missing_score" in str(exc)
    else:
        raise AssertionError("completed run with missing primary metric was silently accepted")
