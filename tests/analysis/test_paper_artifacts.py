import json
from pathlib import Path

import pandas as pd
import pytest

from rl_bgd.analysis.paper_artifacts import (
    PaperArtifactConfig,
    build_paper_artifacts,
)
from rl_bgd.artifacts import (
    RunManifest,
    RunSummary,
    metrics_rows_from_result,
    write_run_artifacts,
)


def _write_run(
    root: Path,
    *,
    run_id: str,
    method: str,
    seed: int,
    final_average: float,
    forgetting: float,
    duration: float,
    status: str = "completed",
    git_commit: str = "deadbeef",
    receives_task_id: bool = False,
) -> None:
    manifest = RunManifest(
        run_id=run_id,
        method=method,
        setting="strict_task_agnostic",
        benchmark="toy_cw",
        seed=seed,
        git_commit=git_commit,
        status=status,  # type: ignore[arg-type]
        task_order=(
            "task_a",
            "task_b",
        ),
        information_access={
            "receives_task_id": receives_task_id,
            "receives_task_boundary": False,
            "receives_environment_context": False,
        },
        metadata={
            "job_id": ("method_a" if method == "A" else "method_b"),
        },
    )
    write_run_artifacts(
        root / run_id,
        manifest=manifest,
        summary=RunSummary(
            run_id=run_id,
            metrics={
                "final_average": final_average,
                "forgetting": forgetting,
            },
            task_metrics={
                "task_a": {
                    "final_performance": (final_average + 0.1),
                    "forgetting": forgetting,
                },
                "task_b": {
                    "final_performance": (final_average - 0.1),
                    "forgetting": (forgetting / 2.0),
                },
            },
            resources={
                "duration_seconds": duration,
            },
        ),
        resolved_config={
            "seed": seed,
            "method": method,
        },
        metrics_rows=[
            {
                "step": 0,
                "return": (final_average - 0.2),
            },
            {
                "step": 10,
                "return": final_average,
            },
        ],
    )


def test_paper_builder_generates_provenance_tables_statistics_and_figures(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_run(
        results,
        run_id="a_seed_0",
        method="A",
        seed=0,
        final_average=0.6,
        forgetting=0.2,
        duration=10.0,
    )
    _write_run(
        results,
        run_id="a_seed_1",
        method="A",
        seed=1,
        final_average=0.8,
        forgetting=0.1,
        duration=12.0,
    )
    _write_run(
        results,
        run_id="b_seed_0",
        method="B",
        seed=0,
        final_average=0.5,
        forgetting=0.3,
        duration=8.0,
    )
    _write_run(
        results,
        run_id="b_seed_1",
        method="B",
        seed=1,
        final_average=0.7,
        forgetting=0.2,
        duration=9.0,
    )

    output = tmp_path / "paper"
    manifest = build_paper_artifacts(
        results,
        output,
        config=PaperArtifactConfig(
            bootstrap_resamples=200,
            seed=17,
            figure_formats=("png",),
        ),
    )

    assert manifest["run_count"] == 4
    assert manifest["integrity"]["manual_result_transcription"] is False
    assert len(manifest["source_runs"]) == 4
    assert all(source["source_hashes"] for source in manifest["source_runs"])

    for stem in (
        "aggregate_statistics",
        "hierarchical_task_statistics",
        "paired_differences",
        "results_summary",
        "information_access",
        "run_index",
    ):
        for extension in (
            "csv",
            "md",
            "tex",
        ):
            assert (output / "tables" / f"{stem}.{extension}").is_file()

    aggregate = pd.read_csv(output / "tables" / "aggregate_statistics.csv")
    a_final = aggregate[
        (aggregate["method"] == "A") & (aggregate["metric"] == "final_average")
    ].iloc[0]
    assert a_final["mean"] == pytest.approx(0.7)
    assert a_final["n"] == 2

    paired = pd.read_csv(output / "tables" / "paired_differences.csv")
    difference = paired[paired["metric"] == "final_average"].iloc[0]
    assert difference["mean"] == pytest.approx(0.1)
    assert difference["n"] == 2

    task_stats = pd.read_csv(output / "tables" / "hierarchical_task_statistics.csv")
    assert "final_performance" in set(task_stats["metric"])

    assert (output / "figures" / "final_performance.png").is_file()
    assert (output / "figures" / "compute_performance.png").is_file()
    assert any(
        path.name.startswith("learning_curve_") for path in (output / "figures").glob("*.png")
    )

    saved_manifest = json.loads((output / "paper_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["run_count"] == 4


def test_paper_builder_refuses_failed_seed(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_run(
        results,
        run_id="ok",
        method="A",
        seed=0,
        final_average=1.0,
        forgetting=0.0,
        duration=1.0,
    )
    _write_run(
        results,
        run_id="failed",
        method="A",
        seed=1,
        final_average=0.0,
        forgetting=0.0,
        duration=1.0,
        status="failed",
    )
    with pytest.raises(
        ValueError,
        match="refuses incomplete",
    ):
        build_paper_artifacts(
            results,
            tmp_path / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=("png",),
            ),
        )


def test_paper_builder_rejects_metric_missing_from_one_matched_seed(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_run(
        results,
        run_id="a0",
        method="A",
        seed=0,
        final_average=1.0,
        forgetting=0.1,
        duration=1.0,
    )
    _write_run(
        results,
        run_id="a1",
        method="A",
        seed=1,
        final_average=1.1,
        forgetting=0.2,
        duration=1.0,
    )
    summary_path = results / "a1" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    del payload["metrics"]["forgetting"]
    summary_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing for a subset",
    ):
        build_paper_artifacts(
            results,
            tmp_path / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=("png",),
            ),
        )


def test_paper_builder_generates_continual_matrix_and_timeline_figures(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    for seed, offset in (
        (0, 0.0),
        (1, 0.2),
    ):
        run_id = f"cw_seed_{seed}"
        result = {
            "task_names": [
                "task_a",
                "task_b",
            ],
            "return_stage_labels": [
                "after_a",
                "after_b",
            ],
            "return_matrix": [
                [
                    1.0 + offset,
                    0.5 + offset,
                ],
                [
                    0.8 + offset,
                    1.5 + offset,
                ],
            ],
            "success_stage_labels": [
                "after_a",
                "after_b",
            ],
            "success_matrix": [
                [
                    0.5,
                    0.0,
                ],
                [
                    0.4,
                    0.8,
                ],
            ],
            "surprise_timeline": [
                {
                    "step": 10,
                    "surprise_smoothed": (0.2 + offset),
                    "retention_lambda": (0.9 - offset / 10.0),
                    "critic1_sigma_mean": (0.1 + offset / 10.0),
                },
                {
                    "step": 20,
                    "surprise_smoothed": (0.5 + offset),
                    "retention_lambda": (0.7 - offset / 10.0),
                    "critic1_sigma_mean": (0.12 + offset / 10.0),
                },
            ],
            "post_shift_loss_timeline": [
                {
                    "step": 0,
                    "normalized_target_loss": 1.0,
                },
                {
                    "step": 1,
                    "normalized_target_loss": (0.6 + offset / 10.0),
                },
            ],
        }
        write_run_artifacts(
            results / run_id,
            manifest=RunManifest(
                run_id=run_id,
                method="Adaptive-BGD",
                setting=("strict_task_agnostic"),
                benchmark="toy_cw",
                seed=seed,
                git_commit="deadbeef",
                task_order=(
                    "task_a",
                    "task_b",
                ),
                information_access={
                    "receives_task_id": False,
                    "receives_task_boundary": False,
                    "receives_environment_context": False,
                },
                metadata={
                    "job_id": ("adaptive_cw"),
                    "primary_metric": ("final_average"),
                },
            ),
            summary=RunSummary(
                run_id=run_id,
                metrics={
                    "final_average": (1.15 + offset),
                },
                task_metrics={
                    "task_a": {
                        "final_performance": (0.8 + offset),
                    },
                    "task_b": {
                        "final_performance": (1.5 + offset),
                    },
                },
            ),
            resolved_config={
                "seed": seed,
            },
            metrics_rows=(metrics_rows_from_result(result)),
        )

    output = tmp_path / "paper"
    manifest = build_paper_artifacts(
        results,
        output,
        config=PaperArtifactConfig(
            bootstrap_resamples=100,
            seed=31,
            figure_formats=("png",),
        ),
    )

    generated = set(manifest["generated_figures"])
    assert any("continual_matrix_return_matrix" in name for name in generated)
    assert any("continual_matrix_success_matrix" in name for name in generated)
    assert any("adaptation_curve_return_matrix" in name for name in generated)
    assert any("timeline_surprise_timeline" in name for name in generated)
    assert any("timeline_post_shift_loss_timeline" in name for name in generated)


def test_paper_builder_aggregates_only_declared_suite_metrics(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    for seed, score in (
        (0, 0.4),
        (1, 0.6),
    ):
        run_id = f"declared_{seed}"
        write_run_artifacts(
            results / run_id,
            manifest=RunManifest(
                run_id=run_id,
                method="DeclaredMethod",
                setting="matched",
                benchmark="toy",
                seed=seed,
                git_commit="deadbeef",
                information_access={
                    "receives_task_id": False,
                    "receives_task_boundary": False,
                    "receives_environment_context": False,
                },
                metadata={
                    "suite": "declared_suite",
                    "job_id": "declared_job",
                    "primary_metric": "score",
                    "secondary_metrics": [
                        "auxiliary",
                        "phase_summaries",
                    ],
                },
            ),
            summary=RunSummary(
                run_id=run_id,
                metrics={
                    "score": score,
                    "auxiliary": score + 0.1,
                    "steps": 600.0,
                    "phase_steps": 120.0,
                    "horizon": 32.0,
                    "phases": 5.0,
                },
                resources={
                    "duration_seconds": 10.0 + seed,
                },
            ),
            resolved_config={
                "steps": 600,
                "phase_steps": 120,
                "horizon": 32,
            },
            metrics_rows=[
                {
                    "series": "summary",
                    "row_index": 0,
                    "score": score,
                }
            ],
        )

    output = tmp_path / "paper"
    build_paper_artifacts(
        results,
        output,
        config=PaperArtifactConfig(
            bootstrap_resamples=50,
            seed=41,
            figure_formats=("png",),
        ),
    )

    aggregate = pd.read_csv(
        output
        / "tables"
        / "aggregate_statistics.csv"
    )
    metrics = set(
        aggregate["metric"]
    )
    assert metrics == {
        "score",
        "auxiliary",
        "duration_seconds",
    }
    assert "steps" not in metrics
    assert "phase_steps" not in metrics
    assert "horizon" not in metrics
    assert "phases" not in metrics
    assert "phase_summaries" not in metrics


def test_paper_builder_accepts_declared_resource_primary_without_duplicate(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    for seed, duration in (
        (0, 2.0),
        (1, 4.0),
    ):
        run_id = f"compute_{seed}"
        write_run_artifacts(
            results / run_id,
            manifest=RunManifest(
                run_id=run_id,
                method="ComputeMethod",
                setting="stationary_compute",
                benchmark="toy",
                seed=seed,
                git_commit="deadbeef",
                information_access={
                    "receives_task_id": False,
                    "receives_task_boundary": False,
                    "receives_environment_context": False,
                },
                metadata={
                    "suite": "compute_analysis",
                    "job_id": "compute_job",
                    "primary_metric": "duration_seconds",
                    "secondary_metrics": ["post_return"],
                },
            ),
            summary=RunSummary(
                run_id=run_id,
                metrics={
                    "post_return": 1.0 + seed,
                },
                resources={
                    "duration_seconds": duration,
                },
            ),
            resolved_config={
                "seed": seed,
            },
            metrics_rows=[
                {
                    "series": "summary",
                    "row_index": 0,
                    "post_return": 1.0 + seed,
                }
            ],
        )

    output = tmp_path / "paper"
    build_paper_artifacts(
        results,
        output,
        config=PaperArtifactConfig(
            bootstrap_resamples=50,
            seed=43,
            figure_formats=("png",),
        ),
    )

    aggregate = pd.read_csv(
        output / "tables" / "aggregate_statistics.csv"
    )
    metrics = set(aggregate["metric"])
    assert "duration_seconds" in metrics
    assert "post_return" in metrics
    assert "resource.duration_seconds" not in metrics
    duration_row = aggregate[
        aggregate["metric"] == "duration_seconds"
    ].iloc[0]
    assert duration_row["mean"] == pytest.approx(3.0)


def _write_contract_seed(
    root: Path,
    *,
    seed: int,
    git_commit: str = "deadbeef",
    receives_task_id: bool = False,
) -> None:
    run_id = f"contract_{seed}"
    write_run_artifacts(
        root / run_id,
        manifest=RunManifest(
            run_id=run_id,
            method="ContractMethod",
            setting="strict_task_agnostic",
            benchmark="toy",
            seed=seed,
            git_commit=git_commit,
            information_access={
                "receives_task_id": receives_task_id,
                "receives_task_boundary": False,
                "receives_environment_context": False,
            },
            metadata={
                "suite": "contract_suite",
                "job_id": "contract_job",
                "hypothesis_id": "A",
                "target": "rl_bgd.runners.example:run",
                "source_config_path": "configs/example.yaml",
                "primary_metric": "score",
                "secondary_metrics": [],
            },
        ),
        summary=RunSummary(
            run_id=run_id,
            metrics={
                "score": 1.0 + seed,
            },
        ),
        resolved_config={
            "seed": seed,
        },
        metrics_rows=[
            {
                "series": "summary",
                "row_index": 0,
                "score": 1.0 + seed,
            }
        ],
    )


def test_paper_builder_rejects_mixed_seed_information_access(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_contract_seed(
        results,
        seed=0,
        receives_task_id=False,
    )
    _write_contract_seed(
        results,
        seed=1,
        receives_task_id=True,
    )

    with pytest.raises(
        ValueError,
        match="information_access",
    ):
        build_paper_artifacts(
            results,
            tmp_path / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=("png",),
            ),
        )


def test_paper_builder_rejects_mixed_seed_git_commits(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_contract_seed(
        results,
        seed=0,
        git_commit="deadbeef",
    )
    _write_contract_seed(
        results,
        seed=1,
        git_commit="cafebabe",
    )

    with pytest.raises(
        ValueError,
        match="git_commit",
    ):
        build_paper_artifacts(
            results,
            tmp_path / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=("png",),
            ),
        )


def test_paper_builder_rejects_unmatched_paired_seed_sets(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_run(
        results,
        run_id="a0",
        method="A",
        seed=0,
        final_average=1.0,
        forgetting=0.1,
        duration=1.0,
    )
    _write_run(
        results,
        run_id="a1",
        method="A",
        seed=1,
        final_average=1.1,
        forgetting=0.1,
        duration=1.0,
    )
    _write_run(
        results,
        run_id="b0",
        method="B",
        seed=0,
        final_average=0.9,
        forgetting=0.2,
        duration=1.0,
    )

    with pytest.raises(
        ValueError,
        match="different seed sets",
    ):
        build_paper_artifacts(
            results,
            tmp_path / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=("png",),
            ),
        )


def test_paper_builder_rejects_cross_method_git_revision_mismatch(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_run(
        results,
        run_id="a0",
        method="A",
        seed=0,
        final_average=1.0,
        forgetting=0.1,
        duration=1.0,
        git_commit="deadbeef",
    )
    _write_run(
        results,
        run_id="b0",
        method="B",
        seed=0,
        final_average=0.9,
        forgetting=0.2,
        duration=1.0,
        git_commit="cafebabe",
    )

    with pytest.raises(
        ValueError,
        match="different git commits",
    ):
        build_paper_artifacts(
            results,
            tmp_path / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=("png",),
            ),
        )


def test_paper_builder_rejects_cross_method_information_access_mismatch(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_run(
        results,
        run_id="a0",
        method="A",
        seed=0,
        final_average=1.0,
        forgetting=0.1,
        duration=1.0,
        receives_task_id=False,
    )
    _write_run(
        results,
        run_id="b0",
        method="B",
        seed=0,
        final_average=0.9,
        forgetting=0.2,
        duration=1.0,
        receives_task_id=True,
    )

    with pytest.raises(
        ValueError,
        match="different information access",
    ):
        build_paper_artifacts(
            results,
            tmp_path / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=("png",),
            ),
        )


def test_paper_builder_rejects_mismatched_nonseed_invocation_contract(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    for seed, learning_rate in (
        (0, 3e-4),
        (1, 1e-3),
    ):
        run_id = f"contract_{seed}"
        write_run_artifacts(
            results / run_id,
            manifest=RunManifest(
                run_id=run_id,
                method="ContractMethod",
                setting="stationary",
                benchmark="toy",
                seed=seed,
                git_commit="deadbeef",
                information_access={
                    "receives_task_id": False,
                    "receives_task_boundary": False,
                    "receives_environment_context": False,
                },
                metadata={
                    "suite": "contract_suite",
                    "job_id": "contract_job",
                    "hypothesis_id": "TEST",
                    "target": "module:runner",
                    "primary_metric": "score",
                    "secondary_metrics": [],
                    "source_config_path": None,
                    "contract_kwargs": {
                        "learning_rate": learning_rate,
                        "steps": 100,
                    },
                },
            ),
            summary=RunSummary(
                run_id=run_id,
                metrics={
                    "score": 1.0 + seed,
                },
            ),
            resolved_config={
                "seed": seed,
                "learning_rate": learning_rate,
                "steps": 100,
            },
            metrics_rows=[
                {
                    "series": "summary",
                    "row_index": 0,
                    "score": 1.0 + seed,
                }
            ],
        )

    with pytest.raises(
        ValueError,
        match="contract_kwargs",
    ):
        build_paper_artifacts(
            results,
            tmp_path / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=50,
                seed=47,
                figure_formats=("png",),
            ),
        )


def _write_paired_contract_run(
    root: Path,
    *,
    method: str,
    job_id: str,
    primary_metric: str,
    suite_revision: int,
) -> None:
    run_id = f"{job_id}_0"
    write_run_artifacts(
        root / run_id,
        manifest=RunManifest(
            run_id=run_id,
            method=method,
            setting="matched",
            benchmark="toy",
            seed=0,
            git_commit="deadbeef",
            information_access={
                "receives_task_id": False,
                "receives_task_boundary": False,
                "receives_environment_context": False,
            },
            metadata={
                "suite": "paired_contract_suite",
                "suite_revision": suite_revision,
                "job_id": job_id,
                "hypothesis_id": "TEST",
                "target": f"module:{job_id}",
                "primary_metric": primary_metric,
                "secondary_metrics": [],
                "source_config_path": None,
                "contract_kwargs": {},
            },
        ),
        summary=RunSummary(
            run_id=run_id,
            metrics={
                "score": 1.0,
                "alternate_score": 2.0,
            },
        ),
        resolved_config={},
        metrics_rows=[
            {
                "series": "summary",
                "row_index": 0,
                "score": 1.0,
                "alternate_score": 2.0,
            }
        ],
    )


def test_paper_builder_rejects_paired_primary_metric_mismatch(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_paired_contract_run(
        results,
        method="A",
        job_id="left",
        primary_metric="score",
        suite_revision=1,
    )
    _write_paired_contract_run(
        results,
        method="B",
        job_id="right",
        primary_metric="alternate_score",
        suite_revision=1,
    )

    with pytest.raises(
        ValueError,
        match="different primary metrics",
    ):
        build_paper_artifacts(
            results,
            tmp_path / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=("png",),
            ),
        )


def test_paper_builder_rejects_paired_suite_revision_mismatch(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _write_paired_contract_run(
        results,
        method="A",
        job_id="left",
        primary_metric="score",
        suite_revision=1,
    )
    _write_paired_contract_run(
        results,
        method="B",
        job_id="right",
        primary_metric="score",
        suite_revision=2,
    )

    with pytest.raises(
        ValueError,
        match="different suite revisions",
    ):
        build_paper_artifacts(
            results,
            tmp_path / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=("png",),
            ),
        )



def _write_grouped_comparison_run(
    root: Path,
    *,
    run_id: str,
    method: str,
    job_id: str,
    comparison_group: str,
    seed: int,
    score: float,
) -> None:
    write_run_artifacts(
        root
        / run_id,
        manifest=RunManifest(
            run_id=run_id,
            method=method,
            setting="stationary",
            benchmark="shared_benchmark",
            seed=seed,
            git_commit="deadbeef",
            information_access={
                "receives_task_id": False,
                "receives_task_boundary": False,
                "receives_environment_context": False,
            },
            metadata={
                "suite": "comparison_suite",
                "suite_revision": 2,
                "job_id": job_id,
                "comparison_group": comparison_group,
                "hypothesis_id": "TEST",
                "target": (
                    "rl_bgd.runners.example:run"
                ),
                "source_config_path": None,
                "primary_metric": "score",
                "secondary_metrics": [],
                "contract_kwargs": {
                    "steps": 10,
                },
            },
        ),
        summary=RunSummary(
            run_id=run_id,
            metrics={
                "score": score,
            },
        ),
        resolved_config={
            "seed": seed,
        },
        metrics_rows=[
            {
                "series": "summary",
                "row_index": 0,
                "score": score,
            }
        ],
    )


def test_paired_statistics_respect_explicit_comparison_groups(
    tmp_path: Path,
) -> None:
    results = (
        tmp_path
        / "results"
    )
    for seed in (
        0,
        1,
    ):
        _write_grouped_comparison_run(
            results,
            run_id=f"sac_adam_{seed}",
            method="SAC-Adam",
            job_id="sac_adam",
            comparison_group="sac_family",
            seed=seed,
            score=1.0 + seed,
        )
        _write_grouped_comparison_run(
            results,
            run_id=f"sac_bgd_{seed}",
            method="SAC-BGD",
            job_id="sac_bgd",
            comparison_group="sac_family",
            seed=seed,
            score=1.2 + seed,
        )
        _write_grouped_comparison_run(
            results,
            run_id=f"ppo_adam_{seed}",
            method="PPO-Adam",
            job_id="ppo_adam",
            comparison_group="ppo_family",
            seed=seed,
            score=0.5 + seed,
        )
        _write_grouped_comparison_run(
            results,
            run_id=f"ppo_bgd_{seed}",
            method="PPO-BGD",
            job_id="ppo_bgd",
            comparison_group="ppo_family",
            seed=seed,
            score=0.7 + seed,
        )

    output = (
        tmp_path
        / "paper"
    )
    build_paper_artifacts(
        results,
        output,
        config=PaperArtifactConfig(
            bootstrap_resamples=50,
            seed=47,
            figure_formats=(
                "png",
            ),
        ),
    )
    paired = pd.read_csv(
        output
        / "tables"
        / "paired_differences.csv"
    )
    assert set(
        paired[
            "comparison_group"
        ]
    ) == {
        "sac_family",
        "ppo_family",
    }
    method_pairs = {
        frozenset(
            (
                row.left_method,
                row.right_method,
            )
        )
        for row in paired.itertuples()
        if row.metric
        == "score"
    }
    assert method_pairs == {
        frozenset(
            (
                "SAC-Adam",
                "SAC-BGD",
            )
        ),
        frozenset(
            (
                "PPO-Adam",
                "PPO-BGD",
            )
        ),
    }
