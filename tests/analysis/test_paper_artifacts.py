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
) -> None:
    manifest = RunManifest(
        run_id=run_id,
        method=method,
        setting="strict_task_agnostic",
        benchmark="toy_cw",
        seed=seed,
        git_commit="deadbeef",
        status=status,  # type: ignore[arg-type]
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
            "job_id": (
                "method_a"
                if method == "A"
                else "method_b"
            ),
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
                    "final_performance": (
                        final_average
                        + 0.1
                    ),
                    "forgetting": forgetting,
                },
                "task_b": {
                    "final_performance": (
                        final_average
                        - 0.1
                    ),
                    "forgetting": (
                        forgetting
                        / 2.0
                    ),
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
                "return": (
                    final_average
                    - 0.2
                ),
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
    results = (
        tmp_path
        / "results"
    )
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

    output = (
        tmp_path
        / "paper"
    )
    manifest = (
        build_paper_artifacts(
            results,
            output,
            config=PaperArtifactConfig(
                bootstrap_resamples=200,
                seed=17,
                figure_formats=(
                    "png",
                ),
            ),
        )
    )

    assert manifest[
        "run_count"
    ] == 4
    assert manifest[
        "integrity"
    ][
        "manual_result_transcription"
    ] is False
    assert len(
        manifest[
            "source_runs"
        ]
    ) == 4
    assert all(
        source[
            "source_hashes"
        ]
        for source in manifest[
            "source_runs"
        ]
    )

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
            assert (
                output
                / "tables"
                / f"{stem}.{extension}"
            ).is_file()

    aggregate = pd.read_csv(
        output
        / "tables"
        / "aggregate_statistics.csv"
    )
    a_final = aggregate[
        (
            aggregate[
                "method"
            ]
            == "A"
        )
        & (
            aggregate[
                "metric"
            ]
            == "final_average"
        )
    ].iloc[0]
    assert a_final[
        "mean"
    ] == pytest.approx(
        0.7
    )
    assert a_final[
        "n"
    ] == 2

    paired = pd.read_csv(
        output
        / "tables"
        / "paired_differences.csv"
    )
    difference = paired[
        paired[
            "metric"
        ]
        == "final_average"
    ].iloc[0]
    assert difference[
        "mean"
    ] == pytest.approx(
        0.1
    )
    assert difference[
        "n"
    ] == 2

    task_stats = pd.read_csv(
        output
        / "tables"
        / "hierarchical_task_statistics.csv"
    )
    assert (
        "final_performance"
        in set(
            task_stats[
                "metric"
            ]
        )
    )

    assert (
        output
        / "figures"
        / "final_performance.png"
    ).is_file()
    assert (
        output
        / "figures"
        / "compute_performance.png"
    ).is_file()
    assert any(
        path.name.startswith(
            "learning_curve_"
        )
        for path in (
            output
            / "figures"
        ).glob(
            "*.png"
        )
    )

    saved_manifest = json.loads(
        (
            output
            / "paper_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert saved_manifest[
        "run_count"
    ] == 4


def test_paper_builder_refuses_failed_seed(
    tmp_path: Path,
) -> None:
    results = (
        tmp_path
        / "results"
    )
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
            tmp_path
            / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=(
                    "png",
                ),
            ),
        )


def test_paper_builder_rejects_metric_missing_from_one_matched_seed(
    tmp_path: Path,
) -> None:
    results = (
        tmp_path
        / "results"
    )
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
    summary_path = (
        results
        / "a1"
        / "summary.json"
    )
    payload = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )
    del payload[
        "metrics"
    ][
        "forgetting"
    ]
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
            tmp_path
            / "paper",
            config=PaperArtifactConfig(
                bootstrap_resamples=20,
                figure_formats=(
                    "png",
                ),
            ),
        )



def test_paper_builder_generates_continual_matrix_and_timeline_figures(
    tmp_path: Path,
) -> None:
    results = (
        tmp_path
        / "results"
    )
    for seed, offset in (
        (0, 0.0),
        (1, 0.2),
    ):
        run_id = (
            f"cw_seed_{seed}"
        )
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
                    "surprise_smoothed": (
                        0.2 + offset
                    ),
                    "retention_lambda": (
                        0.9 - offset / 10.0
                    ),
                    "critic1_sigma_mean": (
                        0.1 + offset / 10.0
                    ),
                },
                {
                    "step": 20,
                    "surprise_smoothed": (
                        0.5 + offset
                    ),
                    "retention_lambda": (
                        0.7 - offset / 10.0
                    ),
                    "critic1_sigma_mean": (
                        0.12 + offset / 10.0
                    ),
                },
            ],
        }
        write_run_artifacts(
            results
            / run_id,
            manifest=RunManifest(
                run_id=run_id,
                method="Adaptive-BGD",
                setting=(
                    "strict_task_agnostic"
                ),
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
                    "job_id": (
                        "adaptive_cw"
                    ),
                    "primary_metric": (
                        "final_average"
                    ),
                },
            ),
            summary=RunSummary(
                run_id=run_id,
                metrics={
                    "final_average": (
                        1.15 + offset
                    ),
                },
                task_metrics={
                    "task_a": {
                        "final_performance": (
                            0.8 + offset
                        ),
                    },
                    "task_b": {
                        "final_performance": (
                            1.5 + offset
                        ),
                    },
                },
            ),
            resolved_config={
                "seed": seed,
            },
            metrics_rows=(
                metrics_rows_from_result(
                    result
                )
            ),
        )

    output = (
        tmp_path
        / "paper"
    )
    manifest = build_paper_artifacts(
        results,
        output,
        config=PaperArtifactConfig(
            bootstrap_resamples=100,
            seed=31,
            figure_formats=(
                "png",
            ),
        ),
    )

    generated = set(
        manifest[
            "generated_figures"
        ]
    )
    assert any(
        "continual_matrix_return_matrix"
        in name
        for name in generated
    )
    assert any(
        "continual_matrix_success_matrix"
        in name
        for name in generated
    )
    assert any(
        "adaptation_curve_return_matrix"
        in name
        for name in generated
    )
    assert any(
        "timeline_surprise_timeline"
        in name
        for name in generated
    )
