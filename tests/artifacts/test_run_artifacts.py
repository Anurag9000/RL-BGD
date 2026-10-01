import json
from pathlib import Path

import pytest

from rl_bgd.artifacts import (
    RunManifest,
    RunSummary,
    load_run_directory,
    metrics_rows_from_result,
    summarize_runner_result,
    write_run_artifacts,
)


def test_run_artifact_round_trip_preserves_provenance(
    tmp_path: Path,
) -> None:
    manifest = RunManifest(
        run_id="method_a__seed_3",
        method="Method A",
        setting="strict_task_agnostic",
        benchmark="CW10",
        seed=3,
        git_commit="abc123",
        task_order=(
            "task_a",
            "task_b",
        ),
        information_access={
            "receives_task_id": False,
            "receives_task_boundary": False,
        },
    )
    summary = RunSummary(
        run_id=manifest.run_id,
        metrics={
            "final_average": 0.7,
            "forgetting": 0.1,
        },
        task_metrics={
            "task_a": {
                "final_performance": 0.8,
            },
            "task_b": {
                "final_performance": 0.6,
            },
        },
        resources={
            "duration_seconds": 12.5,
        },
    )
    write_run_artifacts(
        tmp_path,
        manifest=manifest,
        summary=summary,
        resolved_config={
            "seed": 3,
            "benchmark": "CW10",
        },
        metrics_rows=[
            {
                "step": 0,
                "return": 0.1,
            },
            {
                "step": 1,
                "return": 0.2,
            },
        ],
    )

    loaded = load_run_directory(tmp_path)
    assert loaded.manifest == manifest
    assert loaded.summary == summary
    assert list(loaded.metrics["return"]) == [
        0.1,
        0.2,
    ]
    assert set(loaded.source_hashes) == {
        "manifest.json",
        "config.yaml",
        "metrics.csv",
        "summary.json",
    }


def test_incomplete_run_is_rejected_by_default(
    tmp_path: Path,
) -> None:
    manifest = RunManifest(
        run_id="failed_seed",
        method="Method A",
        setting="stationary",
        benchmark="LQR",
        seed=0,
        git_commit="abc123",
        status="failed",
    )
    write_run_artifacts(
        tmp_path,
        manifest=manifest,
        summary=RunSummary(
            run_id=manifest.run_id,
            metrics={
                "return": 0.0,
            },
        ),
        resolved_config={
            "seed": 0,
        },
        metrics_rows=[
            {
                "return": 0.0,
            }
        ],
    )
    with pytest.raises(
        ValueError,
        match="refuses incomplete",
    ):
        load_run_directory(tmp_path)


def test_manifest_rejects_string_task_order() -> None:
    payload = {
        "run_id": "run",
        "method": "method",
        "setting": "setting",
        "benchmark": "benchmark",
        "seed": 0,
        "git_commit": "abc",
        "task_order": "not-a-task-list",
    }
    with pytest.raises(ValueError):
        RunManifest.from_dict(payload)


def test_runner_result_converter_extracts_cw_metrics_and_tasks() -> None:
    result = {
        "task_names": [
            "a",
            "b",
        ],
        "return_matrix": [
            [1.0, 2.0],
            [0.8, 3.0],
        ],
        "success_matrix": [
            [0.0, 0.5],
            [0.25, 1.0],
        ],
        "return_summary": {
            "final_average": 1.9,
            "mean_forgetting": 0.1,
            "backward_transfer": -0.1,
            "forgetting_by_task": [
                0.2,
                0.0,
            ],
        },
        "training": {
            "final_10_mean_return": 4.0,
        },
    }
    summary = summarize_runner_result(
        "cw__seed_0",
        result,
        duration_seconds=5.0,
    )
    assert summary.metrics["final_average"] == pytest.approx(1.9)
    assert summary.metrics["forgetting"] == pytest.approx(0.1)
    assert summary.metrics["bwt"] == pytest.approx(-0.1)
    assert summary.metrics["final_10_mean_return"] == pytest.approx(4.0)
    assert summary.task_metrics["a"] == {
        "final_performance": 0.8,
        "success_rate": 0.25,
        "forgetting": 0.2,
    }
    assert summary.resources["duration_seconds"] == pytest.approx(5.0)


def test_timeline_rows_are_preferred_over_scalar_fallback() -> None:
    rows = metrics_rows_from_result(
        {
            "post_return": 1.0,
            "surprise_timeline": [
                {
                    "step": 10,
                    "surprise": 0.5,
                },
                {
                    "step": 20,
                    "surprise": 0.7,
                },
            ],
        }
    )
    assert rows == [
        {
            "series": "surprise_timeline",
            "row_index": 0,
            "step": 10.0,
            "surprise": 0.5,
        },
        {
            "series": "surprise_timeline",
            "row_index": 1,
            "step": 20.0,
            "surprise": 0.7,
        },
    ]


def test_summary_run_id_mismatch_is_detected(
    tmp_path: Path,
) -> None:
    manifest = RunManifest(
        run_id="expected",
        method="method",
        setting="setting",
        benchmark="benchmark",
        seed=0,
        git_commit="abc",
    )
    write_run_artifacts(
        tmp_path,
        manifest=manifest,
        summary=RunSummary(
            run_id="expected",
            metrics={
                "score": 1.0,
            },
        ),
        resolved_config={
            "seed": 0,
        },
        metrics_rows=[
            {
                "score": 1.0,
            }
        ],
    )
    summary_path = tmp_path / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["run_id"] = "tampered"
    summary_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="run_id mismatch",
    ):
        load_run_directory(tmp_path)


def test_matrix_rows_preserve_stage_and_task_identity() -> None:
    rows = metrics_rows_from_result(
        {
            "task_names": [
                "a",
                "b",
            ],
            "return_stage_labels": [
                "after_a",
                "after_b",
            ],
            "return_matrix": [
                [
                    1.0,
                    2.0,
                ],
                [
                    3.0,
                    4.0,
                ],
            ],
        }
    )
    matrix_rows = [
        row
        for row in rows
        if row[
            "series"
        ]
        == "return_matrix"
    ]
    assert len(
        matrix_rows
    ) == 4
    assert matrix_rows[-1] == {
        "series": "return_matrix",
        "row_index": 3,
        "stage_index": 1,
        "task_index": 1,
        "stage_label": "after_b",
        "task_name": "b",
        "value": 4.0,
    }
