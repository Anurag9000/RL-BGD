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
    assert summary.task_metrics["b"] == {
        "final_performance": 3.0,
        "success_rate": 1.0,
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
    matrix_rows = [row for row in rows if row["series"] == "return_matrix"]
    assert len(matrix_rows) == 4
    assert matrix_rows[-1] == {
        "series": "return_matrix",
        "row_index": 3,
        "stage_index": 1,
        "task_index": 1,
        "stage_label": "after_b",
        "task_name": "b",
        "value": 4.0,
    }


def test_repeated_task_names_preserve_occurrence_metrics() -> None:
    result = {
        "task_names": [
            "a",
            "b",
            "a",
            "b",
        ],
        "return_matrix": [
            [
                1.0,
                0.0,
                1.1,
                0.1,
            ],
            [
                0.9,
                2.0,
                1.0,
                2.1,
            ],
            [
                0.8,
                1.9,
                3.0,
                2.0,
            ],
            [
                0.7,
                1.8,
                2.9,
                4.0,
            ],
        ],
        "success_matrix": [
            [
                0.4,
                0.0,
                0.5,
                0.0,
            ],
            [
                0.3,
                0.6,
                0.4,
                0.7,
            ],
            [
                0.2,
                0.5,
                0.8,
                0.6,
            ],
            [
                0.1,
                0.4,
                0.7,
                0.9,
            ],
        ],
        "return_summary": {
            "final_average": 2.35,
            "mean_forgetting": 0.2,
            "backward_transfer": -0.2,
            "forgetting_by_task": [
                0.3,
                0.2,
                0.1,
            ],
        },
    }

    summary = summarize_runner_result(
        "cw20_like",
        result,
    )

    assert tuple(summary.task_metrics) == (
        "a#occurrence_1",
        "b#occurrence_1",
        "a#occurrence_2",
        "b#occurrence_2",
    )
    assert summary.task_metrics["a#occurrence_1"] == {
        "final_performance": 0.7,
        "success_rate": 0.1,
        "forgetting": 0.3,
    }
    assert summary.task_metrics["b#occurrence_1"]["forgetting"] == pytest.approx(0.2)
    assert summary.task_metrics["a#occurrence_2"]["forgetting"] == pytest.approx(0.1)
    assert "forgetting" not in summary.task_metrics["b#occurrence_2"]


def _valid_manifest_payload() -> dict[str, object]:
    return {
        "run_id": "run",
        "method": "method",
        "setting": "stationary",
        "benchmark": "LQR",
        "seed": 0,
        "git_commit": "abc123",
    }


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("run_id", 7),
        ("method", 7),
        ("setting", 7),
        ("benchmark", 7),
        ("git_commit", 7),
        ("status", 7),
        ("config_file", 7),
        ("metrics_file", 7),
        ("summary_file", 7),
        ("seed", True),
        ("schema_version", True),
    ],
)
def test_manifest_rejects_identity_string_coercion(
    field: str,
    invalid: object,
) -> None:
    payload = _valid_manifest_payload()
    payload[field] = invalid
    with pytest.raises(ValueError, match="invalid run manifest payload"):
        RunManifest.from_dict(payload)


def test_manifest_rejects_non_string_task_and_information_keys() -> None:
    payload = _valid_manifest_payload()
    payload["task_order"] = ["valid", 3]
    with pytest.raises(ValueError, match="invalid run manifest payload"):
        RunManifest.from_dict(payload)

    payload = _valid_manifest_payload()
    payload["information_access"] = {1: False}
    with pytest.raises(ValueError, match="invalid run manifest payload"):
        RunManifest.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"run_id": 7},
        {"run_id": "run", "metrics": {"score": "1.0"}},
        {"run_id": "run", "metrics": {"score": True}},
        {"run_id": "run", "task_metrics": {1: {"score": 1.0}}},
        {"run_id": "run", "task_metrics": {"task": {1: 1.0}}},
        {"run_id": "run", "resources": {1: 1.0}},
        {"run_id": "run", "schema_version": True},
    ],
)
def test_summary_rejects_schema_coercion(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid run summary payload"):
        RunSummary.from_dict(payload)


def test_direct_summary_validation_rejects_boolean_metric() -> None:
    summary = RunSummary(
        run_id="run",
        metrics={"score": True},  # type: ignore[dict-item]
    )
    with pytest.raises(TypeError, match="must be numeric"):
        summary.validate()


@pytest.mark.parametrize(
    ("config_file", "metrics_file", "summary_file"),
    [
        ("manifest.json", "metrics.csv", "summary.json"),
        ("result.txt", "result.txt", "summary.json"),
        ("config.yaml", "metrics.csv", "config.yaml"),
        ("folder//data.yaml", "folder/data.yaml", "summary.json"),
    ],
)
def test_run_manifest_rejects_colliding_artifact_paths(
    config_file: str,
    metrics_file: str,
    summary_file: str,
) -> None:
    manifest = RunManifest(
        run_id="collision",
        method="SAC",
        setting="stationary",
        benchmark="LQR",
        seed=0,
        git_commit="abc",
        config_file=config_file,
        metrics_file=metrics_file,
        summary_file=summary_file,
    )
    with pytest.raises(ValueError, match="filenames must be distinct"):
        manifest.validate()


def test_run_loader_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "inside"
    manifest = RunManifest(
        run_id="run",
        method="SAC",
        setting="stationary",
        benchmark="LQR",
        seed=0,
        git_commit="abc",
    )
    write_run_artifacts(
        root,
        manifest=manifest,
        summary=RunSummary(run_id="run", metrics={"score": 1.0}),
        resolved_config={"seed": 0},
        metrics_rows=[{"score": 1.0}],
    )
    outside = tmp_path / "outside.yaml"
    outside.write_text("seed: 999\n", encoding="utf-8")
    (root / "config.yaml").unlink()
    (root / "config.yaml").symlink_to(outside)

    with pytest.raises(ValueError, match="resolves outside"):
        load_run_directory(root)


def test_run_writer_rejects_symlink_escape_without_writing_outside(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inside"
    root.mkdir()
    outside = tmp_path / "outside.yaml"
    original = "unrelated: true\n"
    outside.write_text(original, encoding="utf-8")
    (root / "config.yaml").symlink_to(outside)

    with pytest.raises(ValueError, match="resolves outside"):
        write_run_artifacts(
            root,
            manifest=RunManifest(
                run_id="run",
                method="SAC",
                setting="stationary",
                benchmark="LQR",
                seed=0,
                git_commit="abc",
            ),
            summary=RunSummary(run_id="run", metrics={"score": 1.0}),
            resolved_config={"seed": 0},
            metrics_rows=[{"score": 1.0}],
        )

    assert outside.read_text(encoding="utf-8") == original
    assert not (root / "manifest.json").exists()


def test_run_loader_rejects_duplicate_metrics_csv_headers(tmp_path: Path) -> None:
    root = tmp_path / "run"
    write_run_artifacts(
        root,
        manifest=RunManifest(
            run_id="run",
            method="SAC",
            setting="stationary",
            benchmark="LQR",
            seed=0,
            git_commit="abc",
        ),
        summary=RunSummary(run_id="run", metrics={"score": 1.0}),
        resolved_config={"seed": 0},
        metrics_rows=[{"score": 1.0}],
    )
    (root / "metrics.csv").write_text(
        "score,score\n1.0,2.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate column names"):
        load_run_directory(root)


@pytest.mark.parametrize("invalid_name", ["", 0, None])
def test_run_writer_rejects_invalid_metric_column_names(
    tmp_path: Path,
    invalid_name: object,
) -> None:
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="column names must be non-empty strings"):
        write_run_artifacts(
            root,
            manifest=RunManifest(
                run_id="run",
                method="SAC",
                setting="stationary",
                benchmark="LQR",
                seed=0,
                git_commit="abc",
            ),
            summary=RunSummary(run_id="run", metrics={"score": 1.0}),
            resolved_config={"seed": 0},
            metrics_rows=[{invalid_name: 1.0}],  # type: ignore[dict-item]
        )
    assert not root.exists()

