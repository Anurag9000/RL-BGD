"""Strict run-directory schema for reproducible paper aggregation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

RUN_SCHEMA_VERSION = 1
RunStatus = Literal["completed", "failed", "partial"]


def _require_int(
    value: object,
    *,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_nonempty_string(
    value: object,
    *,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_float(
    value: object,
    *,
    name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _finite_mapping(
    values: Mapping[str, object],
    *,
    name: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        result[key] = _require_float(
            value,
            name=f"{name} value for {key}",
        )
    return result


@dataclass(frozen=True)
class RunManifest:
    """Immutable provenance required for every experiment run."""

    run_id: str
    method: str
    setting: str
    benchmark: str
    seed: int
    git_commit: str
    status: RunStatus = "completed"
    task_order: tuple[str, ...] = ()
    information_access: dict[str, bool] = field(default_factory=dict)
    config_file: str = "config.yaml"
    metrics_file: str = "metrics.csv"
    summary_file: str = "summary.json"
    schema_version: int = RUN_SCHEMA_VERSION
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        schema_version = _require_int(
            self.schema_version,
            name="run manifest schema version",
        )
        if schema_version != RUN_SCHEMA_VERSION:
            raise ValueError("unsupported run-manifest schema version")
        for name, value in (
            ("run_id", self.run_id),
            ("method", self.method),
            ("setting", self.setting),
            ("benchmark", self.benchmark),
            ("git_commit", self.git_commit),
        ):
            _require_nonempty_string(
                value,
                name=f"run manifest {name}",
            )
        seed = _require_int(
            self.seed,
            name="run manifest seed",
        )
        if seed < 0:
            raise ValueError("run manifest seed must be non-negative")
        if not isinstance(self.status, str) or self.status not in {
            "completed",
            "failed",
            "partial",
        }:
            raise ValueError(f"unsupported run status: {self.status}")
        if any(not isinstance(task, str) or not task.strip() for task in self.task_order):
            raise ValueError("task_order entries must be non-empty strings")
        if any(
            not isinstance(key, str) or not key or not isinstance(value, bool)
            for key, value in self.information_access.items()
        ):
            raise ValueError("information_access must map string keys to booleans")
        if any(not isinstance(key, str) or not key for key in self.metadata):
            raise ValueError("run manifest metadata keys must be non-empty strings")
        filenames = (
            self.config_file,
            self.metrics_file,
            self.summary_file,
        )
        for filename in filenames:
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError("run artifact filenames must be non-empty strings")
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("run artifact filenames must stay inside the run directory")
        unique_paths = {Path("manifest.json")}
        unique_paths.update(Path(name) for name in filenames)
        if len(unique_paths) != 4:
            raise ValueError(
                "run artifact filenames must be distinct from each other and manifest.json"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["task_order"] = list(self.task_order)
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> RunManifest:
        try:
            information = payload.get(
                "information_access",
                {},
            )
            metadata = payload.get(
                "metadata",
                {},
            )
            if not isinstance(information, Mapping):
                raise TypeError("information_access must be a mapping")
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be a mapping")
            raw_task_order = payload.get(
                "task_order",
                (),
            )
            if isinstance(raw_task_order, (str, bytes)) or not isinstance(
                raw_task_order,
                Sequence,
            ):
                raise TypeError("task_order must be a sequence of task names")

            task_order = tuple(
                _require_nonempty_string(
                    value,
                    name=f"task_order[{index}]",
                )
                for index, value in enumerate(raw_task_order)
            )
            information_access: dict[str, bool] = {}
            for key, value in information.items():
                normalized_key = _require_nonempty_string(
                    key,
                    name="information_access key",
                )
                if not isinstance(value, bool):
                    raise TypeError("information_access values must be booleans")
                information_access[normalized_key] = value

            manifest = cls(
                run_id=_require_nonempty_string(
                    payload["run_id"],
                    name="run manifest run_id",
                ),
                method=_require_nonempty_string(
                    payload["method"],
                    name="run manifest method",
                ),
                setting=_require_nonempty_string(
                    payload["setting"],
                    name="run manifest setting",
                ),
                benchmark=_require_nonempty_string(
                    payload["benchmark"],
                    name="run manifest benchmark",
                ),
                seed=_require_int(
                    payload["seed"],
                    name="run manifest seed",
                ),
                git_commit=_require_nonempty_string(
                    payload["git_commit"],
                    name="run manifest git_commit",
                ),
                status=_require_nonempty_string(
                    payload.get("status", "completed"),
                    name="run manifest status",
                ),  # type: ignore[arg-type]
                task_order=task_order,
                information_access=information_access,
                config_file=_require_nonempty_string(
                    payload.get("config_file", "config.yaml"),
                    name="run manifest config_file",
                ),
                metrics_file=_require_nonempty_string(
                    payload.get("metrics_file", "metrics.csv"),
                    name="run manifest metrics_file",
                ),
                summary_file=_require_nonempty_string(
                    payload.get("summary_file", "summary.json"),
                    name="run manifest summary_file",
                ),
                schema_version=_require_int(
                    payload.get("schema_version", RUN_SCHEMA_VERSION),
                    name="run manifest schema version",
                ),
                metadata=dict(metadata),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid run manifest payload") from exc
        manifest.validate()
        return manifest


@dataclass(frozen=True)
class RunSummary:
    """Paper-facing scalar, task-level, and resource metrics for one seed."""

    run_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    task_metrics: dict[
        str,
        dict[str, float],
    ] = field(default_factory=dict)
    resources: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: int = RUN_SCHEMA_VERSION

    def validate(self) -> None:
        schema_version = _require_int(
            self.schema_version,
            name="run summary schema version",
        )
        if schema_version != RUN_SCHEMA_VERSION:
            raise ValueError("unsupported run-summary schema version")
        _require_nonempty_string(
            self.run_id,
            name="run summary run_id",
        )
        _finite_mapping(
            self.metrics,
            name="summary metrics",
        )
        _finite_mapping(
            self.resources,
            name="summary resources",
        )
        for task, values in self.task_metrics.items():
            _require_nonempty_string(
                task,
                name="task metric name",
            )
            _finite_mapping(
                values,
                name=f"task metrics for {task}",
            )
        if any(not isinstance(key, str) or not key for key in self.metadata):
            raise ValueError("run summary metadata keys must be non-empty strings")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> RunSummary:
        try:
            raw_metrics = payload.get("metrics", {})
            raw_resources = payload.get("resources", {})
            raw_task_metrics = payload.get("task_metrics", {})
            raw_metadata = payload.get("metadata", {})
            if (
                not isinstance(raw_metrics, Mapping)
                or not isinstance(raw_resources, Mapping)
                or not isinstance(raw_task_metrics, Mapping)
                or not isinstance(raw_metadata, Mapping)
            ):
                raise TypeError("summary mappings have invalid types")

            task_metrics: dict[str, dict[str, float]] = {}
            for task, values in raw_task_metrics.items():
                task_name = _require_nonempty_string(
                    task,
                    name="task metric name",
                )
                if not isinstance(values, Mapping):
                    raise TypeError("per-task metrics must be mappings")
                task_metrics[task_name] = _finite_mapping(
                    values,
                    name=f"task metrics for {task_name}",
                )

            summary = cls(
                run_id=_require_nonempty_string(
                    payload["run_id"],
                    name="run summary run_id",
                ),
                metrics=_finite_mapping(
                    raw_metrics,
                    name="summary metrics",
                ),
                task_metrics=task_metrics,
                resources=_finite_mapping(
                    raw_resources,
                    name="summary resources",
                ),
                metadata=dict(raw_metadata),
                schema_version=_require_int(
                    payload.get("schema_version", RUN_SCHEMA_VERSION),
                    name="run summary schema version",
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid run summary payload") from exc
        summary.validate()
        return summary


@dataclass(frozen=True)
class LoadedRun:
    path: Path
    manifest: RunManifest
    summary: RunSummary
    metrics: pd.DataFrame
    source_hashes: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1 << 20),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def _load_json_mapping(
    path: Path,
) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(
        payload,
        Mapping,
    ):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _require_run_local_path(root: Path, artifact: Path) -> None:
    """Reject artifact paths redirected outside their canonical run directory."""

    if not artifact.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"run artifact resolves outside the run directory: {artifact}")


def load_run_directory(
    path: str | Path,
    *,
    require_completed: bool = True,
) -> LoadedRun:
    """Load one run and fail closed on provenance/schema inconsistencies."""

    root = Path(path)
    manifest_path = root / "manifest.json"
    _require_run_local_path(root, manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = RunManifest.from_dict(_load_json_mapping(manifest_path))
    if require_completed and manifest.status != "completed":
        raise ValueError(
            "paper aggregation refuses incomplete run "
            f"{manifest.run_id!r} with status {manifest.status!r}"
        )

    config_path = root / manifest.config_file
    metrics_path = root / manifest.metrics_file
    summary_path = root / manifest.summary_file
    for artifact in (
        config_path,
        metrics_path,
        summary_path,
    ):
        _require_run_local_path(root, artifact)
        if not artifact.is_file():
            raise FileNotFoundError(f"missing run artifact: {artifact}")

    try:
        config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (
        OSError,
        yaml.YAMLError,
    ) as exc:
        raise ValueError(f"cannot read resolved config: {config_path}") from exc
    if not isinstance(
        config_payload,
        Mapping,
    ):
        raise ValueError("resolved config must contain a mapping")

    summary = RunSummary.from_dict(_load_json_mapping(summary_path))
    if summary.run_id != manifest.run_id:
        raise ValueError("manifest and summary run_id mismatch")

    try:
        metrics = pd.read_csv(metrics_path)
    except Exception as exc:
        raise ValueError(f"cannot read metrics CSV: {metrics_path}") from exc
    if metrics.columns.empty:
        raise ValueError("metrics CSV must contain at least one column")

    hashes = {
        "manifest.json": _sha256(manifest_path),
        manifest.config_file: _sha256(config_path),
        manifest.metrics_file: _sha256(metrics_path),
        manifest.summary_file: _sha256(summary_path),
    }
    return LoadedRun(
        path=root.resolve(),
        manifest=manifest,
        summary=summary,
        metrics=metrics,
        source_hashes=hashes,
    )


def discover_run_directories(
    results_root: str | Path,
) -> tuple[Path, ...]:
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"results root does not exist: {root}")
    return tuple(sorted(path.parent for path in root.rglob("manifest.json")))


def flatten_numeric_metrics(
    payload: Mapping[str, object],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten finite scalar numeric leaves using dotted provenance paths."""

    flattened: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            continue
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            continue
        if isinstance(
            value,
            (int, float),
        ):
            converted = float(value)
            if math.isfinite(converted):
                flattened[name] = converted
            continue
        if isinstance(value, Mapping):
            nested = flatten_numeric_metrics(
                {str(nested_key): nested_value for nested_key, nested_value in value.items()},
                prefix=name,
            )
            flattened.update(nested)
    return flattened


def _numeric_mapping(
    value: object,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if (
            isinstance(key, str)
            and not isinstance(item, bool)
            and isinstance(
                item,
                (int, float),
            )
        ):
            converted = float(item)
            if math.isfinite(converted):
                result[key] = converted
    return result


def _occurrence_aware_task_labels(
    names: Sequence[str],
) -> tuple[str, ...]:
    """Return unique labels while preserving repeated benchmark occurrences."""

    totals: dict[
        str,
        int,
    ] = {}
    for name in names:
        totals[name] = (
            totals.get(
                name,
                0,
            )
            + 1
        )

    seen: dict[
        str,
        int,
    ] = {}
    labels: list[str] = []
    for name in names:
        if totals[name] == 1:
            labels.append(name)
            continue
        occurrence = (
            seen.get(
                name,
                0,
            )
            + 1
        )
        seen[name] = occurrence
        labels.append(f"{name}#occurrence_{occurrence}")
    if len(labels) != len(set(labels)):
        raise ValueError("task occurrence labels are not unique")
    return tuple(labels)


def _task_metrics_from_result(
    result: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    task_names = result.get("task_names")
    if isinstance(
        task_names,
        (str, bytes),
    ) or not isinstance(
        task_names,
        Sequence,
    ):
        return {}
    names = tuple(str(value) for value in task_names)
    if not names:
        return {}
    labels = _occurrence_aware_task_labels(names)

    task_metrics: dict[
        str,
        dict[str, float],
    ] = {label: {} for label in labels}

    return_matrix = result.get("return_matrix")
    if (
        isinstance(
            return_matrix,
            Sequence,
        )
        and not isinstance(
            return_matrix,
            (str, bytes),
        )
        and return_matrix
    ):
        final_row = return_matrix[-1]
        if (
            isinstance(
                final_row,
                Sequence,
            )
            and not isinstance(
                final_row,
                (str, bytes),
            )
            and len(final_row) == len(names)
        ):
            for label, value in zip(
                labels,
                final_row,
                strict=True,
            ):
                if (
                    not isinstance(
                        value,
                        bool,
                    )
                    and isinstance(
                        value,
                        (int, float),
                    )
                    and math.isfinite(float(value))
                ):
                    task_metrics[label]["final_performance"] = float(value)

    success_matrix = result.get("success_matrix")
    if (
        isinstance(
            success_matrix,
            Sequence,
        )
        and not isinstance(
            success_matrix,
            (str, bytes),
        )
        and success_matrix
    ):
        final_row = success_matrix[-1]
        if (
            isinstance(
                final_row,
                Sequence,
            )
            and not isinstance(
                final_row,
                (str, bytes),
            )
            and len(final_row) == len(names)
        ):
            for label, value in zip(
                labels,
                final_row,
                strict=True,
            ):
                if (
                    not isinstance(
                        value,
                        bool,
                    )
                    and isinstance(
                        value,
                        (int, float),
                    )
                    and math.isfinite(float(value))
                ):
                    task_metrics[label]["success_rate"] = float(value)

    return_summary = result.get("return_summary")
    if isinstance(
        return_summary,
        Mapping,
    ):
        forgetting = return_summary.get("forgetting_by_task")
        if (
            isinstance(
                forgetting,
                Sequence,
            )
            and not isinstance(
                forgetting,
                (str, bytes),
            )
            and len(forgetting)
            in {
                len(labels),
                max(
                    0,
                    len(labels) - 1,
                ),
            }
        ):
            forgetting_labels = labels[: len(forgetting)]
            for label, value in zip(
                forgetting_labels,
                forgetting,
                strict=True,
            ):
                if (
                    not isinstance(
                        value,
                        bool,
                    )
                    and isinstance(
                        value,
                        (int, float),
                    )
                    and math.isfinite(float(value))
                ):
                    task_metrics[label]["forgetting"] = float(value)

    return {task: values for task, values in task_metrics.items() if values}


def summarize_runner_result(
    run_id: str,
    result: Mapping[str, object],
    *,
    duration_seconds: float | None = None,
) -> RunSummary:
    """Convert heterogeneous runner JSON into traceable scalar/task summaries."""

    metrics = flatten_numeric_metrics(result)

    # Promote standard nested summaries without discarding their dotted paths.
    return_summary = _numeric_mapping(result.get("return_summary"))
    for key, value in return_summary.items():
        metrics.setdefault(
            key,
            value,
        )
    if "mean_forgetting" in return_summary:
        metrics.setdefault(
            "forgetting",
            return_summary["mean_forgetting"],
        )
    if "backward_transfer" in return_summary:
        metrics.setdefault(
            "bwt",
            return_summary["backward_transfer"],
        )

    training = _numeric_mapping(result.get("training"))
    for key, value in training.items():
        metrics.setdefault(
            key,
            value,
        )

    success_summary = _numeric_mapping(result.get("success_summary"))
    if "final_average" in success_summary:
        metrics.setdefault(
            "success_rate",
            success_summary["final_average"],
        )

    resources: dict[str, float] = {}
    if duration_seconds is not None:
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_seconds must be finite and non-negative")
        resources["duration_seconds"] = duration

    return RunSummary(
        run_id=run_id,
        metrics=metrics,
        task_metrics=(_task_metrics_from_result(result)),
        resources=resources,
        metadata={"numeric_metric_paths": sorted(flatten_numeric_metrics(result))},
    )


def metrics_rows_from_result(
    result: Mapping[str, object],
) -> list[dict[str, object]]:
    """Extract timelines and stage-by-task matrices into long-form metric rows."""

    rows: list[dict[str, object]] = []

    # Generic list-of-mapping timelines such as surprise/adaptation traces.
    for series_name, value in result.items():
        if isinstance(
            value,
            (str, bytes),
        ) or not isinstance(
            value,
            Sequence,
        ):
            continue
        mapping_rows = [
            row
            for row in value
            if isinstance(
                row,
                Mapping,
            )
        ]
        if not mapping_rows or len(mapping_rows) != len(value):
            continue
        for index, row in enumerate(mapping_rows):
            numeric = flatten_numeric_metrics({str(key): item for key, item in row.items()})
            if numeric:
                rows.append(
                    {
                        "series": series_name,
                        "row_index": index,
                        **numeric,
                    }
                )

    # Continual World-style stage x task matrices are preserved explicitly.
    task_names_raw = result.get("task_names")
    task_names: tuple[str, ...] = ()
    if not isinstance(
        task_names_raw,
        (str, bytes),
    ) and isinstance(
        task_names_raw,
        Sequence,
    ):
        task_names = tuple(str(value) for value in task_names_raw)

    for prefix in (
        "return",
        "success",
    ):
        matrix_raw = result.get(f"{prefix}_matrix")
        if matrix_raw is None:
            continue
        if isinstance(
            matrix_raw,
            (str, bytes),
        ) or not isinstance(
            matrix_raw,
            Sequence,
        ):
            raise ValueError(f"{prefix}_matrix must be a sequence of rows")
        labels_raw = result.get(
            f"{prefix}_stage_labels",
            (),
        )
        if isinstance(
            labels_raw,
            (str, bytes),
        ) or not isinstance(
            labels_raw,
            Sequence,
        ):
            labels_raw = ()
        stage_labels = tuple(str(value) for value in labels_raw)
        if stage_labels and len(stage_labels) != len(matrix_raw):
            raise ValueError(f"{prefix}_stage_labels length does not match matrix rows")

        for stage_index, matrix_row in enumerate(matrix_raw):
            if isinstance(
                matrix_row,
                (str, bytes),
            ) or not isinstance(
                matrix_row,
                Sequence,
            ):
                raise ValueError(f"{prefix}_matrix rows must be sequences")
            if task_names and len(matrix_row) != len(task_names):
                raise ValueError(f"{prefix}_matrix width does not match task_names")
            for task_index, value in enumerate(matrix_row):
                if (
                    isinstance(
                        value,
                        bool,
                    )
                    or not isinstance(
                        value,
                        (int, float),
                    )
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"{prefix}_matrix contains a nonfinite numeric value")
                rows.append(
                    {
                        "series": f"{prefix}_matrix",
                        "row_index": (
                            stage_index
                            * max(
                                1,
                                len(matrix_row),
                            )
                            + task_index
                        ),
                        "stage_index": stage_index,
                        "task_index": task_index,
                        "stage_label": (
                            stage_labels[stage_index] if stage_labels else f"stage_{stage_index}"
                        ),
                        "task_name": (
                            task_names[task_index] if task_names else f"task_{task_index}"
                        ),
                        "value": float(value),
                    }
                )

    if rows:
        return rows

    flattened = flatten_numeric_metrics(result)
    if not flattened:
        raise ValueError("runner result contains no finite scalar, timeline, or matrix metrics")
    return [
        {
            "series": "summary",
            "row_index": 0,
            **flattened,
        }
    ]


def write_run_artifacts(
    run_dir: str | Path,
    *,
    manifest: RunManifest,
    summary: RunSummary,
    resolved_config: Mapping[
        str,
        object,
    ],
    metrics_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write the minimum self-contained raw run artifacts atomically enough for local use."""

    manifest.validate()
    summary.validate()
    if manifest.run_id != summary.run_id:
        raise ValueError("manifest and summary run_id must match")
    if not isinstance(
        resolved_config,
        Mapping,
    ):
        raise TypeError("resolved_config must be a mapping")

    metrics = pd.DataFrame(list(metrics_rows))
    if metrics.columns.empty:
        raise ValueError("metrics_rows must produce at least one column")

    root = Path(run_dir)
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = root / "manifest.json"
    config_path = root / manifest.config_file
    summary_path = root / manifest.summary_file
    metrics_path = root / manifest.metrics_file
    for artifact in (
        manifest_path,
        config_path,
        summary_path,
        metrics_path,
    ):
        _require_run_local_path(root, artifact)
    for artifact in (
        config_path,
        summary_path,
        metrics_path,
    ):
        artifact.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest_path.write_text(
        json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            dict(resolved_config),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            summary.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics.to_csv(
        metrics_path,
        index=False,
    )
