"""Strict run-directory schema for reproducible paper aggregation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

RUN_SCHEMA_VERSION = 1
RunStatus = Literal["completed", "failed", "partial"]


def _finite_mapping(
    values: Mapping[str, float],
    *,
    name: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"{name} keys must be non-empty strings"
            )
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(
                f"{name} contains nonfinite value for {key}"
            )
        result[key] = converted
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
    information_access: dict[str, bool] = field(
        default_factory=dict
    )
    config_file: str = "config.yaml"
    metrics_file: str = "metrics.csv"
    summary_file: str = "summary.json"
    schema_version: int = RUN_SCHEMA_VERSION
    metadata: dict[str, object] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if self.schema_version != RUN_SCHEMA_VERSION:
            raise ValueError(
                "unsupported run-manifest schema version"
            )
        for name, value in (
            ("run_id", self.run_id),
            ("method", self.method),
            ("setting", self.setting),
            ("benchmark", self.benchmark),
            ("git_commit", self.git_commit),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"run manifest {name} must be non-empty"
                )
        if self.seed < 0:
            raise ValueError(
                "run manifest seed must be non-negative"
            )
        if self.status not in {
            "completed",
            "failed",
            "partial",
        }:
            raise ValueError(
                f"unsupported run status: {self.status}"
            )
        if any(
            not isinstance(task, str)
            or not task.strip()
            for task in self.task_order
        ):
            raise ValueError(
                "task_order entries must be non-empty strings"
            )
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, bool)
            for key, value in self.information_access.items()
        ):
            raise ValueError(
                "information_access must map string keys to booleans"
            )
        for filename in (
            self.config_file,
            self.metrics_file,
            self.summary_file,
        ):
            relative = Path(filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or filename.strip() == ""
            ):
                raise ValueError(
                    "run artifact filenames must stay inside the run directory"
                )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["task_order"] = list(
            self.task_order
        )
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "RunManifest":
        try:
            information = payload.get(
                "information_access",
                {},
            )
            metadata = payload.get(
                "metadata",
                {},
            )
            if not isinstance(
                information,
                Mapping,
            ):
                raise TypeError(
                    "information_access must be a mapping"
                )
            if not isinstance(
                metadata,
                Mapping,
            ):
                raise TypeError(
                    "metadata must be a mapping"
                )
            manifest = cls(
                run_id=str(payload["run_id"]),
                method=str(payload["method"]),
                setting=str(payload["setting"]),
                benchmark=str(payload["benchmark"]),
                seed=int(payload["seed"]),
                git_commit=str(
                    payload["git_commit"]
                ),
                status=str(
                    payload.get(
                        "status",
                        "completed",
                    )
                ),  # type: ignore[arg-type]
                task_order=tuple(
                    str(value)
                    for value in payload.get(
                        "task_order",
                        (),
                    )  # type: ignore[arg-type]
                ),
                information_access={
                    str(key): value
                    for key, value in information.items()
                    if isinstance(value, bool)
                },
                config_file=str(
                    payload.get(
                        "config_file",
                        "config.yaml",
                    )
                ),
                metrics_file=str(
                    payload.get(
                        "metrics_file",
                        "metrics.csv",
                    )
                ),
                summary_file=str(
                    payload.get(
                        "summary_file",
                        "summary.json",
                    )
                ),
                schema_version=int(
                    payload.get(
                        "schema_version",
                        RUN_SCHEMA_VERSION,
                    )
                ),
                metadata=dict(metadata),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "invalid run manifest payload"
            ) from exc
        manifest.validate()
        if len(
            manifest.information_access
        ) != len(information):
            raise ValueError(
                "information_access values must all be booleans"
            )
        return manifest


@dataclass(frozen=True)
class RunSummary:
    """Paper-facing scalar, task-level, and resource metrics for one seed."""

    run_id: str
    metrics: dict[str, float] = field(
        default_factory=dict
    )
    task_metrics: dict[
        str,
        dict[str, float],
    ] = field(
        default_factory=dict
    )
    resources: dict[str, float] = field(
        default_factory=dict
    )
    metadata: dict[str, object] = field(
        default_factory=dict
    )
    schema_version: int = RUN_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != RUN_SCHEMA_VERSION:
            raise ValueError(
                "unsupported run-summary schema version"
            )
        if not self.run_id.strip():
            raise ValueError(
                "run summary run_id must be non-empty"
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
            if not isinstance(task, str) or not task:
                raise ValueError(
                    "task metric names must be non-empty"
                )
            _finite_mapping(
                values,
                name=f"task metrics for {task}",
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "RunSummary":
        try:
            raw_metrics = payload.get(
                "metrics",
                {},
            )
            raw_resources = payload.get(
                "resources",
                {},
            )
            raw_task_metrics = payload.get(
                "task_metrics",
                {},
            )
            raw_metadata = payload.get(
                "metadata",
                {},
            )
            if not isinstance(
                raw_metrics,
                Mapping,
            ) or not isinstance(
                raw_resources,
                Mapping,
            ) or not isinstance(
                raw_task_metrics,
                Mapping,
            ) or not isinstance(
                raw_metadata,
                Mapping,
            ):
                raise TypeError(
                    "summary mappings have invalid types"
                )

            task_metrics: dict[
                str,
                dict[str, float],
            ] = {}
            for task, values in raw_task_metrics.items():
                if not isinstance(
                    values,
                    Mapping,
                ):
                    raise TypeError(
                        "per-task metrics must be mappings"
                    )
                task_metrics[
                    str(task)
                ] = _finite_mapping(
                    {
                        str(key): float(value)
                        for key, value in values.items()
                    },
                    name=f"task metrics for {task}",
                )

            summary = cls(
                run_id=str(
                    payload["run_id"]
                ),
                metrics=_finite_mapping(
                    {
                        str(key): float(value)
                        for key, value in raw_metrics.items()
                    },
                    name="summary metrics",
                ),
                task_metrics=task_metrics,
                resources=_finite_mapping(
                    {
                        str(key): float(value)
                        for key, value in raw_resources.items()
                    },
                    name="summary resources",
                ),
                metadata=dict(
                    raw_metadata
                ),
                schema_version=int(
                    payload.get(
                        "schema_version",
                        RUN_SCHEMA_VERSION,
                    )
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "invalid run summary payload"
            ) from exc
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
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            f"cannot read JSON artifact: {path}"
        ) from exc
    if not isinstance(
        payload,
        Mapping,
    ):
        raise ValueError(
            f"JSON artifact must contain an object: {path}"
        )
    return payload


def load_run_directory(
    path: str | Path,
    *,
    require_completed: bool = True,
) -> LoadedRun:
    """Load one run and fail closed on provenance/schema inconsistencies."""

    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing run manifest: {manifest_path}"
        )
    manifest = RunManifest.from_dict(
        _load_json_mapping(
            manifest_path
        )
    )
    if (
        require_completed
        and manifest.status != "completed"
    ):
        raise ValueError(
            "paper aggregation refuses incomplete run "
            f"{manifest.run_id!r} with status {manifest.status!r}"
        )

    config_path = (
        root / manifest.config_file
    )
    metrics_path = (
        root / manifest.metrics_file
    )
    summary_path = (
        root / manifest.summary_file
    )
    for artifact in (
        config_path,
        metrics_path,
        summary_path,
    ):
        if not artifact.is_file():
            raise FileNotFoundError(
                f"missing run artifact: {artifact}"
            )

    try:
        config_payload = yaml.safe_load(
            config_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        yaml.YAMLError,
    ) as exc:
        raise ValueError(
            f"cannot read resolved config: {config_path}"
        ) from exc
    if not isinstance(
        config_payload,
        Mapping,
    ):
        raise ValueError(
            "resolved config must contain a mapping"
        )

    summary = RunSummary.from_dict(
        _load_json_mapping(
            summary_path
        )
    )
    if (
        summary.run_id
        != manifest.run_id
    ):
        raise ValueError(
            "manifest and summary run_id mismatch"
        )

    try:
        metrics = pd.read_csv(
            metrics_path
        )
    except Exception as exc:
        raise ValueError(
            f"cannot read metrics CSV: {metrics_path}"
        ) from exc
    if metrics.columns.empty:
        raise ValueError(
            "metrics CSV must contain at least one column"
        )

    hashes = {
        "manifest.json": _sha256(
            manifest_path
        ),
        manifest.config_file: _sha256(
            config_path
        ),
        manifest.metrics_file: _sha256(
            metrics_path
        ),
        manifest.summary_file: _sha256(
            summary_path
        ),
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
        raise FileNotFoundError(
            f"results root does not exist: {root}"
        )
    return tuple(
        sorted(
            path.parent
            for path in root.rglob(
                "manifest.json"
            )
        )
    )


def write_run_artifacts(
    run_dir: str | Path,
    *,
    manifest: RunManifest,
    summary: RunSummary,
    resolved_config: Mapping[
        str,
        object,
    ],
    metrics_rows: Sequence[
        Mapping[str, object]
    ],
) -> None:
    """Write the minimum self-contained raw run artifacts atomically enough for local use."""

    manifest.validate()
    summary.validate()
    if (
        manifest.run_id
        != summary.run_id
    ):
        raise ValueError(
            "manifest and summary run_id must match"
        )
    if not isinstance(
        resolved_config,
        Mapping,
    ):
        raise TypeError(
            "resolved_config must be a mapping"
        )

    root = Path(run_dir)
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        root / "manifest.json"
    )
    config_path = (
        root / manifest.config_file
    )
    summary_path = (
        root / manifest.summary_file
    )
    metrics_path = (
        root / manifest.metrics_file
    )
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
    metrics = pd.DataFrame(
        list(metrics_rows)
    )
    if metrics.columns.empty:
        raise ValueError(
            "metrics_rows must produce at least one column"
        )
    metrics.to_csv(
        metrics_path,
        index=False,
    )
