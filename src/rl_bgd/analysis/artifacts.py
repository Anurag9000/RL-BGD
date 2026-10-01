"""Automatic paper artifact generation from raw suite run directories."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapConfig:
    samples: int = 5_000
    confidence: float = 0.95
    seed: int = 2026

    def validate(self) -> None:
        if self.samples < 100:
            raise ValueError("bootstrap samples must be >= 100")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("bootstrap confidence must lie in (0, 1)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, prefix=name))
        return output
    if isinstance(value, list):
        if all(
            isinstance(item, (int, float, bool))
            and not isinstance(item, bool)
            for item in value
        ):
            output[prefix] = json.dumps(value)
        return output
    if isinstance(value, (str, int, float, bool)) or value is None:
        output[prefix] = value
    return output


def _metric_value(
    flat_result: dict[str, Any],
    metric_path: str,
) -> float | None:
    value = flat_result.get(metric_path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    matches = [
        candidate
        for key, candidate in flat_result.items()
        if key.endswith(f".{metric_path}") or key == metric_path
    ]
    numeric = [
        float(candidate)
        for candidate in matches
        if isinstance(candidate, (int, float))
        and not isinstance(candidate, bool)
    ]
    if len(numeric) == 1:
        return numeric[0]
    return None


def discover_run_records(
    run_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(run_root)
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(root.rglob("run_metadata.json")):
        run_dir = metadata_path.parent
        stdout_path = run_dir / "stdout.json"
        if not stdout_path.exists():
            continue
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if metadata.get("status") != "success":
            continue
        result = json.loads(
            stdout_path.read_text(encoding="utf-8")
        )
        flat = _flatten(result)
        run_id = str(metadata["run_id"])
        job_id = str(
            metadata.get(
                "job_id",
                run_id.split("__seed_", 1)[0],
            )
        )
        primary_metric = str(metadata["primary_metric"])
        primary_value = _metric_value(
            flat,
            primary_metric,
        )
        record: dict[str, Any] = {
            "suite": metadata["suite"],
            "job_id": job_id,
            "run_id": run_id,
            "seed": int(metadata["seed"]),
            "git_commit": metadata.get("git_commit"),
            "algorithm": metadata["algorithm"],
            "environment": metadata["environment"],
            "protocol": metadata["protocol"],
            "hypothesis_id": metadata["hypothesis_id"],
            "config_path": metadata.get("config_path"),
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "duration_seconds": float(
                metadata["duration_seconds"]
            ),
            "stdout_sha256": _sha256(stdout_path),
            "metadata_sha256": _sha256(metadata_path),
        }
        for key, value in flat.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                record[f"result.{key}"] = float(value)
        records.append(record)
    return records


def bootstrap_mean_ci(
    values: list[float],
    *,
    config: BootstrapConfig | None = None,
) -> dict[str, float]:
    resolved = config or BootstrapConfig()
    resolved.validate()
    if not values:
        raise ValueError("bootstrap requires at least one value")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite")
    generator = np.random.default_rng(resolved.seed)
    indices = generator.integers(
        0,
        array.size,
        size=(resolved.samples, array.size),
    )
    means = array[indices].mean(axis=1)
    alpha = 1.0 - resolved.confidence
    lower = float(
        np.quantile(means, alpha / 2.0)
    )
    upper = float(
        np.quantile(means, 1.0 - alpha / 2.0)
    )
    return {
        "n": float(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "ci_low": lower,
        "ci_high": upper,
        "confidence": resolved.confidence,
        "bootstrap_samples": float(resolved.samples),
    }


def _bootstrap_table(
    records: list[dict[str, Any]],
    *,
    config: BootstrapConfig,
) -> pd.DataFrame:
    groups: dict[
        tuple[str, str, str],
        list[float],
    ] = {}
    for record in records:
        value = record.get("primary_value")
        if value is None:
            continue
        key = (
            str(record["suite"]),
            str(record["job_id"]),
            str(record["primary_metric"]),
        )
        groups.setdefault(key, []).append(float(value))

    rows: list[dict[str, Any]] = []
    for (suite, job_id, metric), values in sorted(groups.items()):
        stats = bootstrap_mean_ci(
            values,
            config=config,
        )
        rows.append(
            {
                "suite": suite,
                "job_id": job_id,
                "primary_metric": metric,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def _write_markdown_table(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    if frame.empty:
        path.write_text(
            "| suite | job_id | primary_metric | n | mean | ci_low | ci_high |\n"
            "|---|---|---|---:|---:|---:|---:|\n",
            encoding="utf-8",
        )
        return
    columns = [
        "suite",
        "job_id",
        "primary_metric",
        "n",
        "mean",
        "ci_low",
        "ci_high",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(
            [
                "---",
                "---",
                "---",
                "---:",
                "---:",
                "---:",
                "---:",
            ]
        ) + "|",
    ]
    for _, row in frame[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                str(row[column])
                for column in columns
            )
            + " |"
        )
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _plot_primary_metrics(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    figure = plt.figure(
        figsize=(
            max(6.0, 0.7 * max(1, len(frame))),
            4.5,
        )
    )
    axis = figure.add_subplot(111)
    if not frame.empty:
        labels = [
            f"{row.suite}/{row.job_id}"
            for row in frame.itertuples()
        ]
        means = frame["mean"].to_numpy(dtype=float)
        lower = means - frame["ci_low"].to_numpy(dtype=float)
        upper = frame["ci_high"].to_numpy(dtype=float) - means
        positions = np.arange(len(labels))
        axis.bar(
            positions,
            means,
            yerr=np.vstack([lower, upper]),
            capsize=3,
        )
        axis.set_xticks(
            positions,
            labels,
            rotation=45,
            ha="right",
        )
    axis.set_ylabel("primary metric mean")
    axis.set_title("Paper suite primary metrics with bootstrap intervals")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _manifest_index(
    run_root: Path,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(run_root.rglob("suite_manifest.json")):
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
        manifests.append(
            {
                "suite": data["suite"],
                "suite_revision": data["suite_revision"],
                "git_commit": data.get("git_commit"),
                "path": str(path),
                "sha256": _sha256(path),
                "jobs": len(data.get("jobs", [])),
            }
        )
    return manifests


def build_paper_artifacts(
    run_root: str | Path,
    output_dir: str | Path,
    *,
    bootstrap: BootstrapConfig | None = None,
) -> dict[str, Any]:
    """Aggregate raw runs into tables/figures without manual transcription."""

    resolved = bootstrap or BootstrapConfig()
    resolved.validate()
    root = Path(run_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    records = discover_run_records(root)
    runs_frame = pd.DataFrame(records)
    runs_csv = output / "all_runs.csv"
    runs_frame.to_csv(
        runs_csv,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    primary_frame = _bootstrap_table(
        records,
        config=resolved,
    )
    primary_csv = output / "primary_metrics_bootstrap.csv"
    primary_frame.to_csv(
        primary_csv,
        index=False,
    )
    primary_md = output / "primary_metrics_bootstrap.md"
    _write_markdown_table(
        primary_frame,
        primary_md,
    )
    primary_figure = output / "primary_metrics_bootstrap.png"
    _plot_primary_metrics(
        primary_frame,
        primary_figure,
    )

    manifest_data = _manifest_index(root)
    manifest_index = output / "manifest_index.json"
    manifest_index.write_text(
        json.dumps(
            manifest_data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = {
        "schema_version": 1,
        "run_root": str(root),
        "runs_aggregated": len(records),
        "bootstrap_groups": int(len(primary_frame)),
        "manifests_indexed": len(manifest_data),
        "bootstrap": {
            "samples": resolved.samples,
            "confidence": resolved.confidence,
            "seed": resolved.seed,
        },
        "artifacts": [
            runs_csv.name,
            primary_csv.name,
            primary_md.name,
            primary_figure.name,
            manifest_index.name,
            "artifact_report.json",
        ],
    }
    report_path = output / "artifact_report.json"
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report
