"""Compatibility façade for the canonical Phase-15 paper artifact pipeline.

New code should import :mod:`rl_bgd.analysis.paper_artifacts` and
:mod:`rl_bgd.analysis.statistics` directly. This module preserves the first
Phase-15 public API without maintaining a second aggregation implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_bgd.analysis.paper_artifacts import (
    PaperArtifactConfig,
    load_paper_runs,
)
from rl_bgd.analysis.paper_artifacts import (
    build_paper_artifacts as _build_paper_artifacts,
)
from rl_bgd.analysis.statistics import (
    bootstrap_mean_ci as _bootstrap_mean_ci,
)


@dataclass(frozen=True)
class BootstrapConfig:
    """Backward-compatible names for paper bootstrap controls."""

    samples: int = 5_000
    confidence: float = 0.95
    seed: int = 2026

    def validate(self) -> None:
        if self.samples < 1:
            raise ValueError(
                "bootstrap samples must be positive"
            )
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(
                "bootstrap confidence must lie in (0, 1)"
            )
        if self.seed < 0:
            raise ValueError(
                "bootstrap seed must be non-negative"
            )


def bootstrap_mean_ci(
    values: list[float],
    *,
    config: BootstrapConfig | None = None,
) -> dict[str, float]:
    resolved = (
        config
        or BootstrapConfig()
    )
    resolved.validate()
    estimate = _bootstrap_mean_ci(
        values,
        confidence=resolved.confidence,
        resamples=resolved.samples,
        seed=resolved.seed,
    )
    return {
        "n": float(
            estimate.n
        ),
        "mean": estimate.mean,
        "std": estimate.std,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "confidence": (
            estimate.confidence
        ),
        "bootstrap_samples": float(
            estimate.resamples
        ),
    }


def discover_run_records(
    run_root: str | Path,
) -> list[dict[str, Any]]:
    """Expose completed canonical runs as flat records without skipping failures."""

    runs = load_paper_runs(
        run_root
    )
    records: list[
        dict[str, Any]
    ] = []
    for run in runs:
        record: dict[
            str,
            Any,
        ] = {
            "run_id": run.manifest.run_id,
            "seed": run.manifest.seed,
            "git_commit": (
                run.manifest.git_commit
            ),
            "algorithm": (
                run.manifest.method
            ),
            "environment": (
                run.manifest.benchmark
            ),
            "protocol": (
                run.manifest.setting
            ),
            "suite": (
                run.manifest.metadata.get(
                    "suite"
                )
            ),
            "job_id": (
                run.manifest.metadata.get(
                    "job_id"
                )
            ),
            "hypothesis_id": (
                run.manifest.metadata.get(
                    "hypothesis_id"
                )
            ),
            "primary_metric": (
                run.manifest.metadata.get(
                    "primary_metric"
                )
            ),
        }
        for key, value in (
            run.summary.metrics.items()
        ):
            record[
                f"metric.{key}"
            ] = value
        for key, value in (
            run.summary.resources.items()
        ):
            record[
                f"resource.{key}"
            ] = value
        records.append(
            record
        )
    return records


def build_paper_artifacts(
    run_root: str | Path,
    output_dir: str | Path,
    *,
    bootstrap: BootstrapConfig | None = None,
) -> dict[str, Any]:
    """Delegate to the strict builder while returning legacy report keys too."""

    resolved = (
        bootstrap
        or BootstrapConfig()
    )
    resolved.validate()
    manifest = _build_paper_artifacts(
        run_root,
        output_dir,
        config=PaperArtifactConfig(
            confidence=resolved.confidence,
            bootstrap_resamples=(
                resolved.samples
            ),
            seed=resolved.seed,
        ),
    )
    output = Path(
        output_dir
    )
    aggregate_path = (
        output
        / "tables"
        / "aggregate_statistics.csv"
    )
    bootstrap_groups = 0
    if aggregate_path.is_file():
        import pandas as pd

        aggregate = pd.read_csv(
            aggregate_path
        )
        bootstrap_groups = int(
            len(
                aggregate[
                    [
                        "experiment",
                        "method",
                        "setting",
                        "benchmark",
                    ]
                ].drop_duplicates()
            )
        )
    suite_manifests = tuple(
        Path(
            run_root
        ).rglob(
            "suite_manifest.json"
        )
    )
    artifacts = [
        *manifest[
            "generated_tables"
        ],
        *manifest[
            "generated_figures"
        ],
        "paper_manifest.json",
    ]
    return {
        "schema_version": 3,
        "run_schema": (
            "manifest.json + config.yaml + metrics.csv + summary.json"
        ),
        "run_root": str(
            Path(
                run_root
            )
        ),
        "runs_aggregated": (
            manifest[
                "run_count"
            ]
        ),
        "bootstrap_groups": (
            bootstrap_groups
        ),
        "manifests_indexed": len(
            suite_manifests
        ),
        "bootstrap": {
            "samples": (
                resolved.samples
            ),
            "confidence": (
                resolved.confidence
            ),
            "seed": resolved.seed,
        },
        "artifacts": artifacts,
        "canonical_manifest": str(
            output
            / "paper_manifest.json"
        ),
    }
