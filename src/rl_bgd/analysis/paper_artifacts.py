"""Automatic paper tables, figures, bootstrap statistics, and provenance manifests."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_bgd.analysis.statistics import (
    bootstrap_mean_ci,
    hierarchical_bootstrap_mean,
    paired_bootstrap_difference,
)
from rl_bgd.artifacts import (
    LoadedRun,
    discover_run_directories,
    load_run_directory,
)


@dataclass(frozen=True)
class PaperArtifactConfig:
    """Statistical and export controls for one deterministic paper build."""

    confidence: float = 0.95
    bootstrap_resamples: int = 10_000
    seed: int = 0
    figure_formats: tuple[str, ...] = (
        "png",
        "pdf",
        "svg",
    )

    def validate(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(
                "paper confidence must lie strictly between 0 and 1"
            )
        if self.bootstrap_resamples < 1:
            raise ValueError(
                "paper bootstrap resamples must be positive"
            )
        if self.seed < 0:
            raise ValueError(
                "paper artifact seed must be non-negative"
            )
        allowed = {
            "png",
            "pdf",
            "svg",
        }
        if (
            not self.figure_formats
            or any(
                value not in allowed
                for value in self.figure_formats
            )
        ):
            raise ValueError(
                "figure_formats must be a non-empty subset of png/pdf/svg"
            )


@dataclass(frozen=True)
class RunGroup:
    experiment: str
    method: str
    setting: str
    benchmark: str

    @property
    def label(self) -> str:
        return (
            f"{self.experiment} | "
            f"{self.method} | "
            f"{self.setting} | "
            f"{self.benchmark}"
        )


def _experiment_id(
    run: LoadedRun,
) -> str:
    value = run.manifest.metadata.get(
        "job_id"
    )
    if isinstance(
        value,
        str,
    ) and value.strip():
        return value
    return run.manifest.method


def _group(
    run: LoadedRun,
) -> RunGroup:
    return RunGroup(
        experiment=_experiment_id(
            run
        ),
        method=run.manifest.method,
        setting=run.manifest.setting,
        benchmark=run.manifest.benchmark,
    )


def _group_runs(
    runs: Sequence[LoadedRun],
) -> dict[
    RunGroup,
    list[LoadedRun],
]:
    grouped: dict[
        RunGroup,
        list[LoadedRun],
    ] = defaultdict(list)
    for run in runs:
        grouped[
            _group(run)
        ].append(run)

    for group, values in grouped.items():
        seeds = [
            run.manifest.seed
            for run in values
        ]
        if len(seeds) != len(
            set(seeds)
        ):
            raise ValueError(
                "duplicate seed detected within paper group "
                f"{group.label}"
            )
        values.sort(
            key=lambda run: (
                run.manifest.seed,
                run.manifest.run_id,
            )
        )
    return dict(grouped)


def load_paper_runs(
    results_root: str | Path,
) -> tuple[LoadedRun, ...]:
    """Load every discovered run; failed/partial runs abort the build."""

    directories = (
        discover_run_directories(
            results_root
        )
    )
    if not directories:
        raise ValueError(
            "paper build found no run manifests"
        )
    runs = tuple(
        load_run_directory(
            path,
            require_completed=True,
        )
        for path in directories
    )
    for run in runs:
        primary_metric = (
            run.manifest.metadata.get(
                "primary_metric"
            )
        )
        if (
            primary_metric is not None
            and (
                not isinstance(
                    primary_metric,
                    str,
                )
                or not primary_metric
            )
        ):
            raise ValueError(
                "primary_metric metadata must be a non-empty string"
            )
        if (
            isinstance(
                primary_metric,
                str,
            )
            and primary_metric
            not in run.summary.metrics
            and primary_metric
            not in run.summary.resources
        ):
            raise ValueError(
                "completed run "
                f"{run.manifest.run_id!r} does not expose declared "
                f"primary metric {primary_metric!r}"
            )

    run_ids = [
        run.manifest.run_id
        for run in runs
    ]
    if len(run_ids) != len(
        set(run_ids)
    ):
        raise ValueError(
            "run_id values must be globally unique within a paper build"
        )
    return runs


def _metric_map(
    run: LoadedRun,
) -> dict[str, float]:
    values = dict(
        run.summary.metrics
    )
    for key, value in (
        run.summary.resources.items()
    ):
        metric = (
            key
            if key not in values
            else f"resource.{key}"
        )
        values[
            metric
        ] = value
    return values


def aggregate_scalar_metrics(
    runs: Sequence[LoadedRun],
    *,
    config: PaperArtifactConfig,
) -> pd.DataFrame:
    """Aggregate every scalar metric with seed-level bootstrap intervals."""

    rows: list[
        dict[str, object]
    ] = []
    grouped = _group_runs(
        runs
    )
    for group_index, (
        group,
        group_runs,
    ) in enumerate(
        sorted(
            grouped.items(),
            key=lambda item: (
                item[0].experiment,
                item[0].method,
                item[0].setting,
                item[0].benchmark,
            ),
        )
    ):
        metric_maps = [
            _metric_map(run)
            for run in group_runs
        ]
        metric_names = sorted(
            set().union(
                *(
                    set(values)
                    for values in metric_maps
                )
            )
        )
        for metric_index, metric in enumerate(
            metric_names
        ):
            presence = [
                metric in values
                for values in metric_maps
            ]
            if not all(
                presence
            ):
                missing = [
                    run.manifest.run_id
                    for run, present in zip(
                        group_runs,
                        presence,
                        strict=True,
                    )
                    if not present
                ]
                raise ValueError(
                    "metric is missing for a subset of matched seeds: "
                    f"{group.label} / {metric} / {missing}"
                )
            values = [
                metric_map[
                    metric
                ]
                for metric_map in metric_maps
            ]
            estimate = (
                bootstrap_mean_ci(
                    values,
                    confidence=(
                        config.confidence
                    ),
                    resamples=(
                        config.bootstrap_resamples
                    ),
                    seed=(
                        config.seed
                        + group_index
                        * 100_000
                        + metric_index
                    ),
                )
            )
            rows.append(
                {
                    "experiment": group.experiment,
                    "method": group.method,
                    "setting": group.setting,
                    "benchmark": group.benchmark,
                    "metric": metric,
                    **estimate.to_dict(),
                    "run_ids": json.dumps(
                        [
                            run.manifest.run_id
                            for run in group_runs
                        ]
                    ),
                    "seeds": json.dumps(
                        [
                            run.manifest.seed
                            for run in group_runs
                        ]
                    ),
                    "raw_values": json.dumps(
                        values
                    ),
                }
            )
    return pd.DataFrame(
        rows
    )


def aggregate_task_metrics(
    runs: Sequence[LoadedRun],
    *,
    config: PaperArtifactConfig,
) -> pd.DataFrame:
    """Hierarchically bootstrap task-level values nested inside seeds."""

    rows: list[
        dict[str, object]
    ] = []
    grouped = _group_runs(
        runs
    )
    for group_index, (
        group,
        group_runs,
    ) in enumerate(
        sorted(
            grouped.items(),
            key=lambda item: item[
                0
            ].label,
        )
    ):
        metric_names = sorted(
            {
                metric
                for run in group_runs
                for values in (
                    run.summary.task_metrics.values()
                )
                for metric in values
            }
        )
        for metric_index, metric in enumerate(
            metric_names
        ):
            seed_values: dict[
                int,
                list[float],
            ] = {}
            run_task_counts: dict[
                str,
                int,
            ] = {}
            for run in group_runs:
                values = [
                    task_values[
                        metric
                    ]
                    for task_values in (
                        run.summary.task_metrics.values()
                    )
                    if metric
                    in task_values
                ]
                if not values:
                    raise ValueError(
                        "task metric is missing for an entire matched seed: "
                        f"{group.label} / {metric} / {run.manifest.run_id}"
                    )
                seed_values[
                    run.manifest.seed
                ] = values
                run_task_counts[
                    run.manifest.run_id
                ] = len(
                    values
                )
            estimate = (
                hierarchical_bootstrap_mean(
                    seed_values,
                    confidence=(
                        config.confidence
                    ),
                    resamples=(
                        config.bootstrap_resamples
                    ),
                    seed=(
                        config.seed
                        + 1_000_000
                        + group_index
                        * 100_000
                        + metric_index
                    ),
                )
            )
            rows.append(
                {
                    "experiment": group.experiment,
                    "method": group.method,
                    "setting": group.setting,
                    "benchmark": group.benchmark,
                    "metric": metric,
                    **estimate.to_dict(),
                    "run_ids": json.dumps(
                        [
                            run.manifest.run_id
                            for run in group_runs
                        ]
                    ),
                    "task_counts_by_run": json.dumps(
                        run_task_counts,
                        sort_keys=True,
                    ),
                }
            )
    return pd.DataFrame(
        rows
    )


def paired_method_differences(
    runs: Sequence[LoadedRun],
    *,
    config: PaperArtifactConfig,
) -> pd.DataFrame:
    """Create all matched-seed method differences within setting/benchmark."""

    grouped = _group_runs(
        runs
    )
    by_context: dict[
        tuple[str, str],
        list[
            tuple[
                RunGroup,
                list[LoadedRun],
            ]
        ],
    ] = defaultdict(list)
    for group, group_runs in grouped.items():
        by_context[
            (
                group.setting,
                group.benchmark,
            )
        ].append(
            (
                group,
                group_runs,
            )
        )

    rows: list[
        dict[str, object]
    ] = []
    comparison_index = 0
    for (
        setting,
        benchmark,
    ), groups in sorted(
        by_context.items()
    ):
        ordered = sorted(
            groups,
            key=lambda item: (
                item[0].method,
                item[0].experiment,
            ),
        )
        for left_index in range(
            len(ordered)
        ):
            for right_index in range(
                left_index + 1,
                len(ordered),
            ):
                (
                    left_group,
                    left_runs,
                ) = ordered[
                    left_index
                ]
                (
                    right_group,
                    right_runs,
                ) = ordered[
                    right_index
                ]
                left_by_seed = {
                    run.manifest.seed: run
                    for run in left_runs
                }
                right_by_seed = {
                    run.manifest.seed: run
                    for run in right_runs
                }
                matched_seeds = sorted(
                    set(left_by_seed)
                    & set(right_by_seed)
                )
                if not matched_seeds:
                    continue
                left_metrics = {
                    metric
                    for run in left_runs
                    for metric in _metric_map(
                        run
                    )
                }
                right_metrics = {
                    metric
                    for run in right_runs
                    for metric in _metric_map(
                        run
                    )
                }
                common_metrics = sorted(
                    left_metrics
                    & right_metrics
                )
                for metric_index, metric in enumerate(
                    common_metrics
                ):
                    left_values: list[
                        float
                    ] = []
                    right_values: list[
                        float
                    ] = []
                    run_pairs: list[
                        tuple[str, str]
                    ] = []
                    complete = True
                    for seed in matched_seeds:
                        left_map = _metric_map(
                            left_by_seed[
                                seed
                            ]
                        )
                        right_map = _metric_map(
                            right_by_seed[
                                seed
                            ]
                        )
                        if (
                            metric
                            not in left_map
                            or metric
                            not in right_map
                        ):
                            complete = False
                            break
                        left_values.append(
                            left_map[
                                metric
                            ]
                        )
                        right_values.append(
                            right_map[
                                metric
                            ]
                        )
                        run_pairs.append(
                            (
                                left_by_seed[
                                    seed
                                ].manifest.run_id,
                                right_by_seed[
                                    seed
                                ].manifest.run_id,
                            )
                        )
                    if not complete:
                        continue
                    estimate = (
                        paired_bootstrap_difference(
                            left_values,
                            right_values,
                            confidence=(
                                config.confidence
                            ),
                            resamples=(
                                config.bootstrap_resamples
                            ),
                            seed=(
                                config.seed
                                + 2_000_000
                                + comparison_index
                                * 100_000
                                + metric_index
                            ),
                        )
                    )
                    rows.append(
                        {
                            "setting": setting,
                            "benchmark": benchmark,
                            "left_experiment": (
                                left_group.experiment
                            ),
                            "left_method": (
                                left_group.method
                            ),
                            "right_experiment": (
                                right_group.experiment
                            ),
                            "right_method": (
                                right_group.method
                            ),
                            "metric": metric,
                            "difference": (
                                "left_minus_right"
                            ),
                            **estimate.to_dict(),
                            "matched_seeds": json.dumps(
                                matched_seeds
                            ),
                            "run_pairs": json.dumps(
                                run_pairs
                            ),
                        }
                    )
                comparison_index += 1
    return pd.DataFrame(
        rows
    )


def run_index(
    runs: Sequence[LoadedRun],
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []
    for run in sorted(
        runs,
        key=lambda item: (
            _experiment_id(item),
            item.manifest.seed,
            item.manifest.run_id,
        ),
    ):
        rows.append(
            {
                "run_id": run.manifest.run_id,
                "experiment": _experiment_id(
                    run
                ),
                "method": run.manifest.method,
                "setting": run.manifest.setting,
                "benchmark": run.manifest.benchmark,
                "seed": run.manifest.seed,
                "git_commit": run.manifest.git_commit,
                "task_order": json.dumps(
                    list(
                        run.manifest.task_order
                    )
                ),
                "run_path": str(
                    run.path
                ),
            }
        )
    return pd.DataFrame(
        rows
    )


def information_access_table(
    runs: Sequence[LoadedRun],
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []
    for group, group_runs in sorted(
        _group_runs(
            runs
        ).items(),
        key=lambda item: item[
            0
        ].label,
    ):
        access_values = [
            run.manifest.information_access
            for run in group_runs
        ]
        first = access_values[
            0
        ]
        if any(
            value != first
            for value in access_values[
                1:
            ]
        ):
            raise ValueError(
                "information-access assumptions differ within a matched group: "
                f"{group.label}"
            )
        row: dict[
            str,
            object,
        ] = {
            "experiment": group.experiment,
            "method": group.method,
            "setting": group.setting,
            "benchmark": group.benchmark,
            "run_ids": json.dumps(
                [
                    run.manifest.run_id
                    for run in group_runs
                ]
            ),
        }
        row.update(
            {
                key: bool(value)
                for key, value in sorted(
                    first.items()
                )
            }
        )
        rows.append(
            row
        )
    return pd.DataFrame(
        rows
    )


_FINAL_METRIC_CANDIDATES: tuple[
    str,
    ...,
] = (
    "final_average",
    "post_return",
    "final_10_mean_return",
    "mean_episode_return",
    "final_phase_return",
)


def results_summary_table(
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    """Build a human-facing wide table while retaining source metric names."""

    if aggregate.empty:
        return pd.DataFrame()
    group_columns = [
        "experiment",
        "method",
        "setting",
        "benchmark",
    ]
    rows: list[
        dict[str, object]
    ] = []
    for keys, values in aggregate.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        row = {
            column: value
            for column, value in zip(
                group_columns,
                keys,
                strict=True,
            )
        }
        by_metric = {
            str(record["metric"]): record
            for record in values.to_dict(
                "records"
            )
        }

        final_source = next(
            (
                metric
                for metric in _FINAL_METRIC_CANDIDATES
                if metric in by_metric
            ),
            None,
        )
        if final_source is not None:
            record = by_metric[
                final_source
            ]
            row[
                "final_performance_metric"
            ] = final_source
            for statistic in (
                "mean",
                "std",
                "ci_low",
                "ci_high",
                "n",
            ):
                row[
                    f"final_performance_{statistic}"
                ] = record[
                    statistic
                ]

        aliases = {
            "forgetting": (
                "forgetting",
                "mean_forgetting",
            ),
            "forward_transfer": (
                "forward_transfer",
                "fwt",
            ),
            "lifetime_auc": (
                "lifetime_auc",
            ),
            "t80": (
                "t80",
            ),
            "t90": (
                "t90",
            ),
            "plasticity_retention": (
                "plasticity_retention",
            ),
            "compute_seconds": (
                "duration_seconds",
                "resource.duration_seconds",
            ),
            "trainable_parameters": (
                "trainable_parameters",
                "parameter_count",
            ),
            "memory_overhead_bytes": (
                "memory_overhead_bytes",
                "peak_vram_bytes",
            ),
        }
        for output_name, candidates in aliases.items():
            source = next(
                (
                    metric
                    for metric in candidates
                    if metric
                    in by_metric
                ),
                None,
            )
            if source is None:
                continue
            record = by_metric[
                source
            ]
            row[
                f"{output_name}_metric"
            ] = source
            for statistic in (
                "mean",
                "std",
                "ci_low",
                "ci_high",
                "n",
            ):
                row[
                    f"{output_name}_{statistic}"
                ] = record[
                    statistic
                ]
        run_ids = sorted(
            {
                run_id
                for raw in values[
                    "run_ids"
                ]
                for run_id in json.loads(
                    raw
                )
            }
        )
        row[
            "run_ids"
        ] = json.dumps(
            run_ids
        )
        rows.append(
            row
        )
    return pd.DataFrame(
        rows
    )


def _format_cell(
    value: object,
) -> str:
    if value is None:
        return ""
    if isinstance(
        value,
        float,
    ):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _markdown(
    frame: pd.DataFrame,
) -> str:
    if frame.columns.empty:
        return "| No data |\n| --- |\n"
    columns = [
        str(column)
        for column in frame.columns
    ]
    lines = [
        "| "
        + " | ".join(
            columns
        )
        + " |",
        "| "
        + " | ".join(
            "---"
            for _ in columns
        )
        + " |",
    ]
    for record in frame.to_dict(
        "records"
    ):
        lines.append(
            "| "
            + " | ".join(
                _format_cell(
                    record.get(
                        column
                    )
                ).replace(
                    "|",
                    "\\|",
                )
                for column in columns
            )
            + " |"
        )
    return "\n".join(
        lines
    ) + "\n"


def _latex_escape(
    value: object,
) -> str:
    text = _format_cell(
        value
    )
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(
        replacements.get(
            character,
            character,
        )
        for character in text
    )


def _latex(
    frame: pd.DataFrame,
) -> str:
    columns = [
        str(column)
        for column in frame.columns
    ]
    if not columns:
        return (
            "\\begin{tabular}{l}\n"
            "No data \\\\\n"
            "\\end{tabular}\n"
        )
    alignment = (
        "l"
        * len(columns)
    )
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(
            _latex_escape(
                column
            )
            for column in columns
        )
        + r" \\",
        "\\hline",
    ]
    for record in frame.to_dict(
        "records"
    ):
        lines.append(
            " & ".join(
                _latex_escape(
                    record.get(
                        column
                    )
                )
                for column in columns
            )
            + r" \\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
        ]
    )
    return "\n".join(
        lines
    ) + "\n"


def export_table(
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
) -> list[str]:
    """Export one table in CSV, Markdown, and LaTeX formats."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths = {
        "csv": output_dir
        / f"{stem}.csv",
        "md": output_dir
        / f"{stem}.md",
        "tex": output_dir
        / f"{stem}.tex",
    }
    frame.to_csv(
        paths["csv"],
        index=False,
    )
    paths[
        "md"
    ].write_text(
        _markdown(
            frame
        ),
        encoding="utf-8",
    )
    paths[
        "tex"
    ].write_text(
        _latex(
            frame
        ),
        encoding="utf-8",
    )
    return [
        path.name
        for path in paths.values()
    ]


def _safe_name(
    value: str,
) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    ).strip(
        "_"
    )
    return (
        normalized
        if normalized
        else "figure"
    )


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> list[str]:
    artifacts: list[str] = []
    for extension in formats:
        path = output_dir / (
            f"{stem}.{extension}"
        )
        figure.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
        )
        artifacts.append(
            path.name
        )
    plt.close(
        figure
    )
    return artifacts


def _final_performance_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[
    list[str],
    str | None,
]:
    required = {
        "final_performance_mean",
        "final_performance_ci_low",
        "final_performance_ci_high",
    }
    if (
        summary.empty
        or not required.issubset(
            summary.columns
        )
    ):
        return [], (
            "no common final-performance metric was available"
        )
    values = summary.dropna(
        subset=[
            "final_performance_mean",
        ]
    ).copy()
    if values.empty:
        return [], (
            "no finite final-performance values were available"
        )
    labels = [
        f"{row['method']}\n{row['benchmark']}\n{row['setting']}"
        for row in values.to_dict(
            "records"
        )
    ]
    means = values[
        "final_performance_mean"
    ].to_numpy(
        dtype=float
    )
    lower = means - values[
        "final_performance_ci_low"
    ].to_numpy(
        dtype=float
    )
    upper = values[
        "final_performance_ci_high"
    ].to_numpy(
        dtype=float
    ) - means
    figure = plt.figure(
        figsize=(
            max(
                6.0,
                1.4
                * len(labels),
            ),
            4.5,
        )
    )
    axis = figure.add_subplot(
        111
    )
    positions = np.arange(
        len(labels)
    )
    axis.errorbar(
        positions,
        means,
        yerr=np.vstack(
            (
                lower,
                upper,
            )
        ),
        fmt="o",
        capsize=3,
    )
    axis.set_xticks(
        positions,
        labels,
        rotation=35,
        ha="right",
    )
    axis.set_ylabel(
        "final performance"
    )
    axis.set_title(
        "Final performance with bootstrap confidence intervals"
    )
    figure.tight_layout()
    return (
        _save_figure(
            figure,
            output_dir,
            "final_performance",
            formats,
        ),
        None,
    )


def _resource_tradeoff_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[
    list[str],
    str | None,
]:
    required = {
        "final_performance_mean",
        "compute_seconds_mean",
    }
    if (
        summary.empty
        or not required.issubset(
            summary.columns
        )
    ):
        return [], (
            "compute and final-performance metrics were not both available"
        )
    values = summary.dropna(
        subset=list(
            required
        )
    )
    if values.empty:
        return [], (
            "compute/performance rows were empty"
        )
    figure = plt.figure()
    axis = figure.add_subplot(
        111
    )
    axis.scatter(
        values[
            "compute_seconds_mean"
        ],
        values[
            "final_performance_mean"
        ],
    )
    for record in values.to_dict(
        "records"
    ):
        axis.annotate(
            str(
                record[
                    "method"
                ]
            ),
            (
                float(
                    record[
                        "compute_seconds_mean"
                    ]
                ),
                float(
                    record[
                        "final_performance_mean"
                    ]
                ),
            ),
        )
    axis.set_xlabel(
        "wall-clock seconds"
    )
    axis.set_ylabel(
        "final performance"
    )
    axis.set_title(
        "Compute/performance tradeoff"
    )
    figure.tight_layout()
    return (
        _save_figure(
            figure,
            output_dir,
            "compute_performance",
            formats,
        ),
        None,
    )


def _curve_columns(
    metrics: pd.DataFrame,
) -> tuple[
    str,
    str,
] | None:
    x = next(
        (
            column
            for column in (
                "environment_step",
                "step",
                "row_index",
            )
            if column
            in metrics.columns
        ),
        None,
    )
    y = next(
        (
            column
            for column in (
                "evaluation_return",
                "episode_return",
                "return",
                "mean_return",
            )
            if column
            in metrics.columns
        ),
        None,
    )
    if x is None or y is None:
        return None
    return x, y


def _learning_curve_figures(
    runs: Sequence[LoadedRun],
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[
    list[str],
    list[str],
]:
    artifacts: list[str] = []
    skipped: list[str] = []
    for group, group_runs in sorted(
        _group_runs(
            runs
        ).items(),
        key=lambda item: item[
            0
        ].label,
    ):
        available = [
            (
                run,
                _curve_columns(
                    run.metrics
                ),
            )
            for run in group_runs
        ]
        columns = [
            pair
            for _, pair in available
            if pair is not None
        ]
        if not columns:
            skipped.append(
                f"{group.label}: no learning-curve x/y columns"
            )
            continue
        if any(
            pair != columns[0]
            for pair in columns
        ) or len(columns) != len(
            group_runs
        ):
            skipped.append(
                f"{group.label}: inconsistent learning-curve schema across seeds"
            )
            continue
        x_column, y_column = (
            columns[0]
        )
        by_x: dict[
            float,
            list[float],
        ] = defaultdict(list)
        for run in group_runs:
            frame = run.metrics[
                [
                    x_column,
                    y_column,
                ]
            ].dropna()
            for x_value, y_value in frame.itertuples(
                index=False,
                name=None,
            ):
                x_float = float(
                    x_value
                )
                y_float = float(
                    y_value
                )
                if (
                    math.isfinite(
                        x_float
                    )
                    and math.isfinite(
                        y_float
                    )
                ):
                    by_x[
                        x_float
                    ].append(
                        y_float
                    )
        if not by_x:
            skipped.append(
                f"{group.label}: no finite learning-curve points"
            )
            continue

        xs = sorted(
            by_x
        )
        means = [
            float(
                np.mean(
                    by_x[x]
                )
            )
            for x in xs
        ]
        figure = plt.figure()
        axis = figure.add_subplot(
            111
        )
        axis.plot(
            xs,
            means,
        )
        axis.set_xlabel(
            x_column
        )
        axis.set_ylabel(
            y_column
        )
        axis.set_title(
            group.label
        )
        figure.tight_layout()
        stem = (
            "learning_curve_"
            + _safe_name(
                group.label
            )
        )
        artifacts.extend(
            _save_figure(
                figure,
                output_dir,
                stem,
                formats,
            )
        )
    return (
        artifacts,
        skipped,
    )


def build_paper_artifacts(
    results_root: str | Path,
    output_dir: str | Path,
    *,
    config: PaperArtifactConfig | None = None,
) -> dict[str, object]:
    """Build paper-ready statistics strictly from completed raw run artifacts."""

    resolved = (
        config
        or PaperArtifactConfig()
    )
    resolved.validate()
    runs = load_paper_runs(
        results_root
    )
    output = Path(
        output_dir
    )
    tables_dir = (
        output
        / "tables"
    )
    figures_dir = (
        output
        / "figures"
    )
    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate = (
        aggregate_scalar_metrics(
            runs,
            config=resolved,
        )
    )
    task_aggregate = (
        aggregate_task_metrics(
            runs,
            config=resolved,
        )
    )
    paired = (
        paired_method_differences(
            runs,
            config=resolved,
        )
    )
    summary = (
        results_summary_table(
            aggregate
        )
    )
    provenance = run_index(
        runs
    )
    access = (
        information_access_table(
            runs
        )
    )

    table_artifacts: list[
        str
    ] = []
    for stem, frame in (
        (
            "aggregate_statistics",
            aggregate,
        ),
        (
            "hierarchical_task_statistics",
            task_aggregate,
        ),
        (
            "paired_differences",
            paired,
        ),
        (
            "results_summary",
            summary,
        ),
        (
            "information_access",
            access,
        ),
        (
            "run_index",
            provenance,
        ),
    ):
        table_artifacts.extend(
            f"tables/{name}"
            for name in export_table(
                frame,
                tables_dir,
                stem,
            )
        )

    figure_artifacts: list[
        str
    ] = []
    skipped_figures: list[
        str
    ] = []
    final_files, reason = (
        _final_performance_figure(
            summary,
            figures_dir,
            resolved.figure_formats,
        )
    )
    figure_artifacts.extend(
        f"figures/{name}"
        for name in final_files
    )
    if reason is not None:
        skipped_figures.append(
            "final_performance: "
            + reason
        )

    compute_files, reason = (
        _resource_tradeoff_figure(
            summary,
            figures_dir,
            resolved.figure_formats,
        )
    )
    figure_artifacts.extend(
        f"figures/{name}"
        for name in compute_files
    )
    if reason is not None:
        skipped_figures.append(
            "compute_performance: "
            + reason
        )

    curve_files, curve_skips = (
        _learning_curve_figures(
            runs,
            figures_dir,
            resolved.figure_formats,
        )
    )
    figure_artifacts.extend(
        f"figures/{name}"
        for name in curve_files
    )
    skipped_figures.extend(
        curve_skips
    )

    sources = [
        {
            "run_id": run.manifest.run_id,
            "path": str(
                run.path
            ),
            "git_commit": (
                run.manifest.git_commit
            ),
            "source_hashes": (
                run.source_hashes
            ),
        }
        for run in runs
    ]
    manifest = {
        "schema_version": 1,
        "config": asdict(
            resolved
        ),
        "results_root": str(
            Path(
                results_root
            ).resolve()
        ),
        "run_count": len(
            runs
        ),
        "source_runs": sources,
        "generated_tables": (
            table_artifacts
        ),
        "generated_figures": (
            figure_artifacts
        ),
        "skipped_figures": (
            skipped_figures
        ),
        "integrity": {
            "manual_result_transcription": False,
            "incomplete_runs_allowed": False,
            "duplicate_run_ids_allowed": False,
            "duplicate_group_seed_allowed": False,
            "raw_source_hashes_recorded": True,
        },
    }
    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path = (
        output
        / "paper_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest[
        "manifest_path"
    ] = str(
        manifest_path
    )
    return manifest
