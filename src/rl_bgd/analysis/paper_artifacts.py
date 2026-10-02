"""Automatic paper tables, figures, bootstrap statistics, and provenance manifests."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
            raise ValueError("paper confidence must lie strictly between 0 and 1")
        if self.bootstrap_resamples < 1:
            raise ValueError("paper bootstrap resamples must be positive")
        if self.seed < 0:
            raise ValueError("paper artifact seed must be non-negative")
        allowed = {
            "png",
            "pdf",
            "svg",
        }
        if not self.figure_formats or any(value not in allowed for value in self.figure_formats):
            raise ValueError("figure_formats must be a non-empty subset of png/pdf/svg")


@dataclass(frozen=True)
class RunGroup:
    experiment: str
    method: str
    setting: str
    benchmark: str

    @property
    def label(self) -> str:
        return f"{self.experiment} | {self.method} | {self.setting} | {self.benchmark}"


def _experiment_id(
    run: LoadedRun,
) -> str:
    job = run.manifest.metadata.get("job_id")
    suite = run.manifest.metadata.get("suite")
    if (
        isinstance(
            job,
            str,
        )
        and job.strip()
    ):
        if (
            isinstance(
                suite,
                str,
            )
            and suite.strip()
        ):
            return f"{suite}/{job}"
        return job
    return run.manifest.method


def _group(
    run: LoadedRun,
) -> RunGroup:
    return RunGroup(
        experiment=_experiment_id(run),
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
        grouped[_group(run)].append(run)

    for group, values in grouped.items():
        seeds = [run.manifest.seed for run in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed detected within paper group {group.label}")

        reference = values[0]
        reference_suite = reference.manifest.metadata.get("suite")
        reference_contract = {
            "git_commit": reference.manifest.git_commit,
            "information_access": reference.manifest.information_access,
            "hypothesis_id": reference.manifest.metadata.get("hypothesis_id"),
            "target": reference.manifest.metadata.get("target"),
            "source_config_path": reference.manifest.metadata.get("source_config_path"),
            "declared_metrics": _declared_metric_names(reference),
            "contract_kwargs": reference.manifest.metadata.get("contract_kwargs"),
        }
        for run in values[1:]:
            contract = {
                "git_commit": run.manifest.git_commit,
                "information_access": run.manifest.information_access,
                "hypothesis_id": run.manifest.metadata.get("hypothesis_id"),
                "target": run.manifest.metadata.get("target"),
                "source_config_path": run.manifest.metadata.get("source_config_path"),
                "declared_metrics": _declared_metric_names(run),
                "contract_kwargs": run.manifest.metadata.get("contract_kwargs"),
            }
            for field, expected in reference_contract.items():
                if field == "git_commit" and reference_suite is None:
                    continue
                if contract[field] != expected:
                    raise ValueError(
                        "matched paper seeds have inconsistent run contracts: "
                        f"{group.label} / {field} / "
                        f"{reference.manifest.run_id}={expected!r} / "
                        f"{run.manifest.run_id}={contract[field]!r}"
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

    directories = discover_run_directories(results_root)
    if not directories:
        raise ValueError("paper build found no run manifests")
    runs = tuple(
        load_run_directory(
            path,
            require_completed=True,
        )
        for path in directories
    )
    for run in runs:
        primary_metric = run.manifest.metadata.get("primary_metric")
        if primary_metric is not None and (
            not isinstance(
                primary_metric,
                str,
            )
            or not primary_metric
        ):
            raise ValueError("primary_metric metadata must be a non-empty string")
        if (
            isinstance(
                primary_metric,
                str,
            )
            and primary_metric not in run.summary.metrics
            and primary_metric not in run.summary.resources
        ):
            raise ValueError(
                "completed run "
                f"{run.manifest.run_id!r} does not expose declared "
                f"primary metric {primary_metric!r}"
            )

    run_ids = [run.manifest.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run_id values must be globally unique within a paper build")
    return runs


def _declared_metric_names(
    run: LoadedRun,
) -> tuple[str, ...] | None:
    """Return suite-declared paper metrics, or None for non-suite artifacts."""

    primary = (
        run.manifest.metadata.get(
            "primary_metric"
        )
    )
    if primary is None:
        return None
    if (
        not isinstance(
            primary,
            str,
        )
        or not primary
    ):
        raise ValueError(
            f"{run.manifest.run_id}: primary_metric metadata must be a non-empty string"
        )

    secondary_raw = (
        run.manifest.metadata.get(
            "secondary_metrics",
            (),
        )
    )
    if (
        isinstance(
            secondary_raw,
            (str, bytes),
        )
        or not isinstance(
            secondary_raw,
            Sequence,
        )
    ):
        raise ValueError(
            f"{run.manifest.run_id}: secondary_metrics metadata must be a sequence"
        )

    secondary: list[str] = []
    for value in secondary_raw:
        if (
            not isinstance(
                value,
                str,
            )
            or not value
        ):
            raise ValueError(
                f"{run.manifest.run_id}: secondary metric names must be non-empty strings"
            )
        secondary.append(
            value
        )
    return tuple(
        dict.fromkeys(
            (
                primary,
                *secondary,
            )
        )
    )


def _resolve_declared_scalar_metric(
    run: LoadedRun,
    name: str,
) -> float | None:
    metrics = run.summary.metrics
    if name in metrics:
        return metrics[name]
    if name in run.summary.resources:
        return run.summary.resources[name]

    suffix = (
        "."
        + name
    )
    matches = [
        value
        for key, value in metrics.items()
        if key.endswith(
            suffix
        )
    ]
    if len(
        matches
    ) > 1:
        raise ValueError(
            f"{run.manifest.run_id}: declared metric {name!r} is ambiguous"
        )
    if matches:
        return matches[
            0
        ]
    return None


def _metric_map(
    run: LoadedRun,
) -> dict[str, float]:
    """Expose paper-authorized scalar outcomes plus recorded resources."""

    declared = (
        _declared_metric_names(
            run
        )
    )
    declared_resource_names: set[str] = set()
    if declared is None:
        values = dict(
            run.summary.metrics
        )
    else:
        values: dict[
            str,
            float,
        ] = {}
        for name in declared:
            value = (
                _resolve_declared_scalar_metric(
                    run,
                    name,
                )
            )
            if value is not None:
                values[
                    name
                ] = value
                if (
                    name in run.summary.resources
                    and name not in run.summary.metrics
                ):
                    declared_resource_names.add(name)

        primary = declared[
            0
        ]
        if primary not in values:
            raise ValueError(
                f"{run.manifest.run_id}: declared primary metric {primary!r} "
                "is not a scalar paper metric"
            )

    for key, value in run.summary.resources.items():
        if key in declared_resource_names:
            continue
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

    rows: list[dict[str, object]] = []
    grouped = _group_runs(runs)
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
        metric_maps = [_metric_map(run) for run in group_runs]
        metric_names = sorted(set().union(*(set(values) for values in metric_maps)))
        for metric_index, metric in enumerate(metric_names):
            presence = [metric in values for values in metric_maps]
            if not all(presence):
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
            values = [metric_map[metric] for metric_map in metric_maps]
            estimate = bootstrap_mean_ci(
                values,
                confidence=(config.confidence),
                resamples=(config.bootstrap_resamples),
                seed=(config.seed + group_index * 100_000 + metric_index),
            )
            rows.append(
                {
                    "experiment": group.experiment,
                    "method": group.method,
                    "setting": group.setting,
                    "benchmark": group.benchmark,
                    "metric": metric,
                    **estimate.to_dict(),
                    "run_ids": json.dumps([run.manifest.run_id for run in group_runs]),
                    "seeds": json.dumps([run.manifest.seed for run in group_runs]),
                    "raw_values": json.dumps(values),
                }
            )
    return pd.DataFrame(rows)


def aggregate_task_metrics(
    runs: Sequence[LoadedRun],
    *,
    config: PaperArtifactConfig,
) -> pd.DataFrame:
    """Hierarchically bootstrap task-level values nested inside seeds."""

    rows: list[dict[str, object]] = []
    grouped = _group_runs(runs)
    for group_index, (
        group,
        group_runs,
    ) in enumerate(
        sorted(
            grouped.items(),
            key=lambda item: item[0].label,
        )
    ):
        metric_names = sorted(
            {
                metric
                for run in group_runs
                for values in (run.summary.task_metrics.values())
                for metric in values
            }
        )
        for metric_index, metric in enumerate(metric_names):
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
                    task_values[metric]
                    for task_values in (run.summary.task_metrics.values())
                    if metric in task_values
                ]
                if not values:
                    raise ValueError(
                        "task metric is missing for an entire matched seed: "
                        f"{group.label} / {metric} / {run.manifest.run_id}"
                    )
                seed_values[run.manifest.seed] = values
                run_task_counts[run.manifest.run_id] = len(values)
            estimate = hierarchical_bootstrap_mean(
                seed_values,
                confidence=(config.confidence),
                resamples=(config.bootstrap_resamples),
                seed=(config.seed + 1_000_000 + group_index * 100_000 + metric_index),
            )
            rows.append(
                {
                    "experiment": group.experiment,
                    "method": group.method,
                    "setting": group.setting,
                    "benchmark": group.benchmark,
                    "metric": metric,
                    **estimate.to_dict(),
                    "run_ids": json.dumps([run.manifest.run_id for run in group_runs]),
                    "task_counts_by_run": json.dumps(
                        run_task_counts,
                        sort_keys=True,
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_method_differences(
    runs: Sequence[LoadedRun],
    *,
    config: PaperArtifactConfig,
) -> pd.DataFrame:
    """Create matched-seed method differences within suite/setting/benchmark."""

    grouped = _group_runs(runs)
    by_context: dict[
        tuple[str, str, str],
        list[
            tuple[
                RunGroup,
                list[LoadedRun],
            ]
        ],
    ] = defaultdict(list)
    for group, group_runs in grouped.items():
        suite = (
            group.experiment.split(
                "/",
                1,
            )[0]
            if "/" in group.experiment
            else ""
        )
        by_context[
            (
                suite,
                group.setting,
                group.benchmark,
            )
        ].append(
            (
                group,
                group_runs,
            )
        )

    rows: list[dict[str, object]] = []
    comparison_index = 0
    for (
        suite,
        setting,
        benchmark,
    ), groups in sorted(by_context.items()):
        ordered = sorted(
            groups,
            key=lambda item: (
                item[0].method,
                item[0].experiment,
            ),
        )
        for left_index in range(len(ordered)):
            for right_index in range(
                left_index + 1,
                len(ordered),
            ):
                (
                    left_group,
                    left_runs,
                ) = ordered[left_index]
                (
                    right_group,
                    right_runs,
                ) = ordered[right_index]
                left_by_seed = {run.manifest.seed: run for run in left_runs}
                right_by_seed = {run.manifest.seed: run for run in right_runs}
                left_seeds = set(left_by_seed)
                right_seeds = set(right_by_seed)
                if left_seeds != right_seeds:
                    raise ValueError(
                        "paired paper methods have different seed sets: "
                        f"{left_group.label} / {sorted(left_seeds)} versus "
                        f"{right_group.label} / {sorted(right_seeds)}"
                    )
                matched_seeds = sorted(left_seeds)
                if not matched_seeds:
                    continue
                for seed in matched_seeds:
                    left_run = left_by_seed[seed]
                    right_run = right_by_seed[seed]
                    if left_run.manifest.git_commit != right_run.manifest.git_commit:
                        raise ValueError(
                            "paired paper runs use different git commits: "
                            f"seed {seed} / {left_run.manifest.run_id}="
                            f"{left_run.manifest.git_commit!r} / "
                            f"{right_run.manifest.run_id}="
                            f"{right_run.manifest.git_commit!r}"
                        )
                    if (
                        left_run.manifest.information_access
                        != right_run.manifest.information_access
                    ):
                        raise ValueError(
                            "paired paper runs have different information access: "
                            f"seed {seed} / {left_run.manifest.run_id} / "
                            f"{right_run.manifest.run_id}"
                        )
                    if left_run.manifest.task_order != right_run.manifest.task_order:
                        raise ValueError(
                            "paired paper runs have different task order: "
                            f"seed {seed} / {left_run.manifest.run_id} / "
                            f"{right_run.manifest.run_id}"
                        )

                left_metrics = {metric for run in left_runs for metric in _metric_map(run)}
                right_metrics = {metric for run in right_runs for metric in _metric_map(run)}
                common_metrics = sorted(left_metrics & right_metrics)
                for metric_index, metric in enumerate(common_metrics):
                    left_values: list[float] = []
                    right_values: list[float] = []
                    run_pairs: list[tuple[str, str]] = []
                    complete = True
                    for seed in matched_seeds:
                        left_map = _metric_map(left_by_seed[seed])
                        right_map = _metric_map(right_by_seed[seed])
                        if metric not in left_map or metric not in right_map:
                            complete = False
                            break
                        left_values.append(left_map[metric])
                        right_values.append(right_map[metric])
                        run_pairs.append(
                            (
                                left_by_seed[seed].manifest.run_id,
                                right_by_seed[seed].manifest.run_id,
                            )
                        )
                    if not complete:
                        continue
                    estimate = paired_bootstrap_difference(
                        left_values,
                        right_values,
                        confidence=(config.confidence),
                        resamples=(config.bootstrap_resamples),
                        seed=(config.seed + 2_000_000 + comparison_index * 100_000 + metric_index),
                    )
                    rows.append(
                        {
                            "suite": suite,
                            "setting": setting,
                            "benchmark": benchmark,
                            "left_experiment": (left_group.experiment),
                            "left_method": (left_group.method),
                            "right_experiment": (right_group.experiment),
                            "right_method": (right_group.method),
                            "metric": metric,
                            "difference": ("left_minus_right"),
                            **estimate.to_dict(),
                            "matched_seeds": json.dumps(matched_seeds),
                            "run_pairs": json.dumps(run_pairs),
                        }
                    )
                comparison_index += 1
    return pd.DataFrame(rows)


def run_index(
    runs: Sequence[LoadedRun],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
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
                "experiment": _experiment_id(run),
                "method": run.manifest.method,
                "setting": run.manifest.setting,
                "benchmark": run.manifest.benchmark,
                "seed": run.manifest.seed,
                "git_commit": run.manifest.git_commit,
                "task_order": json.dumps(list(run.manifest.task_order)),
                "run_path": str(run.path),
            }
        )
    return pd.DataFrame(rows)


def information_access_table(
    runs: Sequence[LoadedRun],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_runs in sorted(
        _group_runs(runs).items(),
        key=lambda item: item[0].label,
    ):
        access_values = [run.manifest.information_access for run in group_runs]
        first = access_values[0]
        if any(value != first for value in access_values[1:]):
            raise ValueError(
                f"information-access assumptions differ within a matched group: {group.label}"
            )
        row: dict[
            str,
            object,
        ] = {
            "experiment": group.experiment,
            "method": group.method,
            "setting": group.setting,
            "benchmark": group.benchmark,
            "run_ids": json.dumps([run.manifest.run_id for run in group_runs]),
        }
        row.update({key: bool(value) for key, value in sorted(first.items())})
        rows.append(row)
    return pd.DataFrame(rows)


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
    rows: list[dict[str, object]] = []
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
        by_metric = {str(record["metric"]): record for record in values.to_dict("records")}

        final_source = next(
            (metric for metric in _FINAL_METRIC_CANDIDATES if metric in by_metric),
            None,
        )
        if final_source is not None:
            record = by_metric[final_source]
            row["final_performance_metric"] = final_source
            for statistic in (
                "mean",
                "std",
                "ci_low",
                "ci_high",
                "n",
            ):
                row[f"final_performance_{statistic}"] = record[statistic]

        aliases = {
            "forgetting": (
                "forgetting",
                "mean_forgetting",
            ),
            "forward_transfer": (
                "forward_transfer",
                "fwt",
            ),
            "lifetime_auc": ("lifetime_auc",),
            "t80": ("t80",),
            "t90": ("t90",),
            "plasticity_retention": ("plasticity_retention",),
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
                (metric for metric in candidates if metric in by_metric),
                None,
            )
            if source is None:
                continue
            record = by_metric[source]
            row[f"{output_name}_metric"] = source
            for statistic in (
                "mean",
                "std",
                "ci_low",
                "ci_high",
                "n",
            ):
                row[f"{output_name}_{statistic}"] = record[statistic]
        run_ids = sorted({run_id for raw in values["run_ids"] for run_id in json.loads(raw)})
        row["run_ids"] = json.dumps(run_ids)
        rows.append(row)
    return pd.DataFrame(rows)


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
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in frame.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                _format_cell(record.get(column)).replace(
                    "|",
                    "\\|",
                )
                for column in columns
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _latex_escape(
    value: object,
) -> str:
    text = _format_cell(value)
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
    columns = [str(column) for column in frame.columns]
    if not columns:
        return "\\begin{tabular}{l}\nNo data \\\\\n\\end{tabular}\n"
    alignment = "l" * len(columns)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(_latex_escape(column) for column in columns) + r" \\",
        "\\hline",
    ]
    for record in frame.to_dict("records"):
        lines.append(" & ".join(_latex_escape(record.get(column)) for column in columns) + r" \\")
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
        ]
    )
    return "\n".join(lines) + "\n"


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
        "csv": output_dir / f"{stem}.csv",
        "md": output_dir / f"{stem}.md",
        "tex": output_dir / f"{stem}.tex",
    }
    frame.to_csv(
        paths["csv"],
        index=False,
    )
    paths["md"].write_text(
        _markdown(frame),
        encoding="utf-8",
    )
    paths["tex"].write_text(
        _latex(frame),
        encoding="utf-8",
    )
    return [path.name for path in paths.values()]


def _safe_name(
    value: str,
) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    ).strip("_")
    return normalized if normalized else "figure"


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> list[str]:
    artifacts: list[str] = []
    for extension in formats:
        path = output_dir / (f"{stem}.{extension}")
        figure.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
        )
        artifacts.append(path.name)
    plt.close(figure)
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
    if summary.empty or not required.issubset(summary.columns):
        return [], ("no common final-performance metric was available")
    values = summary.dropna(
        subset=[
            "final_performance_mean",
        ]
    ).copy()
    if values.empty:
        return [], ("no finite final-performance values were available")
    labels = [
        f"{row['method']}\n{row['benchmark']}\n{row['setting']}"
        for row in values.to_dict("records")
    ]
    means = values["final_performance_mean"].to_numpy(dtype=float)
    lower = means - values["final_performance_ci_low"].to_numpy(dtype=float)
    upper = values["final_performance_ci_high"].to_numpy(dtype=float) - means
    figure = plt.figure(
        figsize=(
            max(
                6.0,
                1.4 * len(labels),
            ),
            4.5,
        )
    )
    axis = figure.add_subplot(111)
    positions = np.arange(len(labels))
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
    axis.set_ylabel("final performance")
    axis.set_title("Final performance with bootstrap confidence intervals")
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
    if summary.empty or not required.issubset(summary.columns):
        return [], ("compute and final-performance metrics were not both available")
    values = summary.dropna(subset=list(required))
    if values.empty:
        return [], ("compute/performance rows were empty")
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.scatter(
        values["compute_seconds_mean"],
        values["final_performance_mean"],
    )
    for record in values.to_dict("records"):
        axis.annotate(
            str(record["method"]),
            (
                float(record["compute_seconds_mean"]),
                float(record["final_performance_mean"]),
            ),
        )
    axis.set_xlabel("wall-clock seconds")
    axis.set_ylabel("final performance")
    axis.set_title("Compute/performance tradeoff")
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
) -> (
    tuple[
        str,
        str,
    ]
    | None
):
    x = next(
        (
            column
            for column in (
                "environment_step",
                "step",
                "row_index",
            )
            if column in metrics.columns
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
            if column in metrics.columns
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
    *,
    config: PaperArtifactConfig,
) -> tuple[
    list[str],
    list[str],
]:
    """Plot learning curves with matched-seed bootstrap confidence bands."""

    artifacts: list[str] = []
    skipped: list[str] = []
    for group_index, (
        group,
        group_runs,
    ) in enumerate(
        sorted(
            _group_runs(runs).items(),
            key=lambda item: item[0].label,
        )
    ):
        available = [
            (
                run,
                _curve_columns(run.metrics),
            )
            for run in group_runs
        ]
        columns = [pair for _, pair in available if pair is not None]
        if not columns:
            skipped.append(f"{group.label}: no learning-curve x/y columns")
            continue
        if any(pair != columns[0] for pair in columns) or len(columns) != len(group_runs):
            skipped.append(f"{group.label}: inconsistent learning-curve schema across seeds")
            continue
        x_column, y_column = columns[0]

        seed_series: list[dict[float, float]] = []
        for run in group_runs:
            frame = run.metrics[
                [
                    x_column,
                    y_column,
                ]
            ].dropna()
            grouped_values: dict[
                float,
                list[float],
            ] = defaultdict(list)
            for (
                x_value,
                y_value,
            ) in frame.itertuples(
                index=False,
                name=None,
            ):
                x_float = float(x_value)
                y_float = float(y_value)
                if math.isfinite(x_float) and math.isfinite(y_float):
                    grouped_values[x_float].append(y_float)
            seed_series.append(
                {
                    x_value: float(np.mean(values))
                    for (
                        x_value,
                        values,
                    ) in grouped_values.items()
                }
            )

        (
            xs,
            means,
            lows,
            highs,
        ) = _pointwise_seed_statistics(
            seed_series,
            config=config,
            seed_offset=(3_000_000 + group_index * 10_000),
        )
        if not xs:
            skipped.append(f"{group.label}: no aligned finite learning-curve points")
            continue

        figure = plt.figure()
        axis = figure.add_subplot(111)
        axis.plot(
            xs,
            means,
        )
        axis.fill_between(
            xs,
            lows,
            highs,
            alpha=0.2,
        )
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.set_title(group.label)
        figure.tight_layout()
        stem = "learning_curve_" + _safe_name(group.label)
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


def _pointwise_seed_statistics(
    seed_series: Sequence[Mapping[float, float]],
    *,
    config: PaperArtifactConfig,
    seed_offset: int,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Return aligned x, mean, lower, and upper bootstrap bands across seeds."""

    if not seed_series:
        return (
            [],
            [],
            [],
            [],
        )
    common_x = set(seed_series[0])
    for series in seed_series[1:]:
        common_x &= set(series)
    xs = sorted(common_x)
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for index, x_value in enumerate(xs):
        values = [series[x_value] for series in seed_series]
        estimate = bootstrap_mean_ci(
            values,
            confidence=config.confidence,
            resamples=config.bootstrap_resamples,
            seed=(config.seed + seed_offset + index),
        )
        means.append(estimate.mean)
        lows.append(estimate.ci_low)
        highs.append(estimate.ci_high)
    return (
        xs,
        means,
        lows,
        highs,
    )


def _matrix_figures(
    runs: Sequence[LoadedRun],
    output_dir: Path,
    formats: Sequence[str],
    *,
    config: PaperArtifactConfig,
) -> tuple[
    list[str],
    list[str],
]:
    """Plot seed-mean continual matrices and stage-average bootstrap bands."""

    artifacts: list[str] = []
    skipped: list[str] = []
    for group_index, (
        group,
        group_runs,
    ) in enumerate(
        sorted(
            _group_runs(runs).items(),
            key=lambda item: item[0].label,
        )
    ):
        for series_index, series_name in enumerate(
            (
                "return_matrix",
                "success_matrix",
            )
        ):
            run_matrices: list[np.ndarray] = []
            stage_labels: (
                tuple[
                    str,
                    ...,
                ]
                | None
            ) = None
            task_labels: (
                tuple[
                    str,
                    ...,
                ]
                | None
            ) = None
            missing = False

            for run in group_runs:
                if "series" not in run.metrics.columns:
                    missing = True
                    break
                frame = run.metrics[run.metrics["series"] == series_name]
                if frame.empty:
                    missing = True
                    break
                required = {
                    "stage_index",
                    "task_index",
                    "stage_label",
                    "task_name",
                    "value",
                }
                if not required.issubset(frame.columns):
                    raise ValueError(
                        f"{group.label}: {series_name} rows are missing required columns"
                    )

                stages = tuple(
                    str(value)
                    for value in (
                        frame[
                            [
                                "stage_index",
                                "stage_label",
                            ]
                        ]
                        .drop_duplicates()
                        .sort_values("stage_index")["stage_label"]
                    )
                )
                tasks = tuple(
                    str(value)
                    for value in (
                        frame[
                            [
                                "task_index",
                                "task_name",
                            ]
                        ]
                        .drop_duplicates()
                        .sort_values("task_index")["task_name"]
                    )
                )
                pivot = (
                    frame.pivot(
                        index="stage_index",
                        columns="task_index",
                        values="value",
                    )
                    .sort_index()
                    .sort_index(axis=1)
                )
                matrix = pivot.to_numpy(dtype=float)
                if not np.isfinite(matrix).all() or matrix.shape != (
                    len(stages),
                    len(tasks),
                ):
                    raise ValueError(f"{group.label}: invalid {series_name} matrix")
                if stage_labels is None:
                    stage_labels = stages
                    task_labels = tasks
                elif (
                    stages != stage_labels
                    or tasks != task_labels
                    or matrix.shape != run_matrices[0].shape
                ):
                    raise ValueError(
                        f"{group.label}: {series_name} stage/task schema differs across seeds"
                    )
                run_matrices.append(matrix)

            if missing or not run_matrices or stage_labels is None or task_labels is None:
                skipped.append(f"{group.label}: no complete {series_name} across seeds")
                continue

            stacked = np.stack(
                run_matrices,
                axis=0,
            )
            mean_matrix = stacked.mean(axis=0)

            figure = plt.figure(
                figsize=(
                    max(
                        5.5,
                        0.55 * len(task_labels),
                    ),
                    max(
                        4.5,
                        0.45 * len(stage_labels),
                    ),
                )
            )
            axis = figure.add_subplot(111)
            image = axis.imshow(
                mean_matrix,
                aspect="auto",
            )
            figure.colorbar(
                image,
                ax=axis,
                label=("mean return" if series_name == "return_matrix" else "mean success rate"),
            )
            axis.set_xticks(
                np.arange(len(task_labels)),
                task_labels,
                rotation=45,
                ha="right",
            )
            axis.set_yticks(
                np.arange(len(stage_labels)),
                stage_labels,
            )
            axis.set_xlabel("evaluation task")
            axis.set_ylabel("training stage")
            axis.set_title(f"{group.label}\n{series_name}")
            figure.tight_layout()
            stem = "continual_matrix_" + _safe_name(series_name + "_" + group.label)
            artifacts.extend(
                _save_figure(
                    figure,
                    output_dir,
                    stem,
                    formats,
                )
            )

            seed_stage_series: list[dict[float, float]] = []
            for matrix in run_matrices:
                seed_stage_series.append(
                    {float(stage): float(matrix[stage].mean()) for stage in range(matrix.shape[0])}
                )
            (
                xs,
                means,
                lows,
                highs,
            ) = _pointwise_seed_statistics(
                seed_stage_series,
                config=config,
                seed_offset=(4_000_000 + group_index * 10_000 + series_index * 1_000),
            )
            if xs:
                figure = plt.figure()
                axis = figure.add_subplot(111)
                axis.plot(
                    xs,
                    means,
                )
                axis.fill_between(
                    xs,
                    lows,
                    highs,
                    alpha=0.2,
                )
                axis.set_xticks(
                    xs,
                    [stage_labels[int(value)] for value in xs],
                    rotation=45,
                    ha="right",
                )
                axis.set_xlabel("training stage")
                axis.set_ylabel(
                    "mean return across tasks"
                    if series_name == "return_matrix"
                    else "mean success across tasks"
                )
                axis.set_title(f"{group.label}\n{series_name} stage-average")
                figure.tight_layout()
                stem = "adaptation_curve_" + _safe_name(series_name + "_" + group.label)
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


_TIMELINE_VALUE_CANDIDATES: tuple[
    str,
    ...,
] = (
    "surprise",
    "surprise_raw",
    "surprise_smoothed",
    "retention_lambda",
    "sigma_mean",
    "critic1_sigma_mean",
    "critic2_sigma_mean",
    "actor_sigma_mean",
    "effective_lr_mean",
    "critic1_effective_lr_mean",
    "critic2_effective_lr_mean",
    "actor_effective_lr_mean",
    "evaluation_return",
    "episode_return",
    "return",
    "mean_return",
    "normalized_target_loss",
)


def _timeline_figures(
    runs: Sequence[LoadedRun],
    output_dir: Path,
    formats: Sequence[str],
    *,
    config: PaperArtifactConfig,
) -> tuple[
    list[str],
    list[str],
]:
    """Plot pointwise bootstrap bands for diagnostic timeline series."""

    artifacts: list[str] = []
    skipped: list[str] = []
    for group_index, (
        group,
        group_runs,
    ) in enumerate(
        sorted(
            _group_runs(runs).items(),
            key=lambda item: item[0].label,
        )
    ):
        if any("series" not in run.metrics.columns for run in group_runs):
            skipped.append(f"{group.label}: no raw timeline series column")
            continue
        common_series = set(str(value) for value in group_runs[0].metrics["series"].dropna())
        for run in group_runs[1:]:
            common_series &= set(str(value) for value in run.metrics["series"].dropna())
        common_series -= {
            "summary",
            "return_matrix",
            "success_matrix",
        }
        if not common_series:
            skipped.append(f"{group.label}: no shared diagnostic timeline across seeds")
            continue

        for series_index, series_name in enumerate(sorted(common_series)):
            frames = [
                run.metrics[run.metrics["series"] == series_name].copy() for run in group_runs
            ]
            x_column = next(
                (
                    candidate
                    for candidate in (
                        "environment_step",
                        "step",
                        "row_index",
                    )
                    if all(candidate in frame.columns for frame in frames)
                ),
                None,
            )
            if x_column is None:
                skipped.append(f"{group.label} / {series_name}: no shared x-axis column")
                continue
            value_columns = [
                candidate
                for candidate in _TIMELINE_VALUE_CANDIDATES
                if all(candidate in frame.columns for frame in frames)
            ]
            if not value_columns:
                skipped.append(
                    f"{group.label} / {series_name}: no supported diagnostic value column"
                )
                continue

            plotted = 0
            figure = plt.figure()
            axis = figure.add_subplot(111)
            for value_index, value_column in enumerate(value_columns):
                seed_series: list[dict[float, float]] = []
                for frame in frames:
                    subset = frame[
                        [
                            x_column,
                            value_column,
                        ]
                    ].dropna()
                    grouped_values: dict[
                        float,
                        list[float],
                    ] = defaultdict(list)
                    for (
                        x_value,
                        y_value,
                    ) in subset.itertuples(
                        index=False,
                        name=None,
                    ):
                        x_float = float(x_value)
                        y_float = float(y_value)
                        if math.isfinite(x_float) and math.isfinite(y_float):
                            grouped_values[x_float].append(y_float)
                    seed_series.append(
                        {
                            x_value: float(np.mean(values))
                            for (
                                x_value,
                                values,
                            ) in grouped_values.items()
                        }
                    )
                (
                    xs,
                    means,
                    lows,
                    highs,
                ) = _pointwise_seed_statistics(
                    seed_series,
                    config=config,
                    seed_offset=(
                        5_000_000
                        + group_index * 100_000
                        + series_index * 10_000
                        + value_index * 1_000
                    ),
                )
                if not xs:
                    continue
                axis.plot(
                    xs,
                    means,
                    label=value_column,
                )
                axis.fill_between(
                    xs,
                    lows,
                    highs,
                    alpha=0.15,
                )
                plotted += 1

            if plotted == 0:
                plt.close(figure)
                skipped.append(f"{group.label} / {series_name}: no aligned finite timeline points")
                continue
            axis.set_xlabel(x_column)
            axis.set_ylabel("diagnostic value")
            axis.set_title(f"{group.label}\n{series_name}")
            axis.legend()
            figure.tight_layout()
            stem = "timeline_" + _safe_name(series_name + "_" + group.label)
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

    resolved = config or PaperArtifactConfig()
    resolved.validate()
    runs = load_paper_runs(results_root)
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate = aggregate_scalar_metrics(
        runs,
        config=resolved,
    )
    task_aggregate = aggregate_task_metrics(
        runs,
        config=resolved,
    )
    paired = paired_method_differences(
        runs,
        config=resolved,
    )
    summary = results_summary_table(aggregate)
    provenance = run_index(runs)
    access = information_access_table(runs)

    table_artifacts: list[str] = []
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

    figure_artifacts: list[str] = []
    skipped_figures: list[str] = []
    final_files, reason = _final_performance_figure(
        summary,
        figures_dir,
        resolved.figure_formats,
    )
    figure_artifacts.extend(f"figures/{name}" for name in final_files)
    if reason is not None:
        skipped_figures.append("final_performance: " + reason)

    compute_files, reason = _resource_tradeoff_figure(
        summary,
        figures_dir,
        resolved.figure_formats,
    )
    figure_artifacts.extend(f"figures/{name}" for name in compute_files)
    if reason is not None:
        skipped_figures.append("compute_performance: " + reason)

    curve_files, curve_skips = _learning_curve_figures(
        runs,
        figures_dir,
        resolved.figure_formats,
        config=resolved,
    )
    figure_artifacts.extend(f"figures/{name}" for name in curve_files)
    skipped_figures.extend(curve_skips)

    matrix_files, matrix_skips = _matrix_figures(
        runs,
        figures_dir,
        resolved.figure_formats,
        config=resolved,
    )
    figure_artifacts.extend(f"figures/{name}" for name in matrix_files)
    skipped_figures.extend(matrix_skips)

    timeline_files, timeline_skips = _timeline_figures(
        runs,
        figures_dir,
        resolved.figure_formats,
        config=resolved,
    )
    figure_artifacts.extend(f"figures/{name}" for name in timeline_files)
    skipped_figures.extend(timeline_skips)

    sources = [
        {
            "run_id": run.manifest.run_id,
            "path": str(run.path),
            "git_commit": (run.manifest.git_commit),
            "source_hashes": (run.source_hashes),
        }
        for run in runs
    ]
    manifest = {
        "schema_version": 1,
        "config": asdict(resolved),
        "results_root": str(Path(results_root).resolve()),
        "run_count": len(runs),
        "source_runs": sources,
        "generated_tables": (table_artifacts),
        "generated_figures": (figure_artifacts),
        "skipped_figures": (skipped_figures),
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
    manifest_path = output / "paper_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
