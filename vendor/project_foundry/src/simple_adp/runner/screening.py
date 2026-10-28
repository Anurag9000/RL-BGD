"""Report-only hardness summaries from validation-only screening runs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def build_screening_reports(
    runs: Iterable[Any],
    results_dir: str | Path,
    *,
    classification_ceiling: float,
    regression_ceiling_r2: float,
    min_complexity_gain: float,
    min_hard_fold_fraction: float,
    screen_seed_count: int,
) -> dict[str, Any]:
    """Summarize screen predictions without reading or gating on test metrics."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    expected_seeds: dict[tuple[Any, ...], int] = {}
    task_kinds: dict[tuple[str, str, str], str | None] = {}
    expected_splits: dict[tuple[str, str, str], set[tuple[int, int, int]]] = defaultdict(set)
    for spec in runs:
        identity = spec.identity
        task_key = (identity.provider, identity.suite, identity.task_id)
        expected_splits[task_key].add((identity.split_repeat, identity.fold, identity.sample))
        if not spec.route.screening_only:
            continue
        arch = spec.route.screening_architecture
        if arch is None:
            continue
        depth, width = (int(arch[0]), int(arch[1]))
        fold_key = (*task_key, identity.split_repeat, identity.fold, identity.sample)
        architecture_key = (*fold_key, depth, width)
        expected_seeds[architecture_key] = screen_seed_count
        path = Path(spec.result_dir) / "result.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") != "completed" or not record.get("screening_only"):
            continue
        grouped[architecture_key].append(record)
        task_kinds[task_key] = record.get("task_kind")

    by_fold: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for architecture_key, seed_records in sorted(grouped.items()):
        task_key = architecture_key[:3]
        depth, width = architecture_key[-2:]
        kind = task_kinds.get(task_key)
        metric_name = "balanced_accuracy" if kind == "classification" else "r2"
        metric_values: list[float] = []
        ceiling_values: list[bool] = []
        for record in seed_records:
            metrics = record.get("validation_metrics") or {}
            value = metrics.get(metric_name)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            metric_values.append(value)
            ceiling_metric = metrics.get("accuracy") if kind == "classification" else metrics.get("r2")
            ceiling = classification_ceiling if kind == "classification" else regression_ceiling_r2
            ceiling_values.append(ceiling_metric is not None and float(ceiling_metric) >= ceiling)
        key = (*architecture_key[:6],)
        expected = expected_seeds.get(architecture_key, screen_seed_count)
        score = sum(metric_values) / len(metric_values) if len(metric_values) == expected else None
        architecture_record = {
            "depth": depth,
            "width": width,
            "expected_seeds": expected,
            "completed_seeds": len(metric_values),
            "validation_score_name": metric_name,
            "mean_validation_score": score,
            "shallow_ceiling_reached_all_seeds": len(ceiling_values) == expected and all(ceiling_values),
            "status": "complete" if score is not None else "incomplete",
        }
        by_fold[key].append(architecture_record)

    fold_records: list[dict[str, Any]] = []
    by_task: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for fold_key in sorted(expected_splits_fold_keys(runs)):
        task_key = fold_key[:3]
        kind = task_kinds.get(task_key)
        records = by_fold.get(fold_key, [])
        expected_architectures = 0
        for spec in runs:
            if not spec.route.screening_only:
                continue
            identity = spec.identity
            if (*task_key, identity.split_repeat, identity.fold, identity.sample) == fold_key:
                expected_architectures += 1
        expected_architectures //= max(1, screen_seed_count)
        completed = [r for r in records if r["status"] == "complete"]
        shallow = [r for r in completed if r["depth"] == 1]
        deeper = [r for r in completed if r["depth"] > 1]
        saturated = any(r["shallow_ceiling_reached_all_seeds"] for r in shallow)
        best_shallow = max((r["mean_validation_score"] for r in shallow), default=None)
        best_deeper = max((r["mean_validation_score"] for r in deeper), default=None)
        gain = best_deeper - best_shallow if best_shallow is not None and best_deeper is not None else None
        complete = expected_architectures > 0 and len(completed) == expected_architectures
        complexity_sensitive = gain is not None and gain >= min_complexity_gain
        eligible = complete and not saturated and complexity_sensitive
        record = {
            "provider": task_key[0], "suite": task_key[1], "task_id": task_key[2],
            "repeat": fold_key[3], "fold": fold_key[4], "sample": fold_key[5],
            "task_kind": kind, "expected_architectures": expected_architectures,
            "completed_architectures": len(completed), "architectures": records,
            "saturated_by_shallow_mlp": saturated,
            "best_shallow_validation_score": best_shallow,
            "best_deeper_validation_score": best_deeper,
            "validation_complexity_gain": gain,
            "min_complexity_gain": min_complexity_gain,
            "screening_complete": complete,
            "eligible_for_full_comparison": eligible,
            "decision": "include_fold" if eligible else "report_only_excluded_or_incomplete",
        }
        fold_records.append(record)
        by_task[task_key].append(record)

    dataset_records = []
    for task_key in sorted(expected_splits):
        folds = by_task.get(task_key, [])
        expected = len(expected_splits[task_key])
        eligible_count = sum(bool(record["eligible_for_full_comparison"]) for record in folds)
        fraction = eligible_count / expected if expected else 0.0
        dataset_records.append({
            "provider": task_key[0], "suite": task_key[1], "task_id": task_key[2],
            "expected_folds": expected, "screened_folds": len(folds),
            "complexity_sensitive_folds": eligible_count,
            "complexity_sensitive_fraction": fraction,
            "min_hard_fold_fraction": min_hard_fold_fraction,
            "eligible_for_full_comparison": fraction >= min_hard_fold_fraction,
            "decision": "include_dataset" if fraction >= min_hard_fold_fraction else "report_only_excluded_or_incomplete",
        })

    output_root = Path(results_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for filename, records in (("screening-folds.jsonl", fold_records), ("screening-datasets.jsonl", dataset_records)):
        with (output_root / filename).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    summary = {
        "screening_routes": sum(bool(spec.route.screening_only) for spec in runs),
        "folds_reported": len(fold_records),
        "datasets_reported": len(dataset_records),
        "eligible_folds": sum(bool(record["eligible_for_full_comparison"]) for record in fold_records),
        "eligible_datasets": sum(bool(record["eligible_for_full_comparison"]) for record in dataset_records),
        "routes_filtered": False,
        "metrics_source": "validation_only",
    }
    (output_root / "screening-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def expected_splits_fold_keys(runs: Iterable[Any]) -> set[tuple[Any, ...]]:
    return {
        (spec.identity.provider, spec.identity.suite, spec.identity.task_id,
         spec.identity.split_repeat, spec.identity.fold, spec.identity.sample)
        for spec in runs
    }
