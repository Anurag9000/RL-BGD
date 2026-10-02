"""Bridge curated Phase-14 suite jobs into strict Phase-15 run artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from rl_bgd.artifacts.run import (
    RunManifest,
    RunSummary,
    metrics_rows_from_result,
    summarize_runner_result,
    write_run_artifacts,
)


def information_access_for_protocol(
    protocol: str,
) -> dict[str, bool]:
    """Encode only information-access facts implied by registered protocols."""

    normalized = protocol.lower()
    task_aware = "task_aware" in normalized or "canonical" in normalized
    oracle_boundary = "oracle_boundary" in normalized
    task_agnostic = (
        "task_agnostic" in normalized or "3rl-style" in normalized or "hidden" in normalized
    )

    if task_aware and task_agnostic:
        raise ValueError(f"ambiguous information-access protocol: {protocol}")
    return {
        "receives_task_id": task_aware,
        "receives_task_boundary": (task_aware or oracle_boundary),
        "receives_environment_context": False,
    }


def information_access_for_result(
    result: Mapping[str, object],
    *,
    protocol: str,
) -> dict[str, bool]:
    """Merge runner-emitted access facts with protocol invariants.

    Explicit runner fields may add detail, but they may not contradict the
    information-access guarantees encoded by the registered protocol.
    """

    derived = information_access_for_protocol(protocol)
    raw = result.get(
        "information_access",
        {},
    )
    if not isinstance(raw, Mapping):
        raise TypeError("runner information_access must be a mapping when present")
    explicit = {str(key): value for key, value in raw.items() if isinstance(value, bool)}
    for key, expected in derived.items():
        if key in explicit and explicit[key] != expected:
            raise ValueError(
                f"runner information-access metadata contradicts protocol {protocol!r}: {key}"
            )
    return {
        **derived,
        **explicit,
    }


def _source_config(
    config_path: object,
) -> tuple[
    str | None,
    dict[str, object],
]:
    if config_path is None:
        return None, {}
    declared_path = Path(str(config_path))
    path = (
        declared_path
        if declared_path.is_absolute()
        else (Path(__file__).resolve().parents[3] / declared_path)
    )
    if not path.is_file():
        raise FileNotFoundError(f"suite source config does not exist: {declared_path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (
        OSError,
        yaml.YAMLError,
    ) as exc:
        raise ValueError(f"cannot read suite source config: {path}") from exc
    if not isinstance(
        payload,
        Mapping,
    ):
        raise ValueError("suite source config must contain a mapping")
    return (
        str(declared_path),
        {str(key): value for key, value in payload.items()},
    )


def _task_order(
    result: Mapping[str, object],
) -> tuple[str, ...]:
    value = result.get("task_names")
    if isinstance(
        value,
        (str, bytes),
    ) or not isinstance(
        value,
        Sequence,
    ):
        return ()
    names = tuple(str(item) for item in value)
    if any(not name.strip() for name in names):
        raise ValueError("runner task_names contains an empty task")
    return names


def parse_runner_stdout(
    stdout: str,
) -> dict[str, object]:
    """Require one JSON object from a successful suite subprocess."""

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("successful suite runner did not emit valid JSON") from exc
    if not isinstance(
        payload,
        Mapping,
    ):
        raise ValueError("successful suite runner JSON must be an object")
    return {str(key): value for key, value in payload.items()}


def _resolve_primary_metric(
    summary: RunSummary,
    metric: str,
) -> float:
    if metric in summary.metrics:
        return float(summary.metrics[metric])
    if metric in summary.resources:
        return float(summary.resources[metric])

    suffix = f".{metric}"
    matches = [float(value) for key, value in summary.metrics.items() if key.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"declared primary metric {metric!r} is absent from run result")
    raise ValueError(
        f"declared primary metric {metric!r} is ambiguous across {len(matches)} result paths"
    )


def _contract_kwargs(
    job: Mapping[str, Any],
) -> dict[str, object]:
    """Return scientific invocation kwargs with per-run fields removed."""

    raw = job.get(
        "kwargs",
        {},
    )
    if not isinstance(
        raw,
        Mapping,
    ):
        raise TypeError(
            "suite job kwargs must be a mapping"
        )
    normalized: dict[
        str,
        object,
    ] = {}
    for key, value in raw.items():
        if not isinstance(
            key,
            str,
        ):
            raise TypeError(
                "suite job kwarg names must be strings"
            )
        normalized[
            key
        ] = value

    for field_name in (
        "seed_kwarg",
        "output_kwarg",
    ):
        field = job.get(
            field_name
        )
        if isinstance(
            field,
            str,
        ) and field:
            normalized.pop(
                field,
                None,
            )
    return normalized


def record_completed_suite_run(
    run_dir: str | Path,
    *,
    suite_name: str,
    git_commit: str,
    job: Mapping[str, Any],
    result: Mapping[str, object],
    duration_seconds: float,
) -> RunManifest:
    """Write a completed suite invocation in the canonical raw-run schema."""

    if not git_commit.strip():
        raise ValueError("completed paper run requires a git commit")
    run_id = str(job["run_id"])
    protocol = str(job["protocol"])
    source_path, source_payload = _source_config(job.get("config_path"))
    manifest = RunManifest(
        run_id=run_id,
        method=str(job["algorithm"]),
        setting=protocol,
        benchmark=str(job["environment"]),
        seed=int(job["seed"]),
        git_commit=git_commit,
        status="completed",
        task_order=_task_order(result),
        information_access=information_access_for_result(
            result,
            protocol=protocol,
        ),
        metadata={
            "suite": suite_name,
            "job_id": str(job["job_id"]),
            "hypothesis_id": str(job["hypothesis_id"]),
            "target": str(job["target"]),
            "primary_metric": str(job["primary_metric"]),
            "secondary_metrics": list(
                job.get(
                    "secondary_metrics",
                    (),
                )
            ),
            "source_config_path": (source_path),
            "contract_kwargs": _contract_kwargs(
                job
            ),
        },
    )
    summary = summarize_runner_result(
        run_id,
        result,
        duration_seconds=duration_seconds,
    )
    primary_metric = str(job["primary_metric"])
    primary_value = _resolve_primary_metric(
        summary,
        primary_metric,
    )
    if primary_metric not in summary.metrics and primary_metric not in summary.resources:
        summary = RunSummary(
            run_id=summary.run_id,
            metrics={
                **summary.metrics,
                primary_metric: primary_value,
            },
            task_metrics=summary.task_metrics,
            resources=summary.resources,
            metadata=summary.metadata,
        )
    resolved_config = {
        "suite": suite_name,
        "run_id": run_id,
        "git_commit": git_commit,
        "invocation": {
            "target": str(job["target"]),
            "kwargs": dict(job["kwargs"]),
        },
        "source_config_path": source_path,
        "source_config": source_payload,
    }
    write_run_artifacts(
        run_dir,
        manifest=manifest,
        summary=summary,
        resolved_config=resolved_config,
        metrics_rows=(metrics_rows_from_result(result)),
    )
    return manifest


def record_failed_suite_run(
    run_dir: str | Path,
    *,
    suite_name: str,
    git_commit: str,
    job: Mapping[str, Any],
    failure_reason: str,
) -> RunManifest:
    """Record failed/invalid suite jobs so aggregation cannot silently omit them."""

    run_id = str(job["run_id"])
    protocol = str(job["protocol"])
    manifest = RunManifest(
        run_id=run_id,
        method=str(job["algorithm"]),
        setting=protocol,
        benchmark=str(job["environment"]),
        seed=int(job["seed"]),
        git_commit=(git_commit if git_commit.strip() else "unresolved"),
        status="failed",
        information_access=(information_access_for_protocol(protocol)),
        metadata={
            "suite": suite_name,
            "job_id": str(job["job_id"]),
            "hypothesis_id": str(job["hypothesis_id"]),
            "target": str(job["target"]),
            "contract_kwargs": _contract_kwargs(
                job
            ),
            "failure_reason": (failure_reason),
        },
    )
    root = Path(run_dir)
    root.mkdir(
        parents=True,
        exist_ok=True,
    )
    (root / "manifest.json").write_text(
        json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
