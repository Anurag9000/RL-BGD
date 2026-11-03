"""Catalog all declared RL-BGD scientific jobs without suite-name exclusions."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from rl_bgd.experiments.suites import SUITES, _resolved_call_kwargs


def category(suite: str, algorithm: str) -> str:
    if suite == "smoke":
        return "smoke"
    if suite == "dev":
        return "development_training"
    if suite == "mechanism_analysis":
        return "mechanistic_experiment"
    if suite == "compute_analysis":
        return "compute_training"
    if suite in ("ablation_core", "uncertainty_analysis"):
        return "ablation_training"
    if any(x in algorithm.lower() for x in ("adam", "ewc", "mas", "ucl")):
        return "matched_baseline"
    return "primary_training"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def inventory_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for suite_name, suite in sorted(SUITES.items()):
        for job in suite.jobs:
            kwargs = dict(job.kwargs)
            if job.seed_kwarg:
                kwargs[job.seed_kwarg] = job.seeds[0]
            if job.output_kwarg:
                kwargs[job.output_kwarg] = "__OUTPUT__"
            blocker = None
            try:
                resolved = _resolved_call_kwargs(job.target, kwargs)
            except (ImportError, ModuleNotFoundError) as error:
                resolved = kwargs
                blocker = f"{type(error).__name__}: {error}"
            if job.seed_kwarg:
                resolved.pop(job.seed_kwarg, None)
            if job.output_kwarg:
                resolved.pop(job.output_kwarg, None)
            scientific_contract = {
                "target": job.target,
                "kwargs": resolved,
                "algorithm": job.algorithm,
                "environment": job.environment,
                "protocol": job.protocol,
                "primary_metric": job.primary_metric,
                "secondary_metrics": job.secondary_metrics,
            }
            science = hashlib.sha256(canonical(scientific_contract).encode()).hexdigest()
            for seed in job.seeds:
                identity = f"{suite_name}__{job.job_id}__seed_{seed}"
                records.append({
                    "suite": suite_name, "job_id": job.job_id,
                    "foundry_job_id": f"{suite_name}__{job.job_id}",
                    "native_run_id": identity, "seed": seed,
                    "scientific_fingerprint": science,
                    "execution_fingerprint": hashlib.sha256(
                        canonical([identity, science]).encode()
                    ).hexdigest(),
                    "classification": category(suite_name, job.algorithm),
                    "scientific_contract": scientific_contract,
                    "method": job.algorithm, "dataset": job.environment,
                    "optional_extra": job.optional_extra,
                    "blocker": blocker,
                })
    if len({x["execution_fingerprint"] for x in records}) != len(records):
        raise ValueError("duplicate native execution identities")
    return records


def coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record["scientific_fingerprint"]].add(record["foundry_job_id"])
    return {
        "registry_suites": len(SUITES),
        "seed_expanded_records": len(records),
        "unique_scientific_configurations": len(groups),
        "by_classification": dict(sorted(Counter(
            item["classification"] for item in records
        ).items())),
        "duplicate_aliases": {
            key: sorted(value) for key, value in sorted(groups.items()) if len(value) > 1
        },
        "blocked_records": sum(bool(x["blocker"]) for x in records),
        "inventory_scope": "SUITES registry only; independently reachable runners unverified",
    }
