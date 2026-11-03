"""Source-derived Foundry scientific inventory contracts (no research runs)."""

from __future__ import annotations

import csv
import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from foundry_integration.catalog import discover
from foundry_integration.scientific_inventory import (
    coverage_summary,
    inventory_records,
    optional_dependency_blocker,
)
from run_all_training import main

from rl_bgd.experiments.suites import SUITES

ROOT = Path(__file__).resolve().parents[2]


def test_inventory_covers_every_declared_suite_and_training_category() -> None:
    records = inventory_records()
    assert {item["suite"] for item in records} == set(SUITES)
    assert len(records) == sum(
        len(job.seeds) for suite in SUITES.values() for job in suite.jobs
    )
    for name in ("ablation_core", "uncertainty_analysis", "compute_analysis",
                 "mechanism_analysis", "task_agnostic_final"):
        assert any(record["suite"] == name for record in records)
    summary = coverage_summary(records)
    assert summary["unique_scientific_configurations"] > 0
    assert summary["seed_expanded_records"] == len(records)


def test_scientific_identity_is_seed_independent_and_execution_identity_is_not() -> None:
    records = inventory_records()
    a = records[0]
    siblings = [
        item for item in records if item["suite"] == a["suite"]
        and item["job_id"] == a["job_id"]
    ]
    assert len({item["scientific_fingerprint"] for item in siblings}) == 1
    assert len({item["execution_fingerprint"] for item in siblings}) == len(siblings)
    second = inventory_records()
    assert [(x["native_run_id"], x["scientific_fingerprint"])
            for x in records] == [(x["native_run_id"], x["scientific_fingerprint"])
                                 for x in second]


def test_catalog_does_not_drop_analysis_named_training_jobs() -> None:
    args = SimpleNamespace(seed=[], data=[], config=[], action="inventory", out="artifacts")
    jobs = discover(ROOT, args)
    assert len(jobs) == len(inventory_records())
    assert any(job["id"].startswith("ablation_core__") for job in jobs)
    assert any(job["id"].startswith("compute_analysis__") for job in jobs)
    assert all(job["scientific_fingerprint"] for job in jobs)
    assert len({(job["id"], job["seed"]) for job in jobs}) == len(jobs)


def test_inventory_export_distinguishes_methods_and_seed_expansions(
    tmp_path: Path,
) -> None:
    assert main(["inventory", "--write-inventory", "--out", str(tmp_path)]) == 0
    full = json.loads((tmp_path / "scientific_job_inventory.json").read_text())
    coverage = json.loads((tmp_path / "scientific_job_coverage.json").read_text())
    with (tmp_path / "scientific_job_inventory.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert full["distinct_scientific_jobs"] == coverage[
        "unique_scientific_configurations"
    ]
    assert len(rows) == coverage["seed_expanded_records"]
    assert len(full["jobs"]) == len(rows)


def test_optional_benchmark_blockers_are_reported_without_loading_simulators() -> None:
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError()):
        assert "CARL 1.1.1 not installed" in optional_dependency_blocker("carl")
    with patch("importlib.util.find_spec", return_value=None):
        assert "metaworld" in optional_dependency_blocker("continual-world")
        assert "mujoco" in optional_dependency_blocker("continual-world")
    assert optional_dependency_blocker(None) is None
