"""Source-derived Foundry scientific inventory contracts (no research runs)."""

from __future__ import annotations

import csv
import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from foundry_integration.catalog import discover
from foundry_integration.scientific_inventory import (
    coverage_summary,
    inventory_records,
    optional_dependency_blocker,
    training_fingerprint,
)
from run_all_training import main

from rl_bgd.experiments.suites import SUITES

ROOT = Path(__file__).resolve().parents[2]


def test_inventory_covers_every_declared_suite_and_training_category() -> None:
    records = inventory_records()
    assert {item["suite"] for item in records} == set(SUITES)
    assert len(records) == sum(len(job.seeds) for suite in SUITES.values() for job in suite.jobs)
    for name in (
        "ablation_core",
        "uncertainty_analysis",
        "compute_analysis",
        "mechanism_analysis",
        "task_agnostic_final",
    ):
        assert any(record["suite"] == name for record in records)
    summary = coverage_summary(records)
    assert summary["unique_scientific_configurations"] > 0
    assert summary["seed_expanded_records"] == len(records)


def test_oracle_boundary_sac_is_separate_from_task_agnostic_catalog() -> None:
    records = [
        record
        for record in inventory_records()
        if record["suite"] == "baseline_core"
        and record["job_id"].startswith("oracle_boundary_sac_")
    ]
    assert len(records) == 25
    assert len({record["foundry_job_id"] for record in records}) == 5
    assert len({record["scientific_fingerprint"] for record in records}) == 5
    assert {record["scientific_contract"]["protocol"] for record in records} == {"oracle_boundary"}
    assert {record["dataset"] for record in records} == {"oracle_recurring_lqr_regularized_v1"}
    assert {record["scientific_contract"]["kwargs"]["method"] for record in records} == {
        "ewc",
        "online_ewc",
        "si",
        "mas",
    }
    assert all(record["blocker"] is None for record in records)
    sham = [row for row in records if row["job_id"].endswith("_sham")]
    assert len(sham) == 5
    assert all(
        row["scientific_contract"]["kwargs"]["regularization_strength"] == 0.0
        for row in sham
    )
    assert all(
        row["scientific_contract"]["kwargs"]["regularization_strength"] == 1.0
        for row in records if row["job_id"] != "oracle_boundary_sac_ewc_sham"
    )


def test_structured_quadratic_adapter_is_in_foundry_inventory() -> None:
    jobs = [
        row
        for row in inventory_records()
        if row["suite"] == "mechanism_analysis" and row["job_id"] == "continual_quadratic_original"
    ]
    assert len(jobs) == 5
    assert len({item["scientific_fingerprint"] for item in jobs}) == 1
    assert len({item["execution_fingerprint"] for item in jobs}) == 5
    assert all(
        item["scientific_contract"]["target"] == "rl_bgd.runners.quadratic:run_quadratic_suite"
        for item in jobs
    )


def test_matched_stationary_recurrent_sac_adam_bgd_inventory() -> None:
    records = [
        item
        for item in inventory_records()
        if item["suite"] == "stationary_core"
        and item["job_id"].startswith("stationary_recurrent_sac_")
    ]
    assert len(records) == 10
    assert len({item["foundry_job_id"] for item in records}) == 2
    assert len({item["scientific_fingerprint"] for item in records}) == 2
    assert {item["scientific_contract"]["protocol"] for item in records} == {"stationary"}
    assert {item["dataset"] for item in records} == {"recurrent_stationary_lqr_sac_v1"}


def test_registered_stationary_recurrent_ppo_adam_bgd_runs() -> None:
    rows = [
        record
        for record in inventory_records()
        if record["suite"] == "stationary_core"
        and record["job_id"].startswith("stationary_recurrent_ppo_")
    ]
    assert len(rows) == 10
    assert len({row["foundry_job_id"] for row in rows}) == 2
    assert len({row["scientific_fingerprint"] for row in rows}) == 2
    assert {row["scientific_contract"]["protocol"] for row in rows} == {"stationary"}
    assert {row["dataset"] for row in rows} == {"recurrent_stationary_lqr_ppo_v1"}


def test_foundry_registers_both_distinct_matched_ppo_families() -> None:
    rows = [
        record
        for record in inventory_records()
        if record["job_id"].startswith("ta_ppo_") and record["suite"] == "task_agnostic_final"
    ]
    assert len(rows) == 20
    assert len({record["foundry_job_id"] for record in rows}) == 4
    assert len({record["scientific_fingerprint"] for record in rows}) == 4
    assert {record["scientific_contract"]["protocol"] for record in rows} == {
        "strict_task_agnostic"
    }
    assert {record["dataset"] for record in rows} == {
        "recurring_lqr_ppo_feedforward_v1",
        "recurring_lqr_ppo_recurrent_v1",
    }


def test_scientific_identity_is_seed_independent_and_execution_identity_is_not() -> None:
    records = inventory_records()
    a = records[0]
    siblings = [
        item for item in records if item["suite"] == a["suite"] and item["job_id"] == a["job_id"]
    ]
    assert len({item["scientific_fingerprint"] for item in siblings}) == 1
    assert len({item["execution_fingerprint"] for item in siblings}) == len(siblings)
    second = inventory_records()
    assert [(x["native_run_id"], x["scientific_fingerprint"]) for x in records] == [
        (x["native_run_id"], x["scientific_fingerprint"]) for x in second
    ]


def test_catalog_does_not_drop_analysis_named_training_jobs() -> None:
    args = SimpleNamespace(seed=[], data=[], config=[], action="inventory", out="artifacts")
    jobs = discover(ROOT, args)
    assert len(jobs) == len(inventory_records())
    assert any(job["id"].startswith("ablation_core__") for job in jobs)
    assert any(job["id"].startswith("compute_analysis__") for job in jobs)
    assert all(job["scientific_fingerprint"] for job in jobs)
    assert len({(job["id"], job["seed"]) for job in jobs}) == len(jobs)


def test_foundry_gpu_admission_respects_native_cpu_pin() -> None:
    args = SimpleNamespace(seed=[], data=[], config=[], action="inventory", out="artifacts")
    jobs = discover(ROOT, args)
    cpu_pinned = [
        job for job in jobs if job["scientific_contract"]["kwargs"].get("device") == "cpu"
    ]
    assert cpu_pinned, "the native registry must exercise CPU-only admission"
    assert all(not job["gpu_capable"] for job in cpu_pinned)
    assert all(
        job["gpu_capable"]
        for job in jobs
        if job["scientific_contract"]["kwargs"].get("device") != "cpu"
    )
    assert any(
        job["gpu_capable"] and job["scientific_contract"]["kwargs"].get("device") == "auto"
        for job in jobs
    )


def test_rl_bgd_fails_closed_on_unconsumed_external_overrides() -> None:
    args = SimpleNamespace(seed=[], data=[], config=[], action="inventory", out="artifacts")
    args.config = ["configs/custom.yaml"]
    with pytest.raises(ValueError, match="does not support --config"):
        discover(ROOT, args)
    args.config = []
    args.data = ["external_dataset=/tmp/unused"]
    with pytest.raises(ValueError, match="does not consume --data"):
        discover(ROOT, args)


@pytest.mark.parametrize(
    ("selectors", "message"),
    [
        (["--job", "missing"], "unknown native --job"),
        (["--job", "known_a", "--job", "missing"], "unknown native --job"),
        (["--seed", "999"], "unknown native --seed"),
        (["--seed", "0", "--seed", "999"], "unknown native --seed"),
        (["--job", "known_a", "--seed", "1"], "combined --job/--seed"),
    ],
)
def test_native_explicit_selectors_fail_closed(
    selectors: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = [
        {"id": "known_a", "seed": 0, "method": "A", "dataset": "lqr"},
        {"id": "known_b", "seed": 1, "method": "B", "dataset": "lqr"},
    ]
    with (
        patch("foundry_integration.catalog.discover", return_value=catalog),
        pytest.raises(SystemExit) as error,
    ):
        main(["inventory", *selectors])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_native_explicit_known_selection_is_accepted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = [
        {"id": "known_a", "seed": 0, "method": "A", "dataset": "lqr"},
        {"id": "known_b", "seed": 1, "method": "B", "dataset": "lqr"},
    ]
    with patch("foundry_integration.catalog.discover", return_value=catalog):
        assert main(["inventory", "--job", "known_b", "--seed", "1"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["selected"] == 1
    assert output["jobs"][0]["id"] == "known_b"


def test_inventory_export_distinguishes_methods_and_seed_expansions(
    tmp_path: Path,
) -> None:
    assert main(["inventory", "--write-inventory", "--out", str(tmp_path)]) == 0
    full = json.loads((tmp_path / "scientific_job_inventory.json").read_text())
    coverage = json.loads((tmp_path / "scientific_job_coverage.json").read_text())
    with (tmp_path / "scientific_job_inventory.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert full["distinct_scientific_jobs"] == coverage["unique_scientific_configurations"]
    assert len(rows) == coverage["seed_expanded_records"]
    assert len(full["jobs"]) == len(rows)


def test_optional_benchmark_blockers_are_reported_without_loading_simulators() -> None:
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError()):
        assert "CARL 1.1.1 not installed" in optional_dependency_blocker("carl")
    with patch("importlib.util.find_spec", return_value=None):
        assert "metaworld" in optional_dependency_blocker("continual-world")
        assert "mujoco" in optional_dependency_blocker("continual-world")
    assert optional_dependency_blocker(None) is None


def test_scientific_fingerprint_excludes_reporting_and_execution_coordinates() -> None:
    base = {
        "target": "rl_bgd.runners.synthetic:run",
        "kwargs": {"steps": 40, "device": "cpu", "posterior_std": 0.2},
        "algorithm": "BGD",
        "environment": "lqr",
        "protocol": "strict_task_agnostic",
        "primary_metric": "final_return",
        "secondary_metrics": ["sigma_mean"],
    }
    another = {
        **base,
        "kwargs": {**base["kwargs"], "device": "cuda"},
        "primary_metric": "return_auc",
        "secondary_metrics": ["wall_seconds"],
    }
    assert training_fingerprint(base) == training_fingerprint(another)
    changed_method = {
        **base,
        "kwargs": {**base["kwargs"], "posterior_std": 0.4},
    }
    assert training_fingerprint(base) != training_fingerprint(changed_method)
