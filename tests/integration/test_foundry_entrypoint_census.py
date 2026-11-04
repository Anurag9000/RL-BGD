"""Static native CLI coverage: do not infer executable jobs from imports."""

from __future__ import annotations

import json
from pathlib import Path

from foundry_integration.entrypoint_census import entrypoint_census

ROOT = Path(__file__).resolve().parents[2]


def test_all_present_native_scripts_appear_once_in_static_census() -> None:
    report = entrypoint_census(ROOT)
    scripts = report["scripts"]
    paths = {item["path"] for item in scripts}
    assert len(scripts) == len(paths)
    assert report["first_party_scripts"] == len(list((ROOT / "scripts").glob("*.py")))
    assert "scripts/run_bgd_sac_lqr.py" in paths
    assert "scripts/run_paper_suite.py" in paths
    assert report["registered_suite_targets"] > 0
    assert report["suite_only_target_count"] == len(report["suite_only_targets"])
    assert report["suite_only_target_count"] > 0
    assert report["limitations"]
    json.dumps(report, allow_nan=False)


def test_exact_suite_symbol_and_module_only_are_distinguished() -> None:
    report = entrypoint_census(ROOT)
    indexed = {item["path"]: item for item in report["scripts"]}
    sac = indexed["scripts/run_bgd_sac_lqr.py"]
    assert sac["main_guard"]
    assert any(
        item["module"] == "rl_bgd.runners.bgd_sac_lqr"
        and item["symbol"] == "run_bgd_sac_lqr"
        and item["suite_coverage"] == "EXACT_SUITE_TARGET"
        for item in sac["runner_imports"]
    )
    assert indexed["scripts/run_paper_suite.py"]["role"] == "other_script"


def test_runner_definition_census_is_bidirectional_and_source_grounded() -> None:
    report = entrypoint_census(ROOT)
    runner_modules = report["runner_modules"]
    assert report["runner_modules_scanned"] == len(
        list((ROOT / "src" / "rl_bgd" / "runners").glob("*.py"))
    )
    assert len(runner_modules) == report["runner_modules_scanned"]
    assert sum(len(record["public_callables"]) for record in runner_modules) == (
        report["runner_public_callable_count"]
    )
    registered = {
        item["target"]
        for module in runner_modules
        for item in module["public_callables"]
        if item["classification"] == "REGISTERED_SUITE_TARGET"
    }
    assert "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr" in registered
    assert all(
        item["classification"] != "REGISTERED_SUITE_TARGET"
        for item in report["independent_runner_candidates"]
    )
    assert not report["suite_targets_absent_from_public_runner_definitions"]
    assert "rl_bgd.analysis.mechanistic:run_seeded_mechanistic_analysis" in (
        report["registered_non_runner_targets"]
    )
    assert report["registered_non_runner_target_count"] == len(
        report["registered_non_runner_targets"]
    )
    json.dumps(report, allow_nan=False)


def test_unregistered_script_import_and_runner_definition_are_reconciled(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    runners = tmp_path / "src" / "rl_bgd" / "runners"
    scripts.mkdir(parents=True)
    runners.mkdir(parents=True)
    (scripts / "example.py").write_text(
        "from rl_bgd.runners.custom import run_custom\n"
        "if __name__ == '__main__':\n"
        "    run_custom()\n",
        encoding="utf-8",
    )
    (runners / "custom.py").write_text(
        "def run_custom():\n"
        "    return 1\n"
        "def run_hidden():\n"
        "    return 2\n"
        "def helper():\n"
        "    return 3\n"
        "def _internal():\n"
        "    return 4\n",
        encoding="utf-8",
    )
    report = entrypoint_census(tmp_path)
    candidates = {
        item["name"]: item for item in report["independent_runner_candidates"]
    }
    assert candidates["run_custom"]["classification"] == "SCRIPT_IMPORTED_UNREGISTERED"
    assert candidates["run_custom"]["script_imported"]
    assert candidates["run_hidden"]["classification"] == "RUN_PREFIX_UNREGISTERED"
    assert not candidates["run_hidden"]["script_imported"]
    assert set(candidates) == {"run_custom", "run_hidden"}
    callables = report["runner_modules"][0]["public_callables"]
    assert {item["name"] for item in callables} == {
        "run_custom", "run_hidden", "helper"
    }
    assert report["schema_version"] == 2


def test_unregistered_runner_import_is_candidate_not_automatically_scheduled(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_new.py").write_text(
        "from rl_bgd.runners.unregistered import train\nif __name__ == '__main__':\n    train()\n",
        encoding="utf-8",
    )
    report = entrypoint_census(tmp_path)
    assert report["first_party_scripts"] == 1
    assert report["unmatched_import_candidates"] == [
        {
            "path": "scripts/run_new.py",
            "module": "rl_bgd.runners.unregistered",
            "symbol": "train",
            "suite_coverage": "NO_SUITE_RUNNER_MODULE",
        }
    ]
