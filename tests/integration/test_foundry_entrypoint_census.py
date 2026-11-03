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


def test_unregistered_runner_import_is_candidate_not_automatically_scheduled(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_new.py").write_text(
        "from rl_bgd.runners.unregistered import train\n"
        "if __name__ == '__main__':\n    train()\n",
        encoding="utf-8",
    )
    report = entrypoint_census(tmp_path)
    assert report["first_party_scripts"] == 1
    assert report["unmatched_import_candidates"] == [{
        "path": "scripts/run_new.py",
        "module": "rl_bgd.runners.unregistered",
        "symbol": "train",
        "suite_coverage": "NO_SUITE_RUNNER_MODULE",
    }]
