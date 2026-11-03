from pathlib import Path

from rl_bgd.audit.repository import _python_files, audit_repository


def test_repository_closure_audit_passes_current_tree() -> None:
    root = Path(__file__).resolve().parents[2]
    report = audit_repository(root)
    assert report.checks_run > 25
    assert report.findings == ()
    assert report.passed is True


def test_audit_covers_first_party_controller_and_foundry_sources() -> None:
    root = Path(__file__).resolve().parents[2]
    report = audit_repository(root)
    files = set(report.files_inspected)
    for path in (
        "run_all_training.py",
        "foundry_integration/catalog.py",
        "foundry_integration/scientific_inventory.py",
        "foundry_integration/rl_bgd_single.py",
        "tests/audit/test_repository_audit.py",
    ):
        assert path in files
    assert report.source_modules_covered >= 130
    assert report.tests_discovered >= 114
    assert all(not name.startswith("vendor/") for name in files)
    mapping = report.to_dict()
    assert mapping["files_inspected"] == list(report.files_inspected)
    assert mapping["source_modules_covered"] == report.source_modules_covered
    assert mapping["tests_discovered"] == report.tests_discovered


def test_audit_python_inventory_does_not_scan_pinned_vendor(
    tmp_path: Path,
) -> None:
    for path in (
        "run_all_training.py",
        "src/rl_bgd/demo.py",
        "scripts/launch.py",
        "foundry_integration/bridge.py",
        "tests/test_bridge.py",
        "vendor/project_foundry/src/a.py",
    ):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("pass\n", encoding="utf-8")
    paths = {str(path.relative_to(tmp_path)) for path in _python_files(tmp_path)}
    assert paths == {
        "run_all_training.py",
        "src/rl_bgd/demo.py",
        "scripts/launch.py",
        "foundry_integration/bridge.py",
        "tests/test_bridge.py",
    }
