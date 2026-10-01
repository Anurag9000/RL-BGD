from pathlib import Path

from rl_bgd.audit.repository import audit_repository


def test_repository_closure_audit_passes_current_tree() -> None:
    root = Path(__file__).resolve().parents[2]
    report = audit_repository(root)
    assert report.checks_run > 25
    assert report.findings == ()
    assert report.passed is True
