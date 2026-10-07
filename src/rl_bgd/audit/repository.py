"""Executable final-system audit for RL-BGD."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rl_bgd.experiments.suites import SUITES, validate_suite_registry

_REQUIRED_PATHS = (
    "README.md",
    "pyproject.toml",
    "docs/CAPABILITY_LEDGER.md",
    "docs/EXPERIMENT_REGISTRY.md",
    "docs/paper_suites.md",
    "docs/paper_artifacts.md",
    "docs/mechanistic_analysis.md",
    "docs/experiments.md",
    "docs/literature_review.md",
    "docs/reproducibility.md",
    "paper/FIGURE_PLAN.md",
    "paper/RESULTS_TEMPLATE.md",
    "paper/METHOD.md",
    "paper/LIMITATIONS.md",
    "src/rl_bgd/artifacts/run.py",
    "src/rl_bgd/artifacts/suite.py",
    "src/rl_bgd/analysis/mechanistic.py",
    "src/rl_bgd/compat/cora_chores_download.py",
    "src/rl_bgd/analysis/artifacts.py",
    "src/rl_bgd/analysis/paper_artifacts.py",
    "src/rl_bgd/experiments/suites.py",
    "src/rl_bgd/experiments/parallel.py",
    "src/rl_bgd/compat/cora_chores.py",
    "scripts/run_paper_suite.py",
    "scripts/run_paper_suite_parallel.py",
    "scripts/run_recurrent_stationary_lqr.py",
    "scripts/build_paper_artifacts.py",
    "scripts/run_mechanistic_analysis.py",
    "scripts/cora_legacy_smoke.py",
    "scripts/fetch_cora_chores_archive.py",
    "scripts/cora_legacy_procgen_smoke.py",
    "scripts/cora_legacy_minihack_smoke.py",
    "scripts/cora_legacy_chores_smoke.py",
    "scripts/validate_cora_chores_archive.py",
    ".github/workflows/typecheck.yml",
    ".github/workflows/lint.yml",
    ".github/workflows/tests.yml",
    ".github/workflows/carl.yml",
    ".github/workflows/continual-bench.yml",
    ".github/workflows/cora-legacy.yml",
    ".github/workflows/cora-chores.yml",
    ".github/workflows/recurrent-learning.yml",
    ".github/workflows/paper-pipeline-smoke.yml",
)

_FORBIDDEN_SOURCE_MARKERS = (
    "TO" + "DO",
    "FIX" + "ME",
    ".cu" + "da(",
)

_ALLOWED_BLOCKED_CAPABILITIES = {
    "CORA CHORES/ALFRED runtime": (
        "archive",
        "authoritative",
    ),
    "CORA complete four-family runtime including CHORES": (
        "CHORES",
        "archive",
    ),
}

_EXPECTED_EXPERIMENT_IDS = {
    "SYN-Q1",
    "SYN-Q2",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "GB-T",
    "MC-K",
    "H",
    "CW-CAN10",
    "CW-CAN20",
    "CW-TA10",
    "CW-TA20",
    "I",
    "J",
    "UCL",
}

_EXPERIMENT_SUITE_EVIDENCE = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "GB-T": "GB-T",
    "MC-K": "MC-K",
    "H": "H",
    "CW-CAN10": "CW-CAN10",
    "CW-CAN20": "CW-CAN20",
    "CW-TA10": "CW-TA10",
    "CW-TA20": "CW-TA20",
    "I": "I-J",
    "J": "I-J",
    "UCL": "UCL",
}

_SYNTHETIC_TEST_EVIDENCE = {
    "SYN-Q1": (
        "tests/math/test_bgd_quadratic.py",
        "test_positive_quadratic_curvature_reduces_sigma",
    ),
    "SYN-Q2": (
        "tests/math/test_bgd_quadratic.py",
        "test_curvature_signal_approaches_h_sigma",
    ),
}


@dataclass(frozen=True)
class AuditFinding:
    check: str
    path: str
    detail: str


@dataclass(frozen=True)
class RepositoryAuditReport:
    root: str
    checks_run: int
    findings: tuple[AuditFinding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "checks_run": self.checks_run,
            "passed": self.passed,
            "findings": [
                {
                    "check": item.check,
                    "path": item.path,
                    "detail": item.detail,
                }
                for item in self.findings
            ],
        }


def _repo_root(start: str | Path | None) -> Path:
    candidate = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for path in (candidate, *candidate.parents):
        if (path / "pyproject.toml").is_file() and (path / "src" / "rl_bgd").is_dir():
            return path
    raise FileNotFoundError("could not locate RL-BGD repository root")


def _python_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for relative in ("src", "scripts"):
        base = root / relative
        files.extend(path for path in base.rglob("*.py") if "__pycache__" not in path.parts)
    return tuple(sorted(files))


def audit_repository(
    start: str | Path | None = None,
) -> RepositoryAuditReport:
    """Audit repository structure and fail-closed research invariants."""

    root = _repo_root(start)
    findings: list[AuditFinding] = []
    checks_run = 0

    for relative in _REQUIRED_PATHS:
        checks_run += 1
        if not (root / relative).is_file():
            findings.append(
                AuditFinding(
                    "required_path",
                    relative,
                    "required repository surface is missing",
                )
            )

    validate_suite_registry()
    checks_run += 1
    suite_hypothesis_ids = {job.hypothesis_id for suite in SUITES.values() for job in suite.jobs}
    for suite_name, suite in SUITES.items():
        job_ids = [job.job_id for job in suite.jobs]
        checks_run += 1
        if len(job_ids) != len(set(job_ids)):
            findings.append(
                AuditFinding(
                    "unique_suite_job_ids",
                    "src/rl_bgd/experiments/suites.py",
                    f"suite {suite_name} contains duplicate job IDs",
                )
            )
        for job in suite.jobs:
            checks_run += 1
            if len(job.seeds) != len(set(job.seeds)):
                findings.append(
                    AuditFinding(
                        "unique_seeds",
                        "src/rl_bgd/experiments/suites.py",
                        f"{suite_name}/{job.job_id} repeats seeds",
                    )
                )
            if any(seed < 0 for seed in job.seeds):
                findings.append(
                    AuditFinding(
                        "nonnegative_seeds",
                        "src/rl_bgd/experiments/suites.py",
                        f"{suite_name}/{job.job_id} has a negative seed",
                    )
                )
            if job.config_path is not None:
                config = root / job.config_path
                checks_run += 1
                if not config.is_file():
                    findings.append(
                        AuditFinding(
                            "suite_config_exists",
                            job.config_path,
                            f"referenced by {suite_name}/{job.job_id}",
                        )
                    )

    for path in _python_files(root):
        relative = str(path.relative_to(root))
        text = path.read_text(encoding="utf-8")
        if relative != "src/rl_bgd/audit/repository.py":
            for marker in _FORBIDDEN_SOURCE_MARKERS:
                checks_run += 1
                if marker in text:
                    findings.append(
                        AuditFinding(
                            "forbidden_source_marker",
                            relative,
                            f"contains {marker!r}",
                        )
                    )

        checks_run += 1
        try:
            module = ast.parse(
                text,
                filename=relative,
            )
        except SyntaxError as exc:
            findings.append(
                AuditFinding(
                    "python_syntax",
                    relative,
                    str(exc),
                )
            )
            continue
        top_level_names = [
            node.name
            for node in module.body
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            )
        ]
        duplicates = sorted(name for name, count in Counter(top_level_names).items() if count > 1)
        if duplicates:
            findings.append(
                AuditFinding(
                    "duplicate_top_level_definition",
                    relative,
                    "duplicate definitions: " + ", ".join(duplicates),
                )
            )

        checks_run += 1
        for node in ast.walk(module):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(raised, ast.Name) and raised.id == "NotImplementedError":
                findings.append(
                    AuditFinding(
                        "placeholder_not_implemented",
                        relative,
                        "contains raise NotImplementedError",
                    )
                )
                break

    compatibility_artifacts = (root / "src" / "rl_bgd" / "analysis" / "artifacts.py").read_text(
        encoding="utf-8"
    )
    paper_artifacts = (root / "src" / "rl_bgd" / "analysis" / "paper_artifacts.py").read_text(
        encoding="utf-8"
    )
    checks_run += 1
    if (
        "run_metadata.json" in compatibility_artifacts
        or "stdout.json" in compatibility_artifacts
        or "run_metadata.json" in paper_artifacts
        or "stdout.json" in paper_artifacts
    ):
        findings.append(
            AuditFinding(
                "canonical_artifact_authority",
                "src/rl_bgd/analysis",
                "paper aggregation consumes legacy execution logs",
            )
        )
    checks_run += 1
    if "load_run_directory" not in paper_artifacts:
        findings.append(
            AuditFinding(
                "canonical_artifact_authority",
                "src/rl_bgd/analysis/paper_artifacts.py",
                "canonical paper aggregation does not use strict run loader",
            )
        )
    checks_run += 1
    if "paper_artifacts" not in compatibility_artifacts:
        findings.append(
            AuditFinding(
                "canonical_artifact_authority",
                "src/rl_bgd/analysis/artifacts.py",
                "compatibility API is not delegating to canonical paper artifacts",
            )
        )

    readme = (root / "README.md").read_text(encoding="utf-8")
    checks_run += 1
    if "RL agents and benchmark adapters remain explicitly tracked as incomplete" in readme:
        findings.append(
            AuditFinding(
                "stale_status_text",
                "README.md",
                "README still describes the initial pre-RL repository state",
            )
        )

    ledger = (root / "docs" / "CAPABILITY_LEDGER.md").read_text(encoding="utf-8")
    for phrase in (
        "Recurrent BGD-PPO | COMPLETE",
        "Recurrent BGD-SAC | COMPLETE",
        "Mechanistic experiments | COMPLETE",
        "Paper tables/figures | COMPLETE",
    ):
        checks_run += 1
        if phrase not in ledger:
            findings.append(
                AuditFinding(
                    "ledger_coverage",
                    "docs/CAPABILITY_LEDGER.md",
                    f"missing expected ledger entry containing {phrase!r}",
                )
            )

    allowed_closure_statuses = {
        "COMPLETE",
        "BLOCKED",
    }
    for line in ledger.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] == "Capability" or set(cells[0]) <= {"-"}:
            continue
        checks_run += 1
        status = cells[1]
        if status not in allowed_closure_statuses:
            findings.append(
                AuditFinding(
                    "ledger_software_closure",
                    "docs/CAPABILITY_LEDGER.md",
                    (f"{cells[0]!r} still has non-closure status {status!r}"),
                )
            )
        elif status == "BLOCKED":
            rationale = cells[2]
            if not rationale:
                findings.append(
                    AuditFinding(
                        "ledger_blocked_rationale",
                        "docs/CAPABILITY_LEDGER.md",
                        f"{cells[0]!r} is BLOCKED without a rationale",
                    )
                )
                continue
            required_terms = _ALLOWED_BLOCKED_CAPABILITIES.get(cells[0])
            if required_terms is None:
                findings.append(
                    AuditFinding(
                        "ledger_unapproved_blocker",
                        "docs/CAPABILITY_LEDGER.md",
                        f"{cells[0]!r} is BLOCKED but is not an approved external prerequisite",
                    )
                )
                continue
            missing_terms = [
                term for term in required_terms if term.lower() not in rationale.lower()
            ]
            if missing_terms:
                findings.append(
                    AuditFinding(
                        "ledger_blocked_rationale",
                        "docs/CAPABILITY_LEDGER.md",
                        (
                            f"{cells[0]!r} blocker rationale is missing required "
                            f"evidence terms: {', '.join(missing_terms)}"
                        ),
                    )
                )

    registry = (root / "docs" / "EXPERIMENT_REGISTRY.md").read_text(encoding="utf-8")
    registry_ids: set[str] = set()
    for line in registry.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] == "ID" or set(cells[0]) <= {"-"}:
            continue
        experiment_id = cells[0]
        status = cells[2]
        registry_ids.add(experiment_id)
        checks_run += 1
        if "IMPLEMENTED" not in status.upper():
            findings.append(
                AuditFinding(
                    "experiment_registry_implementation",
                    "docs/EXPERIMENT_REGISTRY.md",
                    (
                        f"{experiment_id!r} lacks an IMPLEMENTED status; "
                        "expensive execution may be pending, but software must be closed"
                    ),
                )
            )

    checks_run += 1
    missing_experiments = sorted(_EXPECTED_EXPERIMENT_IDS - registry_ids)
    if missing_experiments:
        findings.append(
            AuditFinding(
                "experiment_registry_coverage",
                "docs/EXPERIMENT_REGISTRY.md",
                "missing required experiment IDs: " + ", ".join(missing_experiments),
            )
        )

    for experiment_id, suite_hypothesis_id in _EXPERIMENT_SUITE_EVIDENCE.items():
        checks_run += 1
        if suite_hypothesis_id not in suite_hypothesis_ids:
            findings.append(
                AuditFinding(
                    "experiment_suite_evidence",
                    "src/rl_bgd/experiments/suites.py",
                    (
                        f"{experiment_id!r} is marked implemented but no suite job "
                        f"carries hypothesis ID {suite_hypothesis_id!r}"
                    ),
                )
            )

    for experiment_id, (relative, marker) in _SYNTHETIC_TEST_EVIDENCE.items():
        checks_run += 1
        evidence_path = root / relative
        if not evidence_path.is_file():
            findings.append(
                AuditFinding(
                    "experiment_test_evidence",
                    relative,
                    f"{experiment_id!r} requires this math-test evidence file",
                )
            )
            continue
        evidence_text = evidence_path.read_text(encoding="utf-8")
        if marker not in evidence_text:
            findings.append(
                AuditFinding(
                    "experiment_test_evidence",
                    relative,
                    (f"{experiment_id!r} is missing required regression test {marker!r}"),
                )
            )

    chores_workflow_relative = ".github/workflows/cora-chores.yml"
    chores_workflow = (root / chores_workflow_relative).read_text(encoding="utf-8")
    required_chores_workflow_snippets = (
        "workflow_dispatch:",
        "chores_archive_url:",
        'default: "https://onedrive.live.com/download?cid=601D311D0FC404D4',
        "scripts/validate_cora_chores_archive.py",
        "PYTHONPATH: src",
        "--root-only",
        'echo "root=$root" >> "$GITHUB_OUTPUT"',
        "xvfb-run -a python scripts/cora_legacy_chores_smoke.py",
    )
    for snippet in required_chores_workflow_snippets:
        checks_run += 1
        if snippet not in chores_workflow:
            findings.append(
                AuditFinding(
                    "chores_recovery_workflow",
                    chores_workflow_relative,
                    f"missing required recovery contract snippet {snippet!r}",
                )
            )

    forbidden_chores_workflow_snippets = (
        "\n  push:",
        "\n  pull_request:",
        "--expected-trajectories 27",
        "\\ \\",
        '--metadata-root "$metadata_root" \\\n          echo "root=$root"',
    )
    for snippet in forbidden_chores_workflow_snippets:
        checks_run += 1
        if snippet in chores_workflow:
            findings.append(
                AuditFinding(
                    "chores_recovery_workflow",
                    chores_workflow_relative,
                    f"contains forbidden recovery workflow pattern {snippet!r}",
                )
            )

    return RepositoryAuditReport(
        root=str(root),
        checks_run=checks_run,
        findings=tuple(findings),
    )
