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
    "paper/METHOD.md",
    "paper/LIMITATIONS.md",
    "src/rl_bgd/artifacts/run.py",
    "src/rl_bgd/artifacts/suite.py",
    "src/rl_bgd/analysis/mechanistic.py",
    "src/rl_bgd/analysis/artifacts.py",
    "src/rl_bgd/analysis/paper_artifacts.py",
    "src/rl_bgd/experiments/suites.py",
    "scripts/run_paper_suite.py",
    "scripts/build_paper_artifacts.py",
    "scripts/run_mechanistic_analysis.py",
    ".github/workflows/typecheck.yml",
    ".github/workflows/paper-pipeline-smoke.yml",
)

_FORBIDDEN_SOURCE_MARKERS = (
    "TO" + "DO",
    "FIX" + "ME",
    "Not" + "ImplementedError",
    ".cu" + "da(",
)


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
    candidate = (
        Path(start).resolve()
        if start is not None
        else Path.cwd().resolve()
    )
    for path in (candidate, *candidate.parents):
        if (
            (path / "pyproject.toml").is_file()
            and (path / "src" / "rl_bgd").is_dir()
        ):
            return path
    raise FileNotFoundError(
        "could not locate RL-BGD repository root"
    )


def _python_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for relative in ("src", "scripts"):
        base = root / relative
        files.extend(
            path
            for path in base.rglob("*.py")
            if "__pycache__" not in path.parts
        )
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
        duplicates = sorted(
            name
            for name, count in Counter(
                top_level_names
            ).items()
            if count > 1
        )
        if duplicates:
            findings.append(
                AuditFinding(
                    "duplicate_top_level_definition",
                    relative,
                    "duplicate definitions: "
                    + ", ".join(
                        duplicates
                    ),
                )
            )

    compatibility_artifacts = (
        root
        / "src"
        / "rl_bgd"
        / "analysis"
        / "artifacts.py"
    ).read_text(encoding="utf-8")
    paper_artifacts = (
        root
        / "src"
        / "rl_bgd"
        / "analysis"
        / "paper_artifacts.py"
    ).read_text(encoding="utf-8")
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

    readme = (root / "README.md").read_text(
        encoding="utf-8"
    )
    checks_run += 1
    if "RL agents and benchmark adapters remain explicitly tracked as incomplete" in readme:
        findings.append(
            AuditFinding(
                "stale_status_text",
                "README.md",
                "README still describes the initial pre-RL repository state",
            )
        )

    ledger = (
        root
        / "docs"
        / "CAPABILITY_LEDGER.md"
    ).read_text(encoding="utf-8")
    for phrase in (
        "Recurrent BGD-PPO | COMPLETE",
        "Recurrent BGD-SAC | COMPLETE",
        "Mechanistic experiments |",
        "Paper tables/figures |",
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

    return RepositoryAuditReport(
        root=str(root),
        checks_run=checks_run,
        findings=tuple(findings),
    )
