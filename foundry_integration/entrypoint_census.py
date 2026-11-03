"""Read-only census of native script entrypoints and suite-target reachability.

An import relationship is not evidence that a script has been exercised or
that a candidate can be executed in a given environment.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any

from rl_bgd.experiments.suites import SUITES


def _has_main_guard(module: ast.Module) -> bool:
    return any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(
            isinstance(value, ast.Constant) and value.value == "__main__"
            for value in node.test.comparators
        )
        for node in module.body
    )


def entrypoint_census(root: Path) -> dict[str, Any]:
    """Enumerate present first-party scripts without inventing jobs."""
    root = root.resolve()
    declared = {
        job.target for suite in SUITES.values() for job in suite.jobs
    }
    registered_modules = {target.split(":", 1)[0] for target in declared}
    records: list[dict[str, Any]] = []
    for script in sorted((root / "scripts").glob("*.py")):
        module = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        references: list[dict[str, str]] = []
        for node in ast.walk(module):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.startswith("rl_bgd.runners."):
                continue
            for alias in node.names:
                target = f"{node.module}:{alias.name}"
                coverage = (
                    "EXACT_SUITE_TARGET"
                    if target in declared
                    else "SAME_RUNNER_MODULE_ONLY"
                    if node.module in registered_modules
                    else "NO_SUITE_RUNNER_MODULE"
                )
                references.append({
                    "module": node.module,
                    "symbol": alias.name,
                    "suite_coverage": coverage,
                })
        references.sort(key=lambda item: (item["module"], item["symbol"]))
        has_guard = _has_main_guard(module)
        role = (
            "runner_wrapper" if references else
            "other_script"
        )
        records.append({
            "path": script.relative_to(root).as_posix(),
            "main_guard": has_guard,
            "role": role,
            "runner_imports": references,
            "evidence_level": "STATIC_IMPORT_AND_AST_ONLY",
        })

    candidates = [
        {"path": record["path"], **item}
        for record in records
        for item in record["runner_imports"]
        if item["suite_coverage"] != "EXACT_SUITE_TARGET"
    ]
    exact_imports = {
        f'{item["module"]}:{item["symbol"]}'
        for record in records
        for item in record["runner_imports"]
        if item["suite_coverage"] == "EXACT_SUITE_TARGET"
    }
    suite_only_targets = sorted(declared - exact_imports)
    return {
        "schema_version": 1,
        "suite_only_target_count": len(suite_only_targets),
        "suite_only_targets": suite_only_targets,
        "first_party_scripts": len(records),
        "registered_suite_targets": len(declared),
        "by_role": dict(sorted(Counter(x["role"] for x in records).items())),
        "by_suite_coverage": dict(sorted(Counter(
            item["suite_coverage"] for record in records
            for item in record["runner_imports"]
        ).items())),
        "unmatched_import_candidates": candidates,
        "scripts": records,
        "limitations": [
            "same-module and unmatched imports are audit candidates, not proof of missing jobs",
            "runtime argument choices and dynamic invocation paths are not enumerated",
            "suite targets absent from standalone scripts may still be directly runnable",
            "no external data, benchmark or training execution was performed",
        ],
    }
