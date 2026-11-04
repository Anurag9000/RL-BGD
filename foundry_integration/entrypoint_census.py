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

    # A script-import census alone cannot discover callable training surfaces
    # defined in a runner module but not imported by any standalone script.
    # Read top-level definitions directly; do not import or execute modules.
    runner_modules: list[dict[str, Any]] = []
    defined_targets: set[str] = set()
    independent_candidates: list[dict[str, Any]] = []
    for module_path in sorted((root / "src" / "rl_bgd" / "runners").glob("*.py")):
        module_name = f"rl_bgd.runners.{module_path.stem}"
        syntax = ast.parse(
            module_path.read_text(encoding="utf-8"), filename=str(module_path)
        )
        public_callables: list[dict[str, Any]] = []
        for node in syntax.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            target = f"{module_name}:{node.name}"
            defined_targets.add(target)
            registered = target in declared
            imported = target in exact_imports or any(
                item["module"] == module_name and item["symbol"] == node.name
                for item in candidates
            )
            looks_like_training_entrypoint = node.name.startswith("run_")
            classification = (
                "REGISTERED_SUITE_TARGET"
                if registered
                else "SCRIPT_IMPORTED_UNREGISTERED"
                if imported
                else "RUN_PREFIX_UNREGISTERED"
                if looks_like_training_entrypoint
                else "OTHER_PUBLIC_CALLABLE"
            )
            record: dict[str, Any] = {
                "target": target,
                "name": node.name,
                "line": node.lineno,
                "classification": classification,
                "script_imported": imported,
            }
            public_callables.append(record)
            if classification in {
                "SCRIPT_IMPORTED_UNREGISTERED",
                "RUN_PREFIX_UNREGISTERED",
            }:
                independent_candidates.append({
                    "path": module_path.relative_to(root).as_posix(),
                    **record,
                })
        runner_modules.append({
            "path": module_path.relative_to(root).as_posix(),
            "module": module_name,
            "public_callables": public_callables,
        })
    runner_classifications = Counter(
        item["classification"]
        for module in runner_modules
        for item in module["public_callables"]
    )
    runner_suite_targets = {
        target for target in declared if target.startswith("rl_bgd.runners.")
    }
    missing_definitions = sorted(runner_suite_targets - defined_targets)
    non_runner_targets = sorted(declared - runner_suite_targets)
    return {
        "schema_version": 2,
        "runner_modules_scanned": len(runner_modules),
        "runner_public_callable_count": sum(
            len(module["public_callables"]) for module in runner_modules
        ),
        "runner_callable_classifications": dict(sorted(runner_classifications.items())),
        "suite_targets_absent_from_public_runner_definitions": missing_definitions,
        "registered_non_runner_targets": non_runner_targets,
        "registered_non_runner_target_count": len(non_runner_targets),
        "independent_runner_candidates": independent_candidates,
        "runner_modules": runner_modules,
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
            "public runner callables and run_-prefixed definitions are static candidates, not executable configurations",
            "suite targets may be dynamically exported and are not necessarily local function definitions",
            "runtime argument choices and dynamic invocation paths are not enumerated",
            "suite targets absent from standalone scripts may still be directly runnable",
            "no external data, benchmark or training execution was performed",
        ],
    }
