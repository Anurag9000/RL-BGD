"""Regenerate the AST inventory in the coordinated runner source map."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

MARKER = "## Generic function and method inventory\n"


def _function_rows(path: Path) -> list[tuple[str, int, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rows: list[tuple[str, int, int]] = []

    def visit(nodes: list[ast.stmt], prefix: str) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                rows.append((f"{prefix}{node.name}", node.lineno, node.end_lineno or node.lineno))
                visit(node.body, f"{prefix}{node.name}.<locals>.")

    visit(tree.body, "")
    return rows


def render(repo_root: Path) -> str:
    runner_root = repo_root / "src" / "simple_adp" / "runner"
    lines = [
        MARKER,
        "The inventory below is generated from the current Python AST. Line spans refer to\n",
        "the current checkout and should be refreshed after code movement. It covers each\n",
        "function, method, and nested callback in `src/simple_adp/runner/`; excluded copied-source behavior\n",
        "is identified in the mapping above rather than silently attributed to a generic\n",
        "function.\n\n",
        "Regenerate or verify it with `python scripts/generate_runner_function_map.py` or\n",
        "`python scripts/generate_runner_function_map.py --check`.\n\n",
        "| Module | Function or method | Current lines |\n",
        "|---|---|---:|\n",
    ]
    for path in sorted(runner_root.glob("*.py")):
        entries = _function_rows(path)
        if not entries:
            lines.append(f"| `{path.relative_to(repo_root).as_posix()}` | *(no function definitions)* | — |\n")
            continue
        for name, start, end in entries:
            lines.append(
                f"| `{path.relative_to(repo_root).as_posix()}` | `{name}` | {start}-{end} |\n"
            )
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in map is stale")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "docs" / "COORDINATED_PRESSURE_AWARE_SOURCE_FUNCTION_MAP.md"
    content = target.read_text(encoding="utf-8")
    prefix, marker, _old_inventory = content.partition(MARKER)
    if not marker:
        parser.error(f"inventory marker is missing from {target}")
    updated = prefix + render(repo_root)
    if args.check:
        if updated != content:
            print(f"stale runner function inventory: {target}")
            return 1
        print("runner function inventory is current")
        return 0
    target.write_text(updated, encoding="utf-8")
    print(f"updated {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
