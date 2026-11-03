"""Complete tracked-file source census; static evidence is not runtime proof.

The census reads every tracked byte, parses Python without importing it, and
distinguishes source declarations from executed and scientifically tested code.
No benchmark, simulator or training job is constructed by this module.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

_MARKDOWN_LINK = re.compile(r"\]\(([^)]+)\)")
_TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".json", ".txt", ".sh", ".ps1"}


def _git_output(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True
    )
    return result.stdout


def tracked_paths(root: Path) -> tuple[Path, ...]:
    """Use the index as source of truth; include all pinned vendor files."""
    paths: list[Path] = []
    for raw_path in _git_output(root, "ls-files", "-z").split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8", errors="strict"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe tracked path: {relative}")
        target = root / relative
        if not target.is_file():
            raise FileNotFoundError(f"tracked file missing from worktree: {relative}")
        paths.append(relative)
    if len(paths) != len(set(paths)):
        raise ValueError("Git index contains duplicate tracked paths")
    return tuple(sorted(paths))


def _category(relative: Path) -> str:
    parts = relative.parts
    if parts[0] == "vendor":
        return "vendor"
    if parts[0] == "tests":
        return "test"
    if parts[0] == "src":
        return "native_source"
    if parts[0] in {"scripts", "foundry_integration"} or relative.name == "run_all_training.py":
        return "entrypoint_or_bridge"
    if parts[0] == "docs" or relative.suffix == ".md":
        return "documentation"
    if parts[0] == "configs" or parts[0] == ".github":
        return "configuration"
    return "other"


def _python_evidence(text: str, relative: str) -> dict[str, object]:
    try:
        module = ast.parse(text, filename=relative)
    except SyntaxError as exc:
        return {"syntax": "DEFECT", "error": str(exc)}
    declarations: list[dict[str, object]] = []
    imports: set[str] = set()
    called_symbols: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def _visit_definition(self, node: ast.AST, name: str, kind: str) -> None:
            self.stack.append(name)
            declarations.append({
                "qualified_name": ".".join(self.stack),
                "kind": kind,
                "line": getattr(node, "lineno", 0),
                "end_line": getattr(node, "end_lineno", 0),
            })
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_definition(node, node.name, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_definition(node, node.name, "async_function")

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._visit_definition(node, node.name, "class")

        def visit_Import(self, node: ast.Import) -> None:
            imports.update(alias.name for alias in node.names)
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module_name = node.module or ""
            imports.update(
                f"{'.' * node.level}{module_name}.{alias.name}" for alias in node.names
            )
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            called_symbols.add(ast.unparse(node.func))
            self.generic_visit(node)

    Visitor().visit(module)
    is_cli = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(isinstance(item, ast.Constant) and item.value == "__main__"
                for item in node.test.comparators)
        for node in module.body
    )
    return {
        "syntax": "VALID",
        "declarations": declarations,
        "imports": sorted(imports),
        "syntactic_call_symbols": sorted(called_symbols),
        "module_entrypoint": is_cli,
        "evidence_level": "STATIC_ONLY",
    }


def _markdown_evidence(root: Path, relative: Path, text: str) -> dict[str, object]:
    broken: list[str] = []
    for token in _MARKDOWN_LINK.findall(text):
        href = token.split(" ", 1)[0].split("#", 1)[0]
        if not href or href.startswith(("http:", "https:", "mailto:", "data:", "/")):
            continue
        target = (root / relative.parent / href).resolve()
        if not target.is_relative_to(root.resolve()):
            broken.append(f"OUTSIDE_ROOT:{token}")
        elif not target.exists():
            broken.append(token)
    return {
        "unresolved_relative_link_candidates": sorted(set(broken)),
        "link_evidence_level": "HEURISTIC_UNVERIFIED",
    }


def scan_tracked_repository(root: Path) -> dict[str, Any]:
    """Read all tracked files, hash content, and index static declarations."""
    root = root.resolve()
    records: list[dict[str, object]] = []
    problems: list[dict[str, str]] = []
    for relative in tracked_paths(root):
        target = root / relative
        raw = target.read_bytes()
        name = relative.as_posix()
        record: dict[str, object] = {
            "path": name,
            "category": _category(relative),
            "bytes": len(raw),
            "line_count": len(raw.splitlines()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        text: str | None = None
        if relative.suffix.lower() in _TEXT_SUFFIXES or relative.name.startswith(".git"):
            try:
                text = raw.decode("utf-8", errors="strict")
                record["utf8"] = "VALID"
            except UnicodeDecodeError as exc:
                record["utf8"] = "INVALID"
                problems.append({
                    "path": name, "kind": "non_utf8_source", "detail": str(exc)
                })
        if text is not None and relative.suffix == ".py":
            python = _python_evidence(text, name)
            record["python"] = python
            if python["syntax"] != "VALID":
                problems.append({
                    "path": name,
                    "kind": "python_syntax",
                    "detail": str(python["error"]),
                })
        if text is not None and relative.suffix == ".md":
            record["markdown"] = _markdown_evidence(root, relative, text)
        if text is not None and relative.suffix == ".json":
            try:
                document = json.loads(text)
                record["top_level_config_keys"] = (
                    sorted(str(key) for key in document) if isinstance(document, dict) else []
                )
            except json.JSONDecodeError as exc:
                problems.append({
                    "path": name, "kind": "invalid_json", "detail": str(exc)
                })
        if text is not None and relative.suffix == ".toml":
            try:
                record["top_level_config_keys"] = sorted(tomllib.loads(text))
            except tomllib.TOMLDecodeError as exc:
                problems.append({
                    "path": name, "kind": "invalid_toml", "detail": str(exc)
                })
        if text is not None and relative.suffix in {".yaml", ".yml"}:
            try:
                document = yaml.safe_load(text)
                record["top_level_config_keys"] = (
                    sorted(str(key) for key in document) if isinstance(document, dict) else []
                )
            except yaml.YAMLError as exc:
                problems.append({
                    "path": name, "kind": "invalid_yaml", "detail": str(exc)
                })
        records.append(record)
    try:
        revision = _git_output(root, "rev-parse", "HEAD").decode().strip()
    except subprocess.CalledProcessError:
        revision = None
    return {
        "schema_version": 1,
        "repository": root.name,
        "git_revision": revision,
        "tracked_file_count": len(records),
        "by_category": dict(sorted(Counter(r["category"] for r in records).items())),
        "by_extension": dict(sorted(Counter(
            Path(str(r["path"])).suffix or "<none>" for r in records
        ).items())),
        "problems": problems,
        "records": records,
        "evidence_level": "TRACKED_FILE_READ_AND_STATIC_PARSE_ONLY",
        "limitations": [
            "symbol calls are syntactic; dynamic dispatch and reachability are unverified",
            "source presence and successful parsing do not prove method correctness",
            "heuristic Markdown links may include examples or generated resources",
            "full benchmark execution and external dependencies are not validated",
        ],
    }
