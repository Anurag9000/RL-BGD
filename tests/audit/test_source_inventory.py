"""Tracked-file source census: content coverage without runtime overclaims."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rl_bgd.audit.source_inventory import scan_tracked_repository, tracked_paths


def _git_index(root: Path, files: dict[str, str]) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)


def test_source_inventory_reads_source_docs_config_and_pinned_vendor(
    tmp_path: Path,
) -> None:
    files = {
        "src/rl_bgd/sample.py": (
            "import math\nclass Model:\n"
            "    def step(self, x):\n        return math.sqrt(x)\n"
            "def main():\n    return Model().step(1)\n"
        ),
        "vendor/project_foundry/engine.py": "def vendor():\n    return True\n",
        "docs/readme.md": "[source](../src/rl_bgd/sample.py)\n[bad](absent.md)\n",
        "configs/evaluation.yaml": "episodes: 3\n",
        "pyproject.toml": '[project]\nname = "fixture"\n',
        "metadata.json": '{"kind": "fixture"}\n',
        "tests/test_sample.py": "def test_sample():\n    assert True\n",
        "scripts/invalid.py": "def broken(:\n    pass\n",
    }
    _git_index(tmp_path, files)
    report = scan_tracked_repository(tmp_path)
    assert report["tracked_file_count"] == len(files)
    assert len(report["records"]) == len(files)
    indexed = {entry["path"]: entry for entry in report["records"]}
    source = indexed["src/rl_bgd/sample.py"]["python"]
    assert source["syntax"] == "VALID"
    assert any(x["qualified_name"] == "Model.step" for x in source["declarations"])
    assert "math.sqrt" in source["syntactic_call_symbols"]
    assert indexed["vendor/project_foundry/engine.py"]["category"] == "vendor"
    assert indexed["configs/evaluation.yaml"]["top_level_config_keys"] == ["episodes"]
    assert indexed["metadata.json"]["top_level_config_keys"] == ["kind"]
    assert indexed["pyproject.toml"]["top_level_config_keys"] == ["project"]
    assert indexed["docs/readme.md"]["markdown"][
        "unresolved_relative_link_candidates"
    ] == ["absent.md"]
    assert any(p["kind"] == "python_syntax" for p in report["problems"])
    assert report["evidence_level"] == "TRACKED_FILE_READ_AND_STATIC_PARSE_ONLY"


def test_git_index_is_authoritative_and_missing_tracked_file_is_rejected(
    tmp_path: Path,
) -> None:
    _git_index(tmp_path, {"tracked.py": "pass\n", "docs/a.md": "hello\n"})
    assert len(tracked_paths(tmp_path)) == 2
    (tmp_path / "tracked.py").unlink()
    with pytest.raises(FileNotFoundError, match="tracked file missing"):
        tracked_paths(tmp_path)


def test_live_repository_source_census_covers_exact_git_index() -> None:
    root = Path(__file__).resolve().parents[2]
    tracked = tracked_paths(root)
    result = scan_tracked_repository(root)
    assert result["tracked_file_count"] == len(tracked)
    assert len(result["records"]) == len(tracked)
    assert sum(result["by_category"].values()) == len(tracked)
    assert all(len(record["sha256"]) == 64 for record in result["records"])
    names = {record["path"] for record in result["records"]}
    assert "src/rl_bgd/bayes/bgd.py" in names
    assert "run_all_training.py" in names
    assert "vendor/project_foundry/README.md" in names or any(
        name.startswith("vendor/project_foundry/") for name in names
    )
