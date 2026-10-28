"""Verify the tracked vendored Project Foundry snapshot with Git blob SHA-1s."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "docs/foundry_integration/SOURCE_FILE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vendor = root / manifest["vendor_root"]
    errors: list[str] = []
    expected: set[str] = set()
    for entry in manifest["files"]:
        rel = entry["path"]
        expected.add(rel)
        dest = vendor / rel
        if not dest.is_file():
            errors.append(f"MISSING: {rel}")
            continue
        data = dest.read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != entry["sha"]:
            errors.append(f"CONTENT MISMATCH: {rel}: {digest} != {entry['sha']}")
        mode = "120000" if dest.is_symlink() else "100755" if (dest.stat().st_mode & 0o111) else "100644"
        if mode != entry["mode"]:
            errors.append(f"MODE MISMATCH: {rel}: {mode} != {entry['mode']}")
    actual = {str(p.relative_to(vendor)) for p in vendor.rglob("*") if p.is_file()}
    for path in sorted(actual - expected):
        errors.append(f"UNEXPECTED FILE: {path}")
    if len(expected) != manifest["expected_file_count"]:
        errors.append("INVALID MANIFEST: unexpected file count")
    return errors


if __name__ == "__main__":
    problems = verify()
    if problems:
        for problem in problems:
            print(problem)
        raise SystemExit(1)
    print("Verified all 78 Project Foundry tracked files against pinned Git blobs.")
