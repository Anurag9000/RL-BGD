"""Chunked resume bundles and safe telemetry shard rehydration."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

PART_SIZE_BYTES = 64 * 1024 * 1024


def _safe_name(path: Path) -> str:
    return "__".join(path.parts)


def bundle_path_for_results_dir(results_dir: str | Path, repository_root: str | Path) -> Path:
    resolved = Path(results_dir).resolve()
    root = Path(repository_root).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        relative = Path(*resolved.parts[1:]) if resolved.is_absolute() else resolved
    return root / "resume_snapshots" / f"{_safe_name(relative)}.tar.gz"


def _manifest_path(bundle_path: Path) -> Path:
    return bundle_path.with_suffix("").with_suffix(".json")


def _parts(bundle_path: Path) -> list[Path]:
    return sorted(bundle_path.parent.glob(f"{bundle_path.name}.part*"))


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except Exception:
            continue
    return total


def _manifest(results_dir: Path, bundle_path: Path) -> dict[str, Any]:
    return {
        "kind": "results",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(results_dir),
        "bundle_path": str(bundle_path),
        "bundle_mode": "chunked",
        "part_size_bytes": PART_SIZE_BYTES,
        "source_exists": results_dir.exists(),
        "source_size_bytes": _dir_size(results_dir),
    }


def _bundle_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if info.name == "telemetry.csv" or info.name.endswith("/telemetry.csv"):
        return None
    return info


def export_results_bundle(results_dir: str | Path, repository_root: str | Path, bundle_path: str | Path | None = None) -> Path | None:
    source = Path(results_dir).resolve()
    if not source.exists():
        return None
    destination = Path(bundle_path) if bundle_path is not None else bundle_path_for_results_dir(source, repository_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The source writes the manifest first and directly (not via atomic replace).
    _manifest_path(destination).write_text(json.dumps(_manifest(source, destination), indent=2, sort_keys=True), encoding="utf-8")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    for part in _parts(destination):
        part.unlink()
    destination.unlink(missing_ok=True)
    with tarfile.open(temporary, mode="w:gz", compresslevel=1) as archive:
        archive.add(source, arcname=source.name, recursive=True, filter=_bundle_filter)
    if temporary.stat().st_size <= PART_SIZE_BYTES:
        temporary.replace(destination)
        return destination
    with temporary.open("rb") as source_stream:
        index = 0
        while chunk := source_stream.read(PART_SIZE_BYTES):
            part_path = destination.with_suffix(destination.suffix + f".part{index:03d}")
            part_path.write_bytes(chunk)
            index += 1
    temporary.unlink()
    return destination


def _safe_extract(archive: tarfile.TarFile, destination: Path, expected_root: str) -> None:
    root = expected_root.strip()
    if not root or Path(root).name != root:
        raise ValueError(f"invalid bundle root {expected_root!r}")
    target_root = destination.resolve()
    members = archive.getmembers()
    for member in members:
        normalized = member.name.replace("\\", "/")
        member_path = PurePosixPath(normalized)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"unsafe bundle member path {member.name!r}")
        if not member_path.parts or member_path.parts[0] != root:
            raise ValueError(f"unexpected bundle root in {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are not permitted in bundles: {member.name!r}")
        target = (target_root / Path(*member_path.parts)).resolve()
        try:
            target.relative_to(target_root)
        except ValueError as exc:
            raise ValueError(f"unsafe bundle member path {member.name!r}") from exc
    archive.extractall(path=target_root)


def restore_results_bundle(results_dir: str | Path, repository_root: str | Path, bundle_path: str | Path | None = None) -> bool:
    target = Path(results_dir).resolve()
    bundle = Path(bundle_path) if bundle_path is not None else bundle_path_for_results_dir(target, repository_root)
    if target.exists():
        try:
            next(target.iterdir())
            return False
        except StopIteration:
            pass
        except Exception:
            return False
    parts = _parts(bundle)
    restore_path: Path | None = None
    if not bundle.exists():
        if not parts:
            return False
        # The source creates the extraction parent before reading/reassembling
        # multipart data; preserve that observable order on failures too.
        target.parent.mkdir(parents=True, exist_ok=True)
        restore_path = bundle.with_suffix(bundle.suffix + ".restore")
        with restore_path.open("wb") as destination:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=PART_SIZE_BYTES)
        archive_path = restore_path
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        archive_path = bundle
    with tarfile.open(archive_path, mode="r:gz") as archive:
        _safe_extract(archive, target.parent, target.name)
    if restore_path is not None:
        restore_path.unlink()
    return True


def _target_path(manifest_path: Path, original_path: str, repository_root: Path) -> Path:
    root = repository_root.resolve()
    original = Path(original_path)
    resolved = original.resolve() if original.is_absolute() else (root / original).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"telemetry target escapes repository: {original_path!r}") from exc
    return resolved


def rehydrate_sharded_telemetry(root: str | Path, repository_root: str | Path) -> int:
    restored = 0
    search_root = Path(root).resolve()
    repo = Path(repository_root).resolve()
    for manifest_path in search_root.rglob("*.manifest.json"):
        if ".csv.manifest.json" not in manifest_path.name:
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        original_path = payload.get("original_path")
        parts = payload.get("parts") or []
        if not original_path or not parts:
            continue
        target = _target_path(manifest_path, str(original_path), repo)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".rehydrate")
        digest = hashlib.sha256()
        with temporary.open("wb") as destination:
            for part_name in parts:
                part = (manifest_path.parent / str(part_name)).resolve()
                try:
                    part.relative_to(manifest_path.parent.resolve())
                except ValueError as exc:
                    raise ValueError(f"telemetry shard escapes manifest directory: {part_name!r}") from exc
                if not part.exists():
                    raise FileNotFoundError(part)
                with part.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        destination.write(chunk)
        expected = payload.get("sha256")
        if expected and digest.hexdigest() != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch for telemetry target {target}")
        temporary.replace(target)
        restored += 1
    return restored
