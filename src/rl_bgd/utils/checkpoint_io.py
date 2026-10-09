"""Crash-resistant atomic training checkpoint saves with unique staging files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry after atomic publication on POSIX systems."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_training_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    """Save with fsynced staging bytes and crash-durable atomic publication.

    Parallel writers use independent staging files. If multiple runs target the
    same destination, the last completed replacement wins; run IDs should still
    select different destination paths for scientific reproducibility.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            torch.save(state, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
