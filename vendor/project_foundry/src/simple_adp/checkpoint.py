"""Atomic checkpoint persistence and reproducibility state."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def stable_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else []
        ),
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    cuda_state = state.get("cuda")
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]


class CheckpointStore:
    """Atomic torch checkpoint store guarded by the complete run signature."""

    FORMAT_VERSION = 1

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def save(self, payload: dict[str, Any], *, signature: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"format_version": self.FORMAT_VERSION, "signature": signature, "payload": payload}
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        os.close(fd)
        try:
            torch.save(record, temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            if os.name == "posix":
                directory_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, *, signature: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        if not self.path.is_file():
            raise ValueError(f"exact-resume failure: checkpoint is missing: {self.path}")
        try:
            record = torch.load(self.path, map_location=map_location, weights_only=False)
        except Exception as exc:
            raise ValueError(f"exact-resume failure: checkpoint could not be loaded: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError("exact-resume failure: checkpoint envelope is not a mapping")
        if record.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("exact-resume failure: unsupported checkpoint format version")
        if record.get("signature") != signature:
            raise ValueError("exact-resume failure: checkpoint configuration/data signature does not match this run")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("exact-resume failure: checkpoint payload is not a mapping")
        return payload
