"""Small compatibility helpers recorded in the runner behavior ledger.

These helpers are intentionally independent of the scheduler.  The reference
runner defines several of them without in-file callers, so keeping them in a
small module preserves their reusable contracts without coupling them to
controller state.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def local_timestamp() -> str:
    """Return the ledger's local-time timestamp with one-second precision."""
    return time.strftime("%Y%m%d_%H%M%S")


def env_int(name: str, default: int, environ: Mapping[str, str] | None = None) -> int:
    """Return an environment integer, defaulting only for absent/invalid text."""
    source = os.environ if environ is None else environ
    value = source.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def env_float(name: str, default: float, environ: Mapping[str, str] | None = None) -> float:
    """Return an environment float, defaulting only for absent/invalid text."""
    source = os.environ if environ is None else environ
    value = source.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def parse_csv_ints(value: Any) -> list[int]:
    """Parse nonempty comma-separated tokens as integers, preserving order."""
    return [int(token.strip()) for token in str(value).split(",") if token.strip()]


def parse_range_tuple(value: Any) -> tuple[int, int]:
    """Parse two nonempty comma-separated integer tokens, dropping empty ones."""
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated integers, got: {value}")
    return int(parts[0]), int(parts[1])


def format_kv(value: Any) -> str:
    """Format primitive values directly and compound values as sorted JSON."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def json_signature(value: Any) -> str:
    """SHA-256 of canonical sorted, compact, ASCII-escaped JSON."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_json(path: str | Path, default: Any = None) -> Any:
    """Read JSON, returning the supplied default for any read/parse failure."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default
