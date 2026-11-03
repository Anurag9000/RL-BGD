"""Dependency-light CLI argument contracts for recurrent Continual World."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_recurrent_continual_world.py"


def test_recurrent_continual_world_cli_exposes_replay_controls() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for option in (
        "--replay-storage",
        "--replay-capacity",
        "--sequence-batch-size",
        "--history-length",
        "--unroll",
        "--initial-random-steps",
        "--episode-horizon",
    ):
        assert option in result.stdout


def test_recurrent_continual_world_cli_rejects_unimplemented_replay_mode() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--replay-storage", "not-a-policy"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
