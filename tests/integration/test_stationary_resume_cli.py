from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("script", "required_flags"),
    [
        (
            "run_sac_lqr.py",
            (
                "--checkpoint-path",
                "--checkpoint-interval",
                "--resume-from",
                "--max-steps-this-call",
            ),
        ),
        (
            "run_ppo_lqr.py",
            (
                "--checkpoint-path",
                "--checkpoint-interval-rollouts",
                "--resume-from",
                "--max-rollouts-this-call",
            ),
        ),
        (
            "run_bgd_sac_lqr.py",
            (
                "--checkpoint-path",
                "--checkpoint-interval",
                "--resume-from",
                "--max-steps-this-call",
            ),
        ),
        (
            "run_bgd_ppo_lqr.py",
            (
                "--checkpoint-path",
                "--checkpoint-interval-rollouts",
                "--resume-from",
                "--max-rollouts-this-call",
            ),
        ),
        (
            "run_recurrent_stationary_lqr.py",
            (
                "--algorithm",
                "--checkpoint-path",
                "--checkpoint-interval",
                "--resume-from",
                "--max-units-this-call",
            ),
        ),
    ],
)
def test_stationary_cli_exposes_resume_controls(
    script: str,
    required_flags: tuple[str, ...],
) -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / script),
            "--help",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    for flag in required_flags:
        assert flag in completed.stdout
