"""Execute and strictly certify exactly one native RL-BGD scientific job.

Project Foundry schedules physical workers; RL-BGD owns scientific artifacts.
A child exit code alone cannot establish scientific success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_bgd.experiments.suites import (
    SUITES,
    _completed_run_matches_job,
    _expanded_jobs,
    _git_head,
    execute_suite,
)


def verify_native_job(
    suite: str,
    job_id: str,
    seed: int,
    output_root: str | Path,
) -> dict[str, str]:
    """Bind a Foundry success to exactly one validated canonical native run."""
    if suite not in SUITES:
        raise ValueError(f"unknown native suite: {suite}")
    expanded = _expanded_jobs(SUITES[suite], Path(output_root))
    selected = [
        job for job in expanded
        if job["job_id"] == job_id and job["seed"] == seed
    ]
    if len(selected) != 1:
        raise ValueError("native scientific job and seed must identify exactly one run")
    job = selected[0]
    git_commit = _git_head()
    if not git_commit:
        raise ValueError("native run verification requires a concrete Git revision")
    run_dir = Path(job["run_dir"])
    if not _completed_run_matches_job(
        run_dir, suite_name=suite, git_commit=git_commit, job=job
    ):
        raise ValueError(
            "native result failed strict canonical artifact, configuration, "
            "revision, or completion verification"
        )
    return {
        "run_id": str(job["run_id"]),
        "run_dir": str(run_dir),
        "git_commit": git_commit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = execute_suite(
        args.suite, args.output_root, continue_on_error=False,
        resume=True, job_ids=(args.job_id,), seeds=(args.seed,),
    )
    if result.get("status") != "success" or result.get("jobs_selected") != 1:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    verified = verify_native_job(
        args.suite, args.job_id, args.seed, args.output_root
    )
    if result.get("selected_run_ids") != [verified["run_id"]]:
        raise ValueError("native suite selected a different scientific run")
    print(json.dumps({"execution": result, "verified_native_run": verified},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
