"""Materialize or execute a curated paper experiment suite."""

from __future__ import annotations

import argparse
import json

from rl_bgd.experiments.suites import (
    SUITES,
    execute_suite,
    materialize_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "suite",
        choices=tuple(SUITES),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/suites",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rerun even strict successful jobs whose saved contract still matches",
    )
    parser.add_argument(
        "--job-id",
        action="append",
        default=[],
        help="execute only this suite job ID; repeat to select multiple jobs",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="execute only this seed; repeat to select multiple seeds",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="execute only this expanded run ID; repeat to select multiple runs",
    )
    args = parser.parse_args()
    if args.execute:
        result = execute_suite(
            args.suite,
            args.output_root,
            continue_on_error=args.continue_on_error,
            resume=not args.no_resume,
            job_ids=(
                tuple(
                    args.job_id
                )
                or None
            ),
            seeds=(
                tuple(
                    args.seed
                )
                or None
            ),
            run_ids=(
                tuple(
                    args.run_id
                )
                or None
            ),
        )
    else:
        result = materialize_suite(
            args.suite,
            args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
