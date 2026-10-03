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
    args = parser.parse_args()
    if args.execute:
        result = execute_suite(
            args.suite,
            args.output_root,
            continue_on_error=args.continue_on_error,
            resume=not args.no_resume,
        )
    else:
        result = materialize_suite(
            args.suite,
            args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
