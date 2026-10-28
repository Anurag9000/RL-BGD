"""Run exactly one native RL-BGD scientific job with strict result checks.

The outer Project Foundry process controls physical job admission. The
native suite still owns paper provenance, scientific metrics and artifacts.
"""
from __future__ import annotations

import argparse
import json

from rl_bgd.experiments.suites import execute_suite


def main():
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
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") != "success" or result.get("jobs_selected") != 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
