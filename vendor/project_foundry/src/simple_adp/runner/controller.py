"""Subprocess entry for the central scheduler, separated from outer launch setup."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import RunnerSettings
from .plan import SuitePlan
from .runtime_tuning import bootstrap_runtime
from .scheduler import SuiteCoordinator
from .logging_context import configure_console_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coordinate a persisted simple-adp suite plan")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--source-run-root")
    args = parser.parse_args(argv)
    configure_console_logging()
    bootstrap_runtime("simple-adp-central-runner")
    plan = SuitePlan.from_manifest(args.plan)
    payload = plan.manifest
    settings = RunnerSettings(**payload["runner_settings"])
    coordinator = SuiteCoordinator(
        plan,
        settings,
        run_root=args.run_root,
        results_dir=args.results_dir,
        repository_root=args.repository_root,
        source_run_root=args.source_run_root,
    )
    summary = coordinator.run()
    print(json.dumps(summary, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
