"""Execute canonical paper-suite runs concurrently across GPU/CPU worker slots."""

from __future__ import annotations

import argparse
import json

from rl_bgd.experiments.parallel import (
    detect_gpu_ids,
    run_suite_parallel,
)
from rl_bgd.experiments.suites import (
    SUITES,
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
        "--gpu-ids",
        default=None,
        help=(
            "Comma-separated physical GPU IDs. When omitted, "
            "CUDA_VISIBLE_DEVICES or nvidia-smi discovery is used."
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="CPU worker count when no GPU is available.",
    )
    parser.add_argument(
        "--min-free-vram-mb",
        type=int,
        default=0,
        help="Wait until an assigned GPU has at least this much free VRAM.",
    )
    parser.add_argument(
        "--max-gpu-utilization",
        type=int,
        default=100,
        help="Wait while assigned GPU utilization exceeds this percentage.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
    )
    parser.add_argument(
        "--job-id",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
    )
    args = parser.parse_args()

    gpu_ids = detect_gpu_ids(
        explicit=args.gpu_ids,
    )
    summary = run_suite_parallel(
        args.suite,
        args.output_root,
        job_ids=tuple(args.job_id),
        seeds=tuple(args.seed),
        run_ids=tuple(args.run_id),
        gpu_ids=gpu_ids,
        workers_per_gpu=(args.workers_per_gpu),
        cpu_workers=(args.cpu_workers),
        min_free_vram_mb=(args.min_free_vram_mb),
        max_gpu_utilization=(args.max_gpu_utilization),
        poll_seconds=(args.poll_seconds),
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )
    if summary["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
