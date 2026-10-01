"""Build paper tables/figures directly from suite raw outputs."""

from __future__ import annotations

import argparse
import json

from rl_bgd.analysis.artifacts import (
    BootstrapConfig,
    build_paper_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        default="artifacts/suites",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/paper",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
    )
    args = parser.parse_args()
    result = build_paper_artifacts(
        args.run_root,
        args.output_dir,
        bootstrap=BootstrapConfig(
            samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.bootstrap_seed,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
