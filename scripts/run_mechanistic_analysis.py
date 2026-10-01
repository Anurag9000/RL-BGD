"""Run the complete deterministic Phase-13 mechanistic suite."""

from __future__ import annotations

import argparse
import json

from rl_bgd.analysis.mechanistic import (
    MechanisticAnalysisConfig,
    run_mechanistic_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/mechanistic",
    )
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = run_mechanistic_analysis(
        args.output_dir,
        config=MechanisticAnalysisConfig(
            seed=args.seed,
            device=args.device,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
