"""Build publication artifacts strictly from completed raw result directories."""

from __future__ import annotations

import argparse
import json

from rl_bgd.analysis.paper_artifacts import (
    PaperArtifactConfig,
    build_paper_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/paper",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--figure-formats",
        nargs="+",
        default=[
            "png",
            "pdf",
            "svg",
        ],
    )
    args = parser.parse_args()
    result = build_paper_artifacts(
        args.results_root,
        args.output_dir,
        config=PaperArtifactConfig(
            confidence=args.confidence,
            bootstrap_resamples=(
                args.bootstrap_resamples
            ),
            seed=args.seed,
            figure_formats=tuple(
                args.figure_formats
            ),
        ),
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
