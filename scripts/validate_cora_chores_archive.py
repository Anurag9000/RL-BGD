"""Validate a recovered CORA CHORES trajectory archive against pinned metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_bgd.compat.cora_chores import (
    EXPECTED_CHORES_TRAJECTORY_COUNT,
    find_chores_archive_root,
    load_chores_trajectory_refs,
    validate_chores_archive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Locate and validate every train/valid_seen trajectory referenced "
            "by CORA's pinned CHORES metadata."
        )
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        required=True,
        help="Directory containing the extracted trajectory archive.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        required=True,
        help="Pinned CORA chores metadata directory.",
    )
    parser.add_argument(
        "--expected-trajectories",
        type=int,
        default=EXPECTED_CHORES_TRAJECTORY_COUNT,
        help=(
            "Exact number of trajectory references expected; defaults to the "
            "pinned CORA benchmark cardinality."
        ),
    )
    parser.add_argument(
        "--root-only",
        action="store_true",
        help="Print only the validated ALFRED_DATA_DIR path.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    refs = load_chores_trajectory_refs(args.metadata_root)
    if len(refs) != args.expected_trajectories:
        raise ValueError(
            "CORA CHORES metadata trajectory count mismatch: "
            f"expected {args.expected_trajectories}, found {len(refs)}"
        )
    archive_root = find_chores_archive_root(
        args.search_root,
        refs,
    )
    report = validate_chores_archive(
        archive_root=archive_root,
        metadata_root=args.metadata_root,
    )
    if args.root_only:
        print(report.archive_root)
        return
    print(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
