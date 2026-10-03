"""Download and safely extract an authoritative CORA CHORES trajectory archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_bgd.compat.cora_chores_download import (
    OFFICIAL_CHORES_ARCHIVE_URL,
    download_chores_archive,
    extract_chores_archive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover CORA's historical CHORES trajectories from the official "
            "OneDrive source or an explicitly supplied authoritative mirror."
        )
    )
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        default=None,
        help=(
            "Archive candidate URL. May be repeated. When omitted, the "
            "official CORA OneDrive URL is tried."
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Path where the validated ZIP will be published.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="Optional directory for safe ZIP extraction.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Optional expected SHA-256 for an authoritative mirror.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-candidate network timeout in seconds.",
    )
    parser.add_argument(
        "--attempts-per-url",
        type=int,
        default=3,
        help="Retries for transient network failures per candidate URL.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    urls = (
        tuple(args.urls)
        if args.urls
        else (
            OFFICIAL_CHORES_ARCHIVE_URL,
        )
    )
    report = download_chores_archive(
        destination=args.destination,
        urls=urls,
        expected_sha256=(
            args.expected_sha256
        ),
        timeout=args.timeout,
        attempts_per_url=(
            args.attempts_per_url
        ),
    )
    payload = report.to_dict()
    if (
        args.extract_dir
        is not None
    ):
        payload[
            "extract_dir"
        ] = str(
            extract_chores_archive(
                archive=report.destination,
                destination=(
                    args.extract_dir
                ),
            )
        )
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
