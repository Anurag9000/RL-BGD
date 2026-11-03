"""Generate complete tracked-file static source inventory, without executing jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rl_bgd.audit.source_inventory import scan_tracked_repository


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/audit/source_inventory.json")
    )
    parser.add_argument("--fail-on-static-errors", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = scan_tracked_repository(root)
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "source_revision": report["git_revision"],
                "tracked_file_count": report["tracked_file_count"],
                "by_category": report["by_category"],
                "static_problems": len(report["problems"]),
                "output": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_static_errors and report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
