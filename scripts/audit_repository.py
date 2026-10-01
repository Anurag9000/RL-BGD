"""Run the Phase-16 repository closure audit."""

from __future__ import annotations

import json

from rl_bgd.audit.repository import audit_repository


def main() -> None:
    report = audit_repository()
    print(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
