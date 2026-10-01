"""CLI for the five-way hidden-context SAC comparison."""

from __future__ import annotations

import argparse
import json

from rl_bgd.runners.hidden_context_sac_comparison import (
    run_hidden_context_sac_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=90)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_hidden_context_sac_comparison(
        steps=args.steps,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
