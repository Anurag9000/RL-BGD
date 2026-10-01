"""CLI for task-agnostic parameter-importance SAC baselines."""

from __future__ import annotations

import argparse
import json

from rl_bgd.runners.regularized_sac_continual_lqr import (
    run_regularized_sac_recurring_lqr,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("ewc", "online_ewc", "si", "mas"),
        default="ewc",
    )
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--consolidation-interval-updates",
        type=int,
        default=8,
    )
    args = parser.parse_args()
    result = run_regularized_sac_recurring_lqr(
        method=args.method,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        consolidation_interval_updates=args.consolidation_interval_updates,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
