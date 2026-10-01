"""CLI for recurrent hidden-context SAC comparisons."""

from __future__ import annotations

import argparse
import json

from rl_bgd.runners.recurrent_sac_continual_lqr import (
    run_recurrent_sac_recurring_lqr,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer",
        choices=("adam", "bgd", "adaptive_bgd"),
        default="adam",
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_recurrent_sac_recurring_lqr(
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        optimizer=args.optimizer,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
