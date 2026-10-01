"""CLI for strict task-agnostic CARL Pendulum SAC experiments."""

from __future__ import annotations

import argparse
import json

from rl_bgd.runners.carl_pendulum_sac import run_carl_pendulum_sac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("abrupt", "smooth", "recurring"),
        default="recurring",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "bgd", "adaptive_bgd"),
        default="adam",
    )
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--phase-steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_carl_pendulum_sac(
        mode=args.mode,
        optimizer=args.optimizer,
        steps=args.steps,
        phase_steps=args.phase_steps,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
