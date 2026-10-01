"""CLI for 3RL-style strict task-agnostic Continual World."""

from __future__ import annotations

import argparse
import json

from rl_bgd.runners.recurrent_continual_world import (
    run_recurrent_ta_continual_world_sac,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=("CW10", "CW20"),
        default="CW10",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "bgd", "adaptive_bgd"),
        default="adam",
    )
    parser.add_argument(
        "--steps-per-task",
        type=int,
        default=1_000_000,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    result = run_recurrent_ta_continual_world_sac(
        benchmark=args.benchmark,
        optimizer=args.optimizer,
        steps_per_task=args.steps_per_task,
        seed=args.seed,
        device=args.device,
        evaluation_episodes=args.evaluation_episodes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
