"""CLI for one resumable recurrent stationary-LQR acceptance run."""

from __future__ import annotations

import argparse
import json

from rl_bgd.runners.recurrent_bgd_stationary_lqr import (
    run_recurrent_adam_ppo_lqr,
    run_recurrent_bgd_ppo_lqr,
    run_recurrent_bgd_sac_lqr,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=("adam_ppo", "bgd_ppo", "bgd_sac"),
        required=True,
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--bayesianization",
        choices=("actor_only", "value_only", "actor_and_value"),
        default="actor_only",
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-units-this-call", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    common = {
        "device": args.device,
        "checkpoint_path": args.checkpoint_path,
        "resume_from": args.resume_from,
    }
    if args.algorithm == "adam_ppo":
        result = run_recurrent_adam_ppo_lqr(
            steps=2_000 if args.steps is None else args.steps,
            seed=100 if args.seed is None else args.seed,
            checkpoint_interval_rollouts=args.checkpoint_interval,
            max_rollouts_this_call=args.max_units_this_call,
            **common,
        )
    elif args.algorithm == "bgd_ppo":
        result = run_recurrent_bgd_ppo_lqr(
            steps=2_000 if args.steps is None else args.steps,
            seed=101 if args.seed is None else args.seed,
            bayesianization=args.bayesianization,
            checkpoint_interval_rollouts=args.checkpoint_interval,
            max_rollouts_this_call=args.max_units_this_call,
            **common,
        )
    else:
        result = run_recurrent_bgd_sac_lqr(
            steps=800 if args.steps is None else args.steps,
            seed=102 if args.seed is None else args.seed,
            checkpoint_interval=args.checkpoint_interval,
            max_steps_this_call=args.max_units_this_call,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
