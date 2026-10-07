"""CLI for resumable stationary PPO on the synthetic LQR environment."""

from __future__ import annotations

import argparse
import json

from rl_bgd.runners.ppo_lqr import run_ppo_lqr


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-interval-rollouts", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-rollouts-this-call", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_ppo_lqr(
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval_rollouts=args.checkpoint_interval_rollouts,
        resume_from=args.resume_from,
        max_rollouts_this_call=args.max_rollouts_this_call,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
