"""CLI for resumable stationary SAC on the synthetic LQR environment."""

from __future__ import annotations

import argparse
import json

from rl_bgd.runners.sac_lqr import run_sac_lqr


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps-this-call", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_sac_lqr(
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
        max_steps_this_call=args.max_steps_this_call,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
