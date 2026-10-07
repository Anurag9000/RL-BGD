"""CLI for resumable stationary BGD-PPO experiments."""

from __future__ import annotations

import argparse
import json

from rl_bgd.runners.bgd_ppo_lqr import run_bgd_ppo_lqr


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--bayesianization",
        choices=("actor_only", "value_only", "actor_and_value"),
        default="actor_and_value",
    )
    parser.add_argument("--posterior-std", type=float, default=0.1)
    parser.add_argument(
        "--evidence-mode",
        choices=("all_epochs", "first_epoch_only", "normalized_epochs"),
        default="first_epoch_only",
    )
    parser.add_argument("--bgd-eta", type=float, default=0.1)
    parser.add_argument("--mc-samples", type=int, default=2)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-interval-rollouts", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-rollouts-this-call", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_bgd_ppo_lqr(
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        bayesianization=args.bayesianization,
        posterior_std=args.posterior_std,
        evidence_mode=args.evidence_mode,
        bgd_eta=args.bgd_eta,
        mc_samples=args.mc_samples,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval_rollouts=args.checkpoint_interval_rollouts,
        resume_from=args.resume_from,
        max_rollouts_this_call=args.max_rollouts_this_call,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
