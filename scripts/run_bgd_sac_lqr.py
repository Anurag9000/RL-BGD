"""CLI for resumable stationary BGD-SAC experiments."""

from __future__ import annotations

import argparse
import json

from rl_bgd.runners.bgd_sac_lqr import run_bgd_sac_lqr


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--bayesianization",
        choices=("critic_only", "actor_only", "actor_and_critic"),
        default="critic_only",
    )
    parser.add_argument("--evidence-temperature", type=float, default=1.0)
    parser.add_argument("--temper-retention", type=float, default=1.0)
    parser.add_argument(
        "--replay-evidence-mode",
        choices=(
            "all_replay",
            "fresh_only_uncertainty",
            "inverse_reuse_weight",
            "normalized_batch_evidence",
        ),
        default="all_replay",
    )
    parser.add_argument("--mc-samples", type=int, default=2)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps-this-call", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_bgd_sac_lqr(
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        bayesianization=args.bayesianization,
        evidence_temperature=args.evidence_temperature,
        temper_retention=args.temper_retention,
        replay_evidence_mode=args.replay_evidence_mode,
        mc_samples=args.mc_samples,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
        max_steps_this_call=args.max_steps_this_call,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
