"""Stationary BGD-SAC validation on the LQR-style environment."""

from __future__ import annotations

import json

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.bgd_agent import (
    BGDSACAgent,
    BGDSACConfig,
)
from rl_bgd.agents.sac.train import (
    SACTrainConfig,
    train_sac,
)
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)
from rl_bgd.replay.evidence_accounting import (
    ReplayEvidenceConfig,
    ReplayEvidenceMode,
)
from rl_bgd.runners.sac_lqr import (
    evaluate_sac_lqr,
)
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import (
    seed_everything,
)


def run_bgd_sac_lqr(
    *,
    steps: int = 600,
    seed: int = 11,
    device: str = "auto",
    bayesianization: str = "critic_only",
    evidence_temperature: float = 1.0,
    temper_retention: float = 1.0,
    replay_evidence_mode: ReplayEvidenceMode = "all_replay",
) -> dict[str, object]:
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    env = LinearQuadraticControlEnv(horizon=30, device=resolved)
    agent = BGDSACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(32, 32),
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        bgd_config=BGDSACConfig(
            bayesianization=bayesianization,  # type: ignore[arg-type]
            posterior_std=0.1,
            replay_evidence=ReplayEvidenceConfig(
                mode=replay_evidence_mode,
            ),
            actor_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
                evidence_temperature=evidence_temperature,
                temper_retention=temper_retention,
            ),
            critic_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
                evidence_temperature=evidence_temperature,
                temper_retention=temper_retention,
            ),
        ),
        device=resolved,
    )
    pre_return = evaluate_sac_lqr(
        env,
        agent,
        seed=30_000,
    )
    summary = train_sac(
        env,
        agent,
        config=SACTrainConfig(
            total_steps=steps,
            random_steps=64,
            batch_size=64,
            replay_capacity=max(2_000, steps),
            seed=seed,
        ),
    )
    post_return = evaluate_sac_lqr(
        env,
        agent,
        seed=30_000,
    )
    return {
        "bayesianization": bayesianization,
        "evidence_temperature": evidence_temperature,
        "temper_retention": temper_retention,
        "replay_evidence_mode": replay_evidence_mode,
        "steps": steps,
        "pre_return": pre_return,
        "post_return": post_return,
        "improvement": (post_return - pre_return),
        "training": summary,
    }


def main() -> None:
    print(
        json.dumps(
            run_bgd_sac_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
