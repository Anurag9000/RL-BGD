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
    mc_samples: int = 2,
    horizon: int = 30,
    hidden_dims: tuple[int, ...] = (32, 32),
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    alpha_lr: float = 1e-3,
    initial_alpha: float = 0.2,
    automatic_entropy_tuning: bool = True,
    target_entropy: float | None = None,
    gradient_clip_norm: float | None = None,
    posterior_std: float = 0.1,
    sigma_min: float = 1e-6,
    sigma_max: float = 10.0,
    bgd_eta: float = 0.1,
    random_steps: int = 64,
    batch_size: int = 64,
    replay_capacity: int = 2_000,
    evaluation_episodes: int = 10,
    evaluation_seed: int = 30_000,
) -> dict[str, object]:
    if mc_samples < 1:
        raise ValueError("mc_samples must be positive")
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    env = LinearQuadraticControlEnv(
        horizon=horizon,
        device=resolved,
    )
    agent = BGDSACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=hidden_dims,
        sac_config=SACConfig(
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            initial_alpha=initial_alpha,
            automatic_entropy_tuning=automatic_entropy_tuning,
            target_entropy=target_entropy,
            gradient_clip_norm=gradient_clip_norm,
        ),
        bgd_config=BGDSACConfig(
            bayesianization=bayesianization,  # type: ignore[arg-type]
            posterior_std=posterior_std,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            replay_evidence=ReplayEvidenceConfig(
                mode=replay_evidence_mode,
            ),
            actor_bgd=BGDConfig(
                eta=bgd_eta,
                mc_samples=mc_samples,
                antithetic=(mc_samples > 1 and mc_samples % 2 == 0),
                evidence_temperature=evidence_temperature,
                temper_retention=temper_retention,
            ),
            critic_bgd=BGDConfig(
                eta=bgd_eta,
                mc_samples=mc_samples,
                antithetic=(mc_samples > 1 and mc_samples % 2 == 0),
                evidence_temperature=evidence_temperature,
                temper_retention=temper_retention,
            ),
        ),
        device=resolved,
    )
    pre_return = evaluate_sac_lqr(
        env,
        agent,
        episodes=evaluation_episodes,
        seed=evaluation_seed,
    )
    summary = train_sac(
        env,
        agent,
        config=SACTrainConfig(
            total_steps=steps,
            random_steps=random_steps,
            batch_size=batch_size,
            replay_capacity=max(replay_capacity, steps),
            seed=seed,
        ),
    )
    post_return = evaluate_sac_lqr(
        env,
        agent,
        episodes=evaluation_episodes,
        seed=evaluation_seed,
    )
    return {
        "bayesianization": bayesianization,
        "evidence_temperature": evidence_temperature,
        "temper_retention": temper_retention,
        "replay_evidence_mode": replay_evidence_mode,
        "mc_samples": mc_samples,
        "antithetic": (mc_samples > 1 and mc_samples % 2 == 0),
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
