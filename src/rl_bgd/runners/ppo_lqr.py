"""Stationary PPO validation on the dependency-light LQR environment."""

from __future__ import annotations

import json

from rl_bgd.agents.ppo.agent import (
    PPOAgent,
    PPOConfig,
)
from rl_bgd.agents.ppo.train import (
    PPOTrainConfig,
    evaluate_ppo,
    train_ppo,
)
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


def run_ppo_lqr(
    *,
    steps: int = 800,
    seed: int = 19,
    device: str = "auto",
    horizon: int = 30,
    hidden_dims: tuple[int, ...] = (32, 32),
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    value_clip_ratio: float | None = 0.2,
    actor_lr: float = 1e-3,
    value_lr: float = 1e-3,
    entropy_coef: float = 0.0,
    value_coef: float = 0.5,
    update_epochs: int = 6,
    minibatch_size: int = 64,
    gradient_clip_norm: float = 0.5,
    normalize_advantages: bool = True,
    target_kl: float | None = None,
    rollout_steps: int = 128,
    evaluation_episodes: int = 5,
    evaluation_seed: int = 30_000,
    checkpoint_path: str | None = None,
    checkpoint_interval_rollouts: int | None = None,
    resume_from: str | None = None,
    max_rollouts_this_call: int | None = None,
) -> dict[str, object]:
    seed_everything(
        seed,
        deterministic=True,
    )
    resolved = resolve_device(device)
    env = LinearQuadraticControlEnv(
        horizon=horizon,
        device=resolved,
    )
    agent = PPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=hidden_dims,
        config=PPOConfig(
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            value_clip_ratio=value_clip_ratio,
            actor_lr=actor_lr,
            value_lr=value_lr,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            gradient_clip_norm=gradient_clip_norm,
            normalize_advantages=normalize_advantages,
            target_kl=target_kl,
        ),
        device=resolved,
    )
    pre_return = evaluate_ppo(
        env,
        agent,
        episodes=evaluation_episodes,
        seed=evaluation_seed,
    )
    summary = train_ppo(
        env,
        agent,
        config=PPOTrainConfig(
            total_steps=steps,
            rollout_steps=rollout_steps,
            seed=seed,
        ),
        checkpoint_path=checkpoint_path,
        checkpoint_interval_rollouts=checkpoint_interval_rollouts,
        resume_from=resume_from,
        max_rollouts_this_call=max_rollouts_this_call,
    )
    post_return = evaluate_ppo(
        env,
        agent,
        episodes=evaluation_episodes,
        seed=evaluation_seed,
    )
    return {
        "steps": steps,
        "pre_return": pre_return,
        "post_return": post_return,
        "improvement": (post_return - pre_return),
        "training": summary,
    }


def main() -> None:
    print(
        json.dumps(
            run_ppo_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
