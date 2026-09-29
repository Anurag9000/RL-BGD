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
) -> dict[str, object]:
    seed_everything(
        seed,
        deterministic=True,
    )
    resolved = resolve_device(device)
    env = LinearQuadraticControlEnv(
        horizon=30,
        device=resolved,
    )
    agent = PPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(32, 32),
        config=PPOConfig(
            actor_lr=1e-3,
            value_lr=1e-3,
            update_epochs=6,
            minibatch_size=64,
            gradient_clip_norm=0.5,
        ),
        device=resolved,
    )
    pre_return = evaluate_ppo(
        env,
        agent,
        episodes=5,
        seed=30_000,
    )
    summary = train_ppo(
        env,
        agent,
        config=PPOTrainConfig(
            total_steps=steps,
            rollout_steps=128,
            seed=seed,
        ),
    )
    post_return = evaluate_ppo(
        env,
        agent,
        episodes=5,
        seed=30_000,
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
