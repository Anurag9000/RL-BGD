"""Stationary BGD-PPO validation on the LQR-style environment."""

from __future__ import annotations

import json

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.bgd_agent import (
    BGDPPOAgent,
    BGDPPOConfig,
)
from rl_bgd.agents.ppo.train import (
    PPOTrainConfig,
    evaluate_ppo,
    train_ppo,
)
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


def run_bgd_ppo_lqr(
    *,
    steps: int = 800,
    seed: int = 29,
    device: str = "auto",
    bayesianization: str = "actor_and_value",
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
    agent = BGDPPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(32, 32),
        ppo_config=PPOConfig(
            actor_lr=1e-3,
            value_lr=1e-3,
            update_epochs=4,
            minibatch_size=64,
        ),
        bgd_config=BGDPPOConfig(
            bayesianization=bayesianization,  # type: ignore[arg-type]
            posterior_std=0.1,
            actor_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
            value_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
        ),
        device=resolved,
    )
    pre_return = evaluate_ppo(
        env,
        agent,
        episodes=5,
        seed=40_000,
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
        seed=40_000,
    )
    return {
        "bayesianization": bayesianization,
        "steps": steps,
        "pre_return": pre_return,
        "post_return": post_return,
        "improvement": (
            post_return - pre_return
        ),
        "training": summary,
    }


def main() -> None:
    print(
        json.dumps(
            run_bgd_ppo_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
