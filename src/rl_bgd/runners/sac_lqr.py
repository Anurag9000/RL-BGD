"""Stationary SAC validation on the dependency-light LQR-style environment."""

from __future__ import annotations

import json

import torch

from rl_bgd.agents.sac.agent import (
    SACAgent,
    SACConfig,
)
from rl_bgd.agents.sac.train import (
    SACTrainConfig,
    train_sac,
)
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import (
    seed_everything,
)


@torch.no_grad()
def evaluate_sac_lqr(
    env: LinearQuadraticControlEnv,
    agent: SACAgent,
    *,
    episodes: int = 10,
    seed: int = 10_000,
) -> float:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    returns: list[float] = []
    for index in range(episodes):
        observation, _ = env.reset(
            seed=seed + index
        )
        total = 0.0
        while True:
            action = agent.act(
                observation,
                deterministic=True,
            )
            (
                observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    return sum(returns) / len(returns)


def run_sac_lqr(
    *,
    steps: int = 800,
    seed: int = 7,
    device: str = "auto",
) -> dict[str, object]:
    seed_everything(
        seed, deterministic=True
    )
    resolved = resolve_device(device)
    env = LinearQuadraticControlEnv(
        horizon=30, device=resolved
    )
    agent = SACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(32, 32),
        config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        device=resolved,
    )
    pre_return = evaluate_sac_lqr(
        env, agent, seed=20_000
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
        env, agent, seed=20_000
    )
    return {
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
            run_sac_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
