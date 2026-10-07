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
        observation, _ = env.reset(seed=seed + index)
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
    random_steps: int = 64,
    batch_size: int = 64,
    replay_capacity: int = 2_000,
    evaluation_episodes: int = 10,
    evaluation_seed: int = 20_000,
    checkpoint_path: str | None = None,
    checkpoint_interval: int | None = None,
    resume_from: str | None = None,
    max_steps_this_call: int | None = None,
) -> dict[str, object]:
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    env = LinearQuadraticControlEnv(
        horizon=horizon,
        device=resolved,
    )
    agent = SACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=hidden_dims,
        config=SACConfig(
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
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval,
        resume_from=resume_from,
        max_steps_this_call=max_steps_this_call,
    )
    post_return = evaluate_sac_lqr(
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
            run_sac_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
