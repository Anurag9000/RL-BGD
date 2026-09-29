"""Small inspectable PPO collection/training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from rl_bgd.agents.ppo.rollout import RolloutBuffer


class PPOTrainAgent(Protocol):
    device: torch.device
    config: object

    def sample_action(
        self,
        observation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def value_of(
        self,
        observation: Tensor,
    ) -> Tensor: ...

    def update(
        self,
        rollout: RolloutBuffer,
    ) -> dict[str, float]: ...

    def act(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor: ...


class ContinuousEnv(Protocol):
    action_space: object
    observation_space: object

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Tensor, dict[str, object]]: ...

    def step(
        self,
        action: Tensor,
    ) -> tuple[
        Tensor,
        float,
        bool,
        bool,
        dict[str, object],
    ]: ...


@dataclass(frozen=True)
class PPOTrainConfig:
    total_steps: int = 10_000
    rollout_steps: int = 1024
    seed: int = 0


def train_ppo(
    env: ContinuousEnv,
    agent: PPOTrainAgent,
    *,
    config: PPOTrainConfig,
) -> dict[str, object]:
    if config.total_steps < 1 or config.rollout_steps < 1:
        raise ValueError(
            "PPO training budgets must be positive"
        )

    observation, _ = env.reset(
        seed=config.seed
    )
    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}
    steps = 0
    rollout_index = 0

    while steps < config.total_steps:
        horizon = min(
            config.rollout_steps,
            config.total_steps - steps,
        )
        observation_dim = int(
            env.observation_space.low.numel()
        )
        action_dim = int(
            env.action_space.low.numel()
        )
        rollout = RolloutBuffer(
            horizon,
            observation_dim,
            action_dim,
            device=agent.device,
        )

        for _ in range(horizon):
            (
                action,
                log_prob,
                value,
            ) = agent.sample_action(observation)
            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)
            next_value = agent.value_of(
                next_observation
            )
            rollout.add(
                observation,
                action,
                reward,
                terminated=terminated,
                truncated=truncated,
                value=value,
                next_value=next_value,
                log_prob=log_prob,
            )
            episode_return += reward
            observation = next_observation
            steps += 1

            if terminated or truncated:
                completed_returns.append(
                    episode_return
                )
                episode_return = 0.0
                observation, _ = env.reset()

        last_metrics = agent.update(rollout)
        rollout_index += 1

    return {
        "steps": steps,
        "rollouts": rollout_index,
        "episodes": len(completed_returns),
        "mean_episode_return": (
            sum(completed_returns)
            / len(completed_returns)
            if completed_returns
            else float("nan")
        ),
        "final_10_mean_return": (
            sum(completed_returns[-10:])
            / min(
                10,
                len(completed_returns),
            )
            if completed_returns
            else float("nan")
        ),
        "last_update_metrics": last_metrics,
    }


@torch.no_grad()
def evaluate_ppo(
    env: ContinuousEnv,
    agent: PPOTrainAgent,
    *,
    episodes: int = 5,
    seed: int = 10_000,
) -> float:
    if episodes < 1:
        raise ValueError("episodes must be positive")

    returns: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(
            seed=seed + episode
        )
        episode_return = 0.0
        while True:
            observation, reward, terminated, truncated, _ = env.step(
                agent.act(
                    observation,
                    deterministic=True,
                )
            )
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(episode_return)
    return sum(returns) / len(returns)
