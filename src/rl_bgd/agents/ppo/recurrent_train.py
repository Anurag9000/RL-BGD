"""Collection and evaluation loops for recurrent PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from rl_bgd.agents.ppo.recurrent_agent import (
    RecurrentPPOAgent,
)
from rl_bgd.agents.ppo.recurrent_rollout import (
    RecurrentRolloutBuffer,
)


class ContinuousEnv(Protocol):
    action_space: object
    observation_space: object

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[
        Tensor,
        dict[str, object],
    ]: ...

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
class RecurrentPPOTrainConfig:
    total_steps: int = 10_000
    rollout_steps: int = 256
    seed: int = 0


def train_recurrent_ppo(
    env: ContinuousEnv,
    agent: RecurrentPPOAgent,
    *,
    config: RecurrentPPOTrainConfig,
) -> dict[str, object]:
    if config.total_steps < 1 or config.rollout_steps < 1:
        raise ValueError("recurrent PPO training budgets must be positive")
    observation, _ = env.reset(seed=config.seed)
    agent.reset_recurrent_state()
    episode_start = True
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
        rollout = RecurrentRolloutBuffer(
            horizon,
            int(env.observation_space.low.numel()),
            int(env.action_space.low.numel()),
            agent.recurrent_config.recurrent_hidden_dim,
            agent.recurrent_config.recurrent_hidden_dim,
            device=agent.device,
        )
        last_transition_done = False
        for _ in range(horizon):
            action_step = agent.sample_action_recurrent(observation)
            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action_step.action)
            next_value = agent.value_from_hidden(
                next_observation,
                action_step.value_hidden_after,
            )
            rollout.add(
                observation,
                action_step.action,
                reward,
                terminated=terminated,
                truncated=truncated,
                value=action_step.value,
                next_value=next_value,
                log_prob=action_step.log_prob,
                episode_start=episode_start,
                actor_hidden=action_step.actor_hidden_before,
                value_hidden=action_step.value_hidden_before,
            )
            episode_return += reward
            observation = next_observation
            steps += 1
            last_transition_done = terminated or truncated
            if last_transition_done:
                completed_returns.append(episode_return)
                episode_return = 0.0
                observation, _ = env.reset()
                agent.reset_recurrent_state()
                episode_start = True
            else:
                episode_start = False

        last_metrics = agent.update(rollout)
        if not last_transition_done:
            agent.refresh_recurrent_state(rollout)
        rollout_index += 1

    return {
        "steps": steps,
        "rollouts": rollout_index,
        "episodes": len(completed_returns),
        "mean_episode_return": (
            sum(completed_returns) / len(completed_returns) if completed_returns else float("nan")
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
        "recurrent_reset_count": agent.recurrent_reset_count,
    }


@torch.no_grad()
def evaluate_recurrent_ppo(
    env: ContinuousEnv,
    agent: RecurrentPPOAgent,
    *,
    episodes: int = 5,
    seed: int = 20_000,
) -> float:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    returns: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        agent.reset_recurrent_state()
        episode_return = 0.0
        while True:
            action = agent.act_recurrent(
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
            episode_return += reward
            if terminated or truncated:
                break
        returns.append(episode_return)
    return sum(returns) / len(returns)
