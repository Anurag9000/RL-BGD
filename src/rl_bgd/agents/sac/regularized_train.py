"""Oracle-boundary training loop for parameter-regularized SAC baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from rl_bgd.agents.sac.regularized_agent import RegularizedSACAgent
from rl_bgd.replay.buffer import ReplayBuffer


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
class BoundaryRegularizedSACTrainConfig:
    total_steps: int
    consolidation_steps: tuple[int, ...]
    random_steps: int = 1_000
    batch_size: int = 256
    replay_capacity: int = 100_000
    importance_batch_size: int = 256
    updates_per_step: int = 1
    seed: int = 0

    def validate(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.random_steps < 0 or self.updates_per_step < 1:
            raise ValueError("invalid SAC warmup/update configuration")
        if min(
            self.batch_size,
            self.replay_capacity,
            self.importance_batch_size,
        ) < 1:
            raise ValueError("replay and batch sizes must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if tuple(sorted(set(self.consolidation_steps))) != self.consolidation_steps:
            raise ValueError("consolidation_steps must be sorted and unique")
        if any(
            step <= 0 or step >= self.total_steps
            for step in self.consolidation_steps
        ):
            raise ValueError("consolidation step lies outside the training stream")


def train_boundary_regularized_sac(
    env: ContinuousEnv,
    agent: RegularizedSACAgent,
    *,
    config: BoundaryRegularizedSACTrainConfig,
) -> dict[str, object]:
    """Train SAC while using true stream boundaries only for consolidation."""

    config.validate()
    observation_dim = int(env.observation_space.low.numel())
    action_dim = int(env.action_space.low.numel())

    def new_replay(capacity: int) -> ReplayBuffer:
        return ReplayBuffer(
            capacity,
            observation_dim,
            action_dim,
            storage_device=agent.device,
        )

    replay = new_replay(config.replay_capacity)
    phase_replay = new_replay(config.replay_capacity)
    train_generator = torch.Generator(device=agent.device).manual_seed(
        config.seed + 17
    )
    consolidation_generator = torch.Generator(device=agent.device).manual_seed(
        config.seed + 9_001
    )
    observation, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}
    consolidation_log: list[dict[str, float]] = []
    boundary_set = set(config.consolidation_steps)

    for step in range(config.total_steps):
        if step < config.random_steps:
            action = env.action_space.sample(generator=train_generator)
        else:
            action = agent.act(
                observation,
                deterministic=False,
            )
        next_observation, reward, terminated, truncated, _ = env.step(action)
        for buffer in (replay, phase_replay):
            buffer.add(
                observation,
                action,
                reward,
                next_observation,
                terminated=terminated,
                truncated=truncated,
                insertion_step=step,
            )
        episode_return += reward
        observation = next_observation

        if terminated or truncated:
            completed_returns.append(episode_return)
            episode_return = 0.0
            observation, _ = env.reset()

        if len(replay) >= config.batch_size and step >= config.random_steps:
            for _ in range(config.updates_per_step):
                batch = replay.sample(
                    config.batch_size,
                    generator=train_generator,
                )
                last_metrics = agent.update(batch)

        completed_steps = step + 1
        if completed_steps in boundary_set:
            if len(phase_replay) < 1:
                raise RuntimeError("boundary reached with no phase evidence")
            importance_batch = phase_replay.sample(
                min(
                    config.importance_batch_size,
                    len(phase_replay),
                ),
                generator=consolidation_generator,
            )
            agent.consolidate_from_batch(importance_batch)
            consolidation_log.append(
                {
                    "environment_step": float(completed_steps),
                    "consolidation_count": float(agent.consolidation_count),
                }
            )
            phase_replay = new_replay(config.replay_capacity)

    return {
        "steps": config.total_steps,
        "episodes": len(completed_returns),
        "mean_episode_return": (
            sum(completed_returns) / len(completed_returns)
            if completed_returns
            else float("nan")
        ),
        "final_10_mean_return": (
            sum(completed_returns[-10:]) / min(10, len(completed_returns))
            if completed_returns
            else float("nan")
        ),
        "last_update_metrics": last_metrics,
        "replay_size": len(replay),
        "consolidations": consolidation_log,
    }
