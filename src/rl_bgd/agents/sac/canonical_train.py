"""Canonical task-aware Continual World SAC training loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from rl_bgd.agents.sac.task_aware_agent import TaskAwareSACAgent
from rl_bgd.envs.continual_world.canonical import CanonicalContinualWorldStreamEnv
from rl_bgd.replay.buffer import ReplayBuffer

StageObserver = Callable[[int, TaskAwareSACAgent], None]


@dataclass(frozen=True)
class CanonicalSACTrainConfig:
    replay_capacity: int = 1_000_000
    batch_size: int = 128
    start_steps_per_task: int = 10_000
    update_after: int = 1_000
    update_every: int = 50
    seed: int = 0
    reset_buffer_on_task_change: bool = True
    reset_optimizer_on_task_change: bool = True

    def validate(self) -> None:
        if self.replay_capacity < self.batch_size or self.batch_size < 1:
            raise ValueError("invalid replay/batch configuration")
        if self.start_steps_per_task < 0 or self.update_after < 0:
            raise ValueError("warm-up/update delay cannot be negative")
        if self.update_every < 1:
            raise ValueError("update_every must be positive")


def train_canonical_sac(
    env: CanonicalContinualWorldStreamEnv,
    agent: TaskAwareSACAgent,
    *,
    config: CanonicalSACTrainConfig,
    stage_observer: StageObserver | None = None,
) -> dict[str, object]:
    config.validate()
    observation_dim = int(env.observation_space.low.numel())
    action_dim = int(env.action_space.low.numel())
    generator = torch.Generator(device=agent.device).manual_seed(config.seed + 17)

    def new_replay() -> ReplayBuffer:
        return ReplayBuffer(
            config.replay_capacity,
            observation_dim,
            action_dim,
            storage_device=agent.device,
        )

    replay = new_replay()
    replay_reset_count = 1
    if config.reset_optimizer_on_task_change:
        agent.reset_optimizers()
    observation, _ = env.reset(seed=config.seed)
    current_task_index = env.cur_seq_idx
    task_elapsed = 0
    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}
    stage_replay_sizes: list[int] = []

    for global_step in range(env.total_step_limit):
        if env.cur_seq_idx != current_task_index:
            current_task_index = env.cur_seq_idx
            task_elapsed = 0
            if config.reset_buffer_on_task_change:
                replay = new_replay()
                replay_reset_count += 1
            if config.reset_optimizer_on_task_change:
                agent.reset_optimizers()

        action = (
            env.action_space.sample(generator=generator)
            if task_elapsed < config.start_steps_per_task
            else agent.act(observation, deterministic=False)
        )
        next_observation, reward, terminated, truncated, _ = env.step(action)
        replay.add(
            observation,
            action,
            reward,
            next_observation,
            terminated=terminated,
            truncated=truncated,
            insertion_step=global_step,
        )
        episode_return += reward
        observation = next_observation
        task_elapsed += 1

        if (
            task_elapsed >= config.update_after
            and task_elapsed % config.update_every == 0
            and len(replay) >= config.batch_size
        ):
            for _ in range(config.update_every):
                last_metrics = agent.update(
                    replay.sample(config.batch_size, generator=generator)
                )

        if terminated or truncated:
            completed_returns.append(episode_return)
            episode_return = 0.0
            if global_step + 1 < env.total_step_limit:
                observation, _ = env.reset()

        if (global_step + 1) % env.config.steps_per_task == 0:
            stage_replay_sizes.append(len(replay))
            if stage_observer is not None:
                stage_observer(
                    (global_step + 1) // env.config.steps_per_task,
                    agent,
                )

    return {
        "steps": env.total_step_limit,
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
        "replay_reset_count": replay_reset_count,
        "optimizer_reset_count": agent.optimizer_reset_count,
        "stage_replay_sizes": stage_replay_sizes,
    }
