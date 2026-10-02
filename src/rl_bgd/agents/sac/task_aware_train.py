"""Canonical task-aware SAC training semantics for Continual World."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import torch
from torch import Tensor

from rl_bgd.agents.sac.task_aware_agent import TaskAwareSACAgent
from rl_bgd.envs.protocols import TaskAwareTensorEnv
from rl_bgd.replay.buffer import ReplayBuffer

StageObserver = Callable[[int, TaskAwareSACAgent], None]
UpdateObserver = Callable[[int, dict[str, float]], None]


@dataclass(frozen=True)
class CanonicalSACTrainConfig:
    """Published Continual World SAC lifecycle defaults."""

    total_steps: int
    start_steps: int = 10_000
    update_after: int = 1_000
    update_every: int = 50
    batch_size: int = 128
    replay_capacity: int = 1_000_000
    seed: int = 0
    reset_buffer_on_task_change: bool = True
    reset_optimizer_on_task_change: bool = True
    agent_policy_exploration: bool = False

    def validate(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.start_steps < 0 or self.update_after < 0:
            raise ValueError("exploration/update delays must be non-negative")
        if self.update_every < 1 or self.batch_size < 1:
            raise ValueError("update_every and batch_size must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")


def train_canonical_task_aware_sac(
    env: TaskAwareTensorEnv,
    agent: TaskAwareSACAgent,
    *,
    config: CanonicalSACTrainConfig,
    stage_observer: StageObserver | None = None,
    update_observer: UpdateObserver | None = None,
) -> dict[str, object]:
    """Train canonical CW SAC with explicit task-aware lifecycle controls."""

    config.validate()
    observation_dim = int(env.observation_space.low.numel())
    action_dim = int(env.action_space.low.numel())

    def new_replay() -> ReplayBuffer:
        return ReplayBuffer(
            config.replay_capacity,
            observation_dim,
            action_dim,
            storage_device=agent.device,
        )

    replay = new_replay()
    generator = torch.Generator(device=agent.device).manual_seed(config.seed + 31)
    observation, _ = env.reset(seed=config.seed)
    current_task_index = -1
    current_task_timestep = 0
    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}

    for global_timestep in range(config.total_steps):
        if current_task_index != env.cur_seq_idx:
            current_task_index = env.cur_seq_idx
            current_task_timestep = 0
            if config.reset_buffer_on_task_change:
                replay = new_replay()
            if config.reset_optimizer_on_task_change:
                agent.reset_optimizers()

        use_policy = current_task_timestep > config.start_steps or (
            config.agent_policy_exploration and current_task_index > 0
        )
        if use_policy:
            action = agent.act(observation, deterministic=False)
        else:
            action = env.action_space.sample(generator=generator)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        replay.add(
            observation,
            action,
            reward,
            next_observation,
            terminated=terminated,
            truncated=truncated,
            insertion_step=global_timestep,
        )
        episode_return += reward
        observation = next_observation

        if terminated or truncated:
            completed_returns.append(episode_return)
            episode_return = 0.0
            if global_timestep + 1 < config.total_steps:
                observation, _ = env.reset()

        if (
            current_task_timestep >= config.update_after
            and current_task_timestep % config.update_every == 0
        ):
            if len(replay) < config.batch_size:
                raise RuntimeError(
                    "canonical SAC update scheduled before replay reached batch_size"
                )
            for _ in range(config.update_every):
                batch = replay.sample(
                    config.batch_size,
                    generator=generator,
                )
                last_metrics = agent.update(batch)
                if update_observer is not None:
                    update_observer(global_timestep, dict(last_metrics))

        if stage_observer is not None:
            stage_observer(global_timestep + 1, agent)
        current_task_timestep += 1

    return {
        "steps": config.total_steps,
        "episodes": len(completed_returns),
        "mean_episode_return": (
            sum(completed_returns) / len(completed_returns) if completed_returns else float("nan")
        ),
        "final_10_mean_return": (
            sum(completed_returns[-10:]) / min(10, len(completed_returns))
            if completed_returns
            else float("nan")
        ),
        "last_update_metrics": last_metrics,
        "replay_size": len(replay),
        "optimizer_resets": agent.optimizer_reset_count,
    }
