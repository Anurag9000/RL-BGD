"""Small inspectable SAC collection/training loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import torch
from torch import Tensor

from rl_bgd.agents.sac.agent import SACAgent
from rl_bgd.envs.protocols import ContinuousTensorEnv
from rl_bgd.replay.buffer import ReplayBuffer

UpdateObserver = Callable[[int, dict[str, float]], None]
EpisodeObserver = Callable[[int, float], None]
PostStepObserver = Callable[[int, SACAgent], None]


@dataclass(frozen=True)
class SACTrainConfig:
    total_steps: int = 10_000
    random_steps: int = 1_000
    batch_size: int = 256
    replay_capacity: int = 100_000
    updates_per_step: int = 1
    seed: int = 0


def train_sac(
    env: ContinuousTensorEnv,
    agent: SACAgent,
    *,
    config: SACTrainConfig,
    update_observer: UpdateObserver | None = None,
    episode_observer: EpisodeObserver | None = None,
    post_step_observer: PostStepObserver | None = None,
) -> dict[str, object]:
    if (
        config.total_steps < 1
        or config.batch_size < 1
        or config.replay_capacity < config.batch_size
    ):
        raise ValueError("invalid SAC training budget/replay configuration")
    if config.random_steps < 0 or config.updates_per_step < 1:
        raise ValueError("invalid SAC warmup/update configuration")
    action_dim = int(env.action_space.low.numel())
    observation_dim = int(env.observation_space.low.numel())
    replay = ReplayBuffer(
        config.replay_capacity,
        observation_dim,
        action_dim,
        storage_device=agent.device,
    )
    generator = torch.Generator(device=agent.device).manual_seed(config.seed + 17)
    observation, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}
    for step in range(config.total_steps):
        if step < config.random_steps:
            action = env.action_space.sample(generator=generator)
        else:
            action = agent.act(
                observation,
                deterministic=False,
            )
        (
            next_observation,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action)
        replay.add(
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
            if episode_observer is not None:
                episode_observer(
                    step,
                    episode_return,
                )
            episode_return = 0.0
            observation, _ = env.reset()

        if len(replay) >= config.batch_size and step >= config.random_steps:
            for _ in range(config.updates_per_step):
                batch = replay.sample(
                    config.batch_size,
                    generator=generator,
                )
                last_metrics = agent.update(batch)
                if update_observer is not None:
                    update_observer(
                        step,
                        dict(last_metrics),
                    )

        if post_step_observer is not None:
            post_step_observer(
                step + 1,
                agent,
            )

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
    }
