"""Training and evaluation loops for recurrent sequence-replay SAC."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import torch
from torch import Tensor

from rl_bgd.agents.sac.recurrent_agent import RecurrentSACAgent
from rl_bgd.envs.protocols import ContinuousTensorEnv
from rl_bgd.replay.sequence_buffer import SequenceReplayBuffer

PostStepObserver = Callable[[int, RecurrentSACAgent], None]


@dataclass(frozen=True)
class RecurrentSACTrainConfig:
    total_steps: int = 10_000
    random_steps: int = 1_000
    sequence_batch_size: int = 32
    burn_in: int = 8
    unroll: int = 16
    replay_capacity: int = 100_000
    updates_per_step: int = 1
    seed: int = 0

    def validate(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.random_steps < 0:
            raise ValueError("random_steps must be non-negative")
        if (
            self.sequence_batch_size < 1
            or self.burn_in < 0
            or self.unroll < 1
            or self.updates_per_step < 1
        ):
            raise ValueError("invalid recurrent SAC sequence/update configuration")
        if self.replay_capacity < self.burn_in + self.unroll:
            raise ValueError("replay capacity is shorter than one sequence window")


def train_recurrent_sac(
    env: ContinuousTensorEnv,
    agent: RecurrentSACAgent,
    *,
    config: RecurrentSACTrainConfig,
    post_step_observer: PostStepObserver | None = None,
) -> dict[str, object]:
    config.validate()
    observation_dim = int(env.observation_space.low.numel())
    action_dim = int(env.action_space.low.numel())
    replay = SequenceReplayBuffer(
        config.replay_capacity,
        observation_dim,
        action_dim,
        storage_device=agent.device,
    )
    generator = torch.Generator(device=agent.device).manual_seed(config.seed + 27)
    observation, _ = env.reset(seed=config.seed)
    agent.reset_recurrent_state()
    episode_start = True
    episode_history: list[Tensor] = []
    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}

    window = config.burn_in + config.unroll
    required_size = window + config.sequence_batch_size - 1

    for step in range(config.total_steps):
        episode_history.append(observation.detach().clone())
        if step < config.random_steps:
            agent.advance_actor_hidden(observation)
            action = env.action_space.sample(generator=generator)
        else:
            action = agent.act_recurrent(
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
            episode_start=episode_start,
            insertion_step=step,
        )
        episode_return += reward
        observation = next_observation
        episode_done = terminated or truncated

        if episode_done:
            completed_returns.append(episode_return)
            episode_return = 0.0
            observation, _ = env.reset()
            agent.reset_recurrent_state()
            episode_history = []
            episode_start = True
        else:
            episode_start = False

        updated = False
        if len(replay) >= required_size and step >= config.random_steps:
            for _ in range(config.updates_per_step):
                batch = replay.sample_sequences(
                    config.sequence_batch_size,
                    burn_in=config.burn_in,
                    unroll=config.unroll,
                    generator=generator,
                )
                last_metrics = agent.update(batch)
                updated = True

        if updated and not episode_done:
            agent.rebuild_actor_hidden(episode_history)

        if post_step_observer is not None:
            post_step_observer(step + 1, agent)

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
        "recurrent_reset_count": agent.recurrent_reset_count,
    }


@torch.no_grad()
def evaluate_recurrent_sac(
    env: ContinuousTensorEnv,
    agent: RecurrentSACAgent,
    *,
    episodes: int = 5,
    seed: int = 30_000,
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
