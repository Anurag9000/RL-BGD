"""Oracle-boundary training loop for parameter-regularized SAC baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rl_bgd.agents.sac.regularized_agent import RegularizedSACAgent
from rl_bgd.envs.protocols import ContinuousTensorEnv
from rl_bgd.replay.buffer import ReplayBuffer
from rl_bgd.utils.checkpoint_io import save_training_checkpoint as _save_training_checkpoint
from rl_bgd.utils.checkpoint_payload import (
    checkpoint_finite_float,
    checkpoint_float_list,
    checkpoint_float_mapping,
    checkpoint_generator_state,
    checkpoint_observation,
)
from rl_bgd.utils.checkpoint_progress import checkpoint_integer, checkpoint_step
from rl_bgd.utils.checkpoint_transaction import transactional_state_load
from rl_bgd.utils.config_validation import (
    config_nonnegative_integer,
    config_positive_integer,
)
from rl_bgd.utils.randomness import (
    load_random_state_dict,
    preserved_random_state,
    random_state_dict,
)

_BOUNDARY_REGULARIZED_SAC_CHECKPOINT_VERSION = 1


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
        config_positive_integer(self.total_steps, name="total_steps")
        config_nonnegative_integer(self.random_steps, name="random_steps")
        config_positive_integer(self.batch_size, name="batch_size")
        config_positive_integer(self.replay_capacity, name="replay_capacity")
        config_positive_integer(self.importance_batch_size, name="importance_batch_size")
        config_positive_integer(self.updates_per_step, name="updates_per_step")
        config_nonnegative_integer(self.seed, name="seed")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        for index, step in enumerate(self.consolidation_steps):
            config_positive_integer(step, name=f"consolidation_steps[{index}]")
        if tuple(sorted(set(self.consolidation_steps))) != self.consolidation_steps:
            raise ValueError("consolidation_steps must be sorted and unique")
        if any(step >= self.total_steps for step in self.consolidation_steps):
            raise ValueError("consolidation step lies outside the training stream")


def _environment_state_dict(env: ContinuousTensorEnv) -> dict[str, Any]:
    state_fn = getattr(env, "state_dict", None)
    if not callable(state_fn):
        raise TypeError("environment does not support checkpointing")
    state = state_fn()
    if not isinstance(state, dict):
        raise TypeError("environment state_dict must return a dictionary")
    return state


def _load_environment_state(
    env: ContinuousTensorEnv,
    state: object,
) -> None:
    load_fn = getattr(env, "load_state_dict", None)
    if not callable(load_fn):
        raise TypeError("environment does not support checkpoint restore")
    if not isinstance(state, dict):
        raise TypeError("environment checkpoint state must be a dictionary")
    load_fn(state)


def _load_training_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise TypeError("boundary-regularized SAC checkpoint must contain a dictionary")
    return payload


def _phase_transition_count(
    completed_steps: int,
    consolidation_steps: tuple[int, ...],
) -> int:
    latest_boundary = 0
    for boundary in consolidation_steps:
        if boundary > completed_steps:
            break
        latest_boundary = boundary
    return completed_steps - latest_boundary


def _validate_replay_checkpoint_progress(
    state: dict[str, Any],
    *,
    name: str,
    transition_count: int,
    first_global_step: int,
    capacity: int,
) -> None:
    """Bind saved replay provenance to the completed training steps."""

    size = checkpoint_integer(state.get("size"), name=f"{name} size")
    if size != min(transition_count, capacity):
        raise ValueError(f"{name} size disagrees with checkpoint progress")

    ids = state.get("transition_ids")
    insertion_steps = state.get("insertion_steps")
    if not isinstance(ids, torch.Tensor) or not isinstance(insertion_steps, torch.Tensor):
        raise TypeError(f"{name} transition IDs/insertion steps must be tensors")
    if ids.shape != (size, 1) or insertion_steps.shape != (size, 1):
        raise ValueError(f"{name} transition IDs/insertion steps shape mismatch")
    if ids.dtype != torch.int64 or insertion_steps.dtype != torch.int64:
        raise ValueError(f"{name} transition IDs/insertion steps dtype mismatch")
    if not torch.equal(insertion_steps, ids + first_global_step):
        raise ValueError(f"{name} insertion-step chronology mismatch")


def _checkpoint_consolidation_log(
    value: object,
    *,
    completed_steps: int,
    consolidation_steps: tuple[int, ...],
) -> list[dict[str, float]]:
    if not isinstance(value, list):
        raise TypeError("boundary-regularized SAC consolidation_log must be a list")
    expected_steps = [boundary for boundary in consolidation_steps if boundary <= completed_steps]
    if len(value) != len(expected_steps):
        raise ValueError(
            "boundary-regularized SAC consolidation log disagrees with checkpoint progress"
        )

    result: list[dict[str, float]] = []
    previous_count = 0
    for index, (entry, expected_step) in enumerate(zip(value, expected_steps, strict=True)):
        metrics = checkpoint_float_mapping(
            entry,
            name=f"boundary-regularized SAC consolidation_log[{index}]",
        )
        if set(metrics) != {"environment_step", "consolidation_count"}:
            raise ValueError("boundary-regularized SAC consolidation log fields are invalid")
        environment_step = metrics["environment_step"]
        consolidation_count = metrics["consolidation_count"]
        if not environment_step.is_integer() or int(environment_step) != expected_step:
            raise ValueError("boundary-regularized SAC consolidation log step is inconsistent")
        if (
            not consolidation_count.is_integer()
            or int(consolidation_count) < 1
            or int(consolidation_count) <= previous_count
        ):
            raise ValueError("boundary-regularized SAC consolidation count is inconsistent")
        previous_count = int(consolidation_count)
        result.append(metrics)
    return result


def train_boundary_regularized_sac(
    env: ContinuousTensorEnv,
    agent: RegularizedSACAgent,
    *,
    config: BoundaryRegularizedSACTrainConfig,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int | None = None,
    resume_from: str | Path | None = None,
    max_steps_this_call: int | None = None,
) -> dict[str, object]:
    """Train SAC while using true stream boundaries only for consolidation."""

    config.validate()
    if checkpoint_interval is not None:
        config_positive_integer(checkpoint_interval, name="checkpoint_interval")
    if checkpoint_interval is not None and checkpoint_path is None:
        raise ValueError("checkpoint_interval requires checkpoint_path")
    if max_steps_this_call is not None:
        config_positive_integer(max_steps_this_call, name="max_steps_this_call")

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
    train_generator = torch.Generator(device=agent.device).manual_seed(config.seed + 17)
    consolidation_generator = torch.Generator(device=agent.device).manual_seed(config.seed + 9_001)

    start_step = 0
    episode_return = 0.0
    completed_returns: list[float] = []
    last_metrics: dict[str, float] = {}
    consolidation_log: list[dict[str, float]] = []
    boundary_set = set(config.consolidation_steps)

    if resume_from is None:
        observation, _ = env.reset(seed=config.seed)
    else:
        checkpoint = _load_training_checkpoint(
            resume_from,
            device=agent.device,
        )
        version = checkpoint_integer(
            checkpoint.get("version"),
            name="boundary-regularized SAC checkpoint version",
        )
        if version != _BOUNDARY_REGULARIZED_SAC_CHECKPOINT_VERSION:
            raise ValueError("unsupported boundary-regularized SAC checkpoint version")
        if checkpoint.get("train_config") != asdict(config):
            raise ValueError("boundary-regularized SAC checkpoint configuration mismatch")
        start_step = checkpoint_step(
            checkpoint.get("next_step"),
            name="boundary-regularized SAC checkpoint next_step",
            limit=config.total_steps,
        )

        agent_state = checkpoint.get("agent")
        replay_state = checkpoint.get("replay")
        phase_replay_state = checkpoint.get("phase_replay")
        environment_state = checkpoint.get("environment")
        if not isinstance(agent_state, dict):
            raise TypeError("boundary-regularized SAC agent checkpoint must be a dictionary")
        if not isinstance(replay_state, dict) or not isinstance(phase_replay_state, dict):
            raise TypeError("boundary-regularized SAC replay checkpoints must be dictionaries")
        if not isinstance(environment_state, dict):
            raise TypeError("boundary-regularized SAC environment checkpoint must be a dictionary")

        replay_transition_id = checkpoint_integer(
            replay_state.get("next_transition_id"),
            name="boundary-regularized SAC replay next_transition_id",
        )
        if replay_transition_id != start_step:
            raise ValueError("boundary-regularized SAC replay/step progress mismatch")
        phase_transition_id = checkpoint_integer(
            phase_replay_state.get("next_transition_id"),
            name="boundary-regularized SAC phase replay next_transition_id",
        )
        expected_phase_transitions = _phase_transition_count(
            start_step,
            config.consolidation_steps,
        )
        if phase_transition_id != expected_phase_transitions:
            raise ValueError("boundary-regularized SAC phase replay/step progress mismatch")

        _validate_replay_checkpoint_progress(
            replay_state,
            name="boundary-regularized SAC replay",
            transition_count=start_step,
            first_global_step=0,
            capacity=config.replay_capacity,
        )
        _validate_replay_checkpoint_progress(
            phase_replay_state,
            name="boundary-regularized SAC phase replay",
            transition_count=expected_phase_transitions,
            first_global_step=start_step - expected_phase_transitions,
            capacity=config.replay_capacity,
        )

        observation = checkpoint_observation(
            checkpoint.get("observation"),
            name="boundary-regularized SAC checkpoint observation",
            device=agent.device,
            expected_shape=env.observation_space.shape,
        )
        episode_return = checkpoint_finite_float(
            checkpoint.get("episode_return"),
            name="boundary-regularized SAC checkpoint episode_return",
        )
        completed_returns = checkpoint_float_list(
            checkpoint.get("completed_returns"),
            name="boundary-regularized SAC checkpoint completed_returns",
        )
        last_metrics = checkpoint_float_mapping(
            checkpoint.get("last_metrics"),
            name="boundary-regularized SAC checkpoint last_metrics",
        )
        consolidation_log = _checkpoint_consolidation_log(
            checkpoint.get("consolidation_log"),
            completed_steps=start_step,
            consolidation_steps=config.consolidation_steps,
        )
        train_generator_state = checkpoint_generator_state(
            checkpoint.get("train_generator_state"),
            name="boundary-regularized SAC train generator checkpoint",
            device=agent.device,
        )
        consolidation_generator_state = checkpoint_generator_state(
            checkpoint.get("consolidation_generator_state"),
            name="boundary-regularized SAC consolidation generator checkpoint",
            device=agent.device,
        )
        process_rng = checkpoint.get("process_rng")
        if not isinstance(process_rng, dict):
            raise TypeError("boundary-regularized SAC process RNG checkpoint must be a dictionary")
        with preserved_random_state():
            load_random_state_dict(process_rng)

        resume_state = {
            "agent": agent_state,
            "replay": replay_state,
            "phase_replay": phase_replay_state,
            "environment": environment_state,
            "train_generator_state": train_generator_state,
            "consolidation_generator_state": consolidation_generator_state,
            "process_rng": process_rng,
        }

        def current_resume_state() -> dict[str, Any]:
            return {
                "agent": agent.state_dict(),
                "replay": replay.state_dict(),
                "phase_replay": phase_replay.state_dict(),
                "environment": _environment_state_dict(env),
                "train_generator_state": train_generator.get_state().clone(),
                "consolidation_generator_state": (consolidation_generator.get_state().clone()),
                "process_rng": random_state_dict(),
            }

        def apply_resume(payload: dict[str, Any]) -> None:
            agent.load_state_dict(payload["agent"])
            replay.load_state_dict(payload["replay"])
            phase_replay.load_state_dict(payload["phase_replay"])
            _load_environment_state(env, payload["environment"])
            train_generator.set_state(payload["train_generator_state"].cpu())
            consolidation_generator.set_state(payload["consolidation_generator_state"].cpu())
            load_random_state_dict(payload["process_rng"])

        transactional_state_load(
            resume_state,
            current_state=current_resume_state,
            apply=apply_resume,
        )

    call_end = config.total_steps
    if max_steps_this_call is not None:
        call_end = min(
            config.total_steps,
            start_step + max_steps_this_call,
        )

    def save(next_step: int) -> None:
        if checkpoint_path is None:
            return
        _save_training_checkpoint(
            checkpoint_path,
            {
                "version": _BOUNDARY_REGULARIZED_SAC_CHECKPOINT_VERSION,
                "train_config": asdict(config),
                "next_step": next_step,
                "agent": agent.state_dict(),
                "replay": replay.state_dict(),
                "phase_replay": phase_replay.state_dict(),
                "environment": _environment_state_dict(env),
                "observation": observation.detach().clone(),
                "episode_return": episode_return,
                "completed_returns": list(completed_returns),
                "last_metrics": dict(last_metrics),
                "consolidation_log": [dict(entry) for entry in consolidation_log],
                "train_generator_state": train_generator.get_state().clone(),
                "consolidation_generator_state": (consolidation_generator.get_state().clone()),
                "process_rng": random_state_dict(),
            },
        )

    for step in range(start_step, call_end):
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

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and completed_steps % checkpoint_interval == 0
        ):
            save(completed_steps)

    if checkpoint_path is not None:
        save(call_end)

    return {
        "steps": call_end,
        "target_steps": config.total_steps,
        "completed": call_end >= config.total_steps,
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
        "phase_replay_size": len(phase_replay),
        "consolidations": consolidation_log,
    }
