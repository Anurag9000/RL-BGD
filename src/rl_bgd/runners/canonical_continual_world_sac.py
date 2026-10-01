"""Canonical task-aware Continual World multi-head SAC runner."""

from __future__ import annotations

import json

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.canonical_train import CanonicalSACTrainConfig, train_canonical_sac
from rl_bgd.agents.sac.task_aware_agent import TaskAwareSACAgent
from rl_bgd.envs.continual_world.evaluation import PerformanceMatrixRecorder, evaluate_tasks
from rl_bgd.envs.continual_world.metaworld import make_continual_world_protocol
from rl_bgd.envs.continual_world.stream import ContinualWorldBenchmark
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


def run_canonical_continual_world_sac(
    *,
    benchmark: ContinualWorldBenchmark = "CW10",
    steps_per_task: int = 1_000_000,
    seed: int = 0,
    device: str = "auto",
    evaluation_episodes: int = 5,
    episode_horizon: int = 200,
    hidden_dims: tuple[int, ...] = (256, 256, 256, 256),
    replay_capacity: int = 1_000_000,
    batch_size: int = 128,
    start_steps_per_task: int = 10_000,
    update_after: int = 1_000,
    update_every: int = 50,
) -> dict[str, object]:
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    bundle = make_continual_world_protocol(
        benchmark,
        protocol="canonical",
        steps_per_task=steps_per_task,
        seed=seed,
        device=resolved,
        episode_horizon=episode_horizon,
    )
    agent = TaskAwareSACAgent(
        int(bundle.train_env.observation_space.low.numel()),
        int(bundle.train_env.action_space.low.numel()),
        num_tasks=len(bundle.task_names),
        action_low=bundle.train_env.action_space.low,
        action_high=bundle.train_env.action_space.high,
        hidden_dims=hidden_dims,
        config=SACConfig(actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3),
        device=resolved,
    )
    return_matrix = PerformanceMatrixRecorder(bundle.task_names)
    success_matrix = PerformanceMatrixRecorder(bundle.task_names)

    def stage_observer(completed_stages: int, current_agent: TaskAwareSACAgent) -> None:
        stage_index = completed_stages - 1
        evaluations = evaluate_tasks(
            current_agent,
            bundle.evaluation_envs,
            episodes=evaluation_episodes,
            seed=70_000 + stage_index * 100_000,
            max_episode_steps=episode_horizon,
        )
        stage_label = f"after_{completed_stages:02d}_{bundle.task_names[stage_index]}"
        return_matrix.append(stage_label, [result.mean_return for result in evaluations])
        success_scores: list[float] = []
        for result in evaluations:
            if result.success_rate is None:
                raise RuntimeError("Continual World evaluation did not expose success")
            success_scores.append(result.success_rate)
        success_matrix.append(stage_label, success_scores)

    training = train_canonical_sac(
        bundle.train_env,
        agent,
        config=CanonicalSACTrainConfig(
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            start_steps_per_task=start_steps_per_task,
            update_after=update_after,
            update_every=update_every,
            seed=seed,
        ),
        stage_observer=stage_observer,
    )
    bundle.train_env.close()
    for env in bundle.evaluation_envs:
        if hasattr(env, "close"):
            env.close()
    return {
        "benchmark": benchmark,
        "protocol": "canonical",
        "optimizer": "adam",
        "seed": seed,
        "steps_per_task": steps_per_task,
        "task_names": list(bundle.task_names),
        "training": training,
        "return_stage_labels": list(return_matrix.stage_labels),
        "return_matrix": return_matrix.matrix.tolist(),
        "success_stage_labels": list(success_matrix.stage_labels),
        "success_matrix": success_matrix.matrix.tolist(),
        "return_summary": return_matrix.summary(),
        "success_summary": success_matrix.summary(),
    }


def main() -> None:
    print(json.dumps(run_canonical_continual_world_sac(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
