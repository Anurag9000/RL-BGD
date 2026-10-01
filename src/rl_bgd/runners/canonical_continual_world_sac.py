"""Canonical task-aware Continual World SAC experiment runner."""

from __future__ import annotations

import json
import math

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.task_aware_agent import TaskAwareSACAgent
from rl_bgd.agents.sac.task_aware_train import (
    CanonicalSACTrainConfig,
    train_canonical_task_aware_sac,
)
from rl_bgd.envs.continual_world.evaluation import (
    PerformanceMatrixRecorder,
    evaluate_tasks,
)
from rl_bgd.envs.continual_world.metaworld import make_continual_world_protocol
from rl_bgd.envs.continual_world.stream import ContinualWorldBenchmark
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import preserved_random_state, seed_everything


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
    start_steps: int = 10_000,
    update_after: int = 1_000,
    update_every: int = 50,
    target_output_std: float = 0.089,
    reset_buffer_on_task_change: bool = True,
    reset_optimizer_on_task_change: bool = True,
    agent_policy_exploration: bool = False,
) -> dict[str, object]:
    """Run the published task-aware CW10/CW20 baseline protocol."""

    if steps_per_task < 1 or evaluation_episodes < 1 or target_output_std <= 0:
        raise ValueError("canonical Continual World budgets/scales must be positive")
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
    try:
        observation_dim = int(bundle.train_env.observation_space.low.numel())
        action_dim = int(bundle.train_env.action_space.low.numel())
        target_entropy = action_dim * math.log(
            target_output_std * math.sqrt(2.0 * math.pi * math.e)
        )
        agent = TaskAwareSACAgent(
            observation_dim,
            action_dim,
            num_tasks=len(bundle.task_names),
            action_low=bundle.train_env.action_space.low,
            action_high=bundle.train_env.action_space.high,
            hidden_dims=hidden_dims,
            config=SACConfig(
                gamma=0.99,
                tau=0.005,
                actor_lr=1e-3,
                critic_lr=1e-3,
                alpha_lr=1e-3,
                initial_alpha=math.e,
                automatic_entropy_tuning=True,
                target_entropy=target_entropy,
            ),
            device=resolved,
        )
        return_matrix = PerformanceMatrixRecorder(bundle.task_names)
        success_matrix = PerformanceMatrixRecorder(bundle.task_names)

        def evaluation_observer(
            completed_steps: int,
            current_agent: TaskAwareSACAgent,
        ) -> None:
            if completed_steps % steps_per_task != 0:
                return
            stage_index = completed_steps // steps_per_task - 1
            with preserved_random_state():
                evaluations = evaluate_tasks(
                    current_agent,
                    bundle.evaluation_envs,
                    episodes=evaluation_episodes,
                    seed=50_000 + stage_index * 100_000,
                    max_episode_steps=episode_horizon,
                )
            stage_label = f"after_{stage_index + 1:02d}_{bundle.task_names[stage_index]}"
            return_matrix.append(
                stage_label,
                [result.mean_return for result in evaluations],
            )
            success_scores: list[float] = []
            for result in evaluations:
                if result.success_rate is None:
                    raise RuntimeError("Continual World evaluation did not expose success")
                success_scores.append(result.success_rate)
            success_matrix.append(stage_label, success_scores)

        total_steps = steps_per_task * len(bundle.task_names)
        training = train_canonical_task_aware_sac(
            bundle.train_env,  # type: ignore[arg-type]
            agent,
            config=CanonicalSACTrainConfig(
                total_steps=total_steps,
                start_steps=start_steps,
                update_after=update_after,
                update_every=update_every,
                batch_size=batch_size,
                replay_capacity=max(replay_capacity, batch_size),
                seed=seed,
                reset_buffer_on_task_change=reset_buffer_on_task_change,
                reset_optimizer_on_task_change=reset_optimizer_on_task_change,
                agent_policy_exploration=agent_policy_exploration,
            ),
            stage_observer=evaluation_observer,
        )
        return {
            "benchmark": benchmark,
            "protocol": "canonical",
            "optimizer": "adam",
            "architecture": "multihead",
            "seed": seed,
            "steps_per_task": steps_per_task,
            "total_steps": total_steps,
            "task_names": list(bundle.task_names),
            "reset_buffer_on_task_change": reset_buffer_on_task_change,
            "reset_optimizer_on_task_change": reset_optimizer_on_task_change,
            "reset_critic_on_task_change": False,
            "agent_policy_exploration": agent_policy_exploration,
            "training": training,
            "return_stage_labels": list(return_matrix.stage_labels),
            "return_matrix": return_matrix.matrix.tolist(),
            "success_stage_labels": list(success_matrix.stage_labels),
            "success_matrix": success_matrix.matrix.tolist(),
            "return_summary": return_matrix.summary(),
            "success_summary": success_matrix.summary(),
        }
    finally:
        bundle.close()


def main() -> None:
    print(
        json.dumps(
            run_canonical_continual_world_sac(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
