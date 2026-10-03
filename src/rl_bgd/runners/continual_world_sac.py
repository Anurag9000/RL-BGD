"""Strict task-agnostic Continual World SAC/BGD-SAC experiment runner."""

from __future__ import annotations

import json
from typing import Literal

from rl_bgd.agents.sac.agent import SACAgent, SACConfig
from rl_bgd.agents.sac.bgd_agent import BGDSACAgent, BGDSACConfig
from rl_bgd.agents.sac.regularized_agent import (
    RegularizationMethod,
    RegularizationTarget,
    RegularizedSACAgent,
    RegularizedSACConfig,
)
from rl_bgd.agents.sac.train import SACTrainConfig, train_sac
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.envs.continual_world.evaluation import (
    PerformanceMatrixRecorder,
    evaluate_tasks,
    repeated_sequence_recurrence_summary,
)
from rl_bgd.envs.continual_world.metaworld import (
    make_continual_world_protocol,
)
from rl_bgd.envs.continual_world.stream import ContinualWorldBenchmark
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import preserved_random_state, seed_everything

OptimizerFamily = Literal[
    "adam",
    "bgd",
    "ewc",
    "online_ewc",
    "si",
    "mas",
]


def run_ta_continual_world_sac(
    *,
    benchmark: ContinualWorldBenchmark = "CW10",
    optimizer: OptimizerFamily = "adam",
    steps_per_task: int = 1_000_000,
    seed: int = 0,
    device: str = "auto",
    evaluation_episodes: int = 5,
    episode_horizon: int = 200,
    hidden_dims: tuple[int, ...] = (256, 256, 256, 256),
    replay_capacity: int = 1_000_000,
    batch_size: int = 128,
    random_steps: int = 10_000,
    bayesianization: str = "critic_only",
    consolidation_interval_updates: int = 50_000,
    regularization_strength: float = 0.1,
    regularization_target: RegularizationTarget = "actor_and_critic",
    importance_samples: int = 4,
) -> dict[str, object]:
    """Run TA-CW10/TA-CW20 while keeping stage knowledge evaluator-only."""

    if steps_per_task < 1 or evaluation_episodes < 1:
        raise ValueError("Continual World budgets must be positive")
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    bundle = make_continual_world_protocol(
        benchmark,
        protocol="task_agnostic",
        steps_per_task=steps_per_task,
        seed=seed,
        device=resolved,
        episode_horizon=episode_horizon,
    )
    try:
        observation_dim = int(bundle.train_env.observation_space.low.numel())
        action_dim = int(bundle.train_env.action_space.low.numel())
        sac_config = SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        )

        if optimizer == "adam":
            agent: SACAgent = SACAgent(
                observation_dim,
                action_dim,
                action_low=bundle.train_env.action_space.low,
                action_high=bundle.train_env.action_space.high,
                hidden_dims=hidden_dims,
                config=sac_config,
                device=resolved,
            )
        elif optimizer == "bgd":
            agent = BGDSACAgent(
                observation_dim,
                action_dim,
                action_low=bundle.train_env.action_space.low,
                action_high=bundle.train_env.action_space.high,
                hidden_dims=hidden_dims,
                sac_config=sac_config,
                bgd_config=BGDSACConfig(
                    bayesianization=bayesianization,  # type: ignore[arg-type]
                    posterior_std=0.1,
                    actor_bgd=BGDConfig(
                        eta=0.1,
                        mc_samples=4,
                        antithetic=True,
                    ),
                    critic_bgd=BGDConfig(
                        eta=0.1,
                        mc_samples=4,
                        antithetic=True,
                    ),
                ),
                device=resolved,
            )
        elif optimizer in {
            "ewc",
            "online_ewc",
            "si",
            "mas",
        }:
            methods: dict[
                str,
                RegularizationMethod,
            ] = {
                "ewc": "ewc",
                "online_ewc": "online_ewc",
                "si": "si",
                "mas": "mas",
            }
            agent = RegularizedSACAgent(
                observation_dim,
                action_dim,
                action_low=bundle.train_env.action_space.low,
                action_high=bundle.train_env.action_space.high,
                hidden_dims=hidden_dims,
                sac_config=sac_config,
                regularization_config=RegularizedSACConfig(
                    method=methods[optimizer],
                    target=regularization_target,
                    strength=regularization_strength,
                    consolidation_interval_updates=(consolidation_interval_updates),
                    importance_samples=importance_samples,
                    online_ewc_decay=0.95,
                    si_damping=0.1,
                ),
                device=resolved,
            )
        else:
            raise ValueError(f"unsupported optimizer family: {optimizer}")

        return_matrix = PerformanceMatrixRecorder(bundle.task_names)
        success_matrix = PerformanceMatrixRecorder(bundle.task_names)

        def evaluation_observer(
            completed_steps: int,
            current_agent: SACAgent,
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
        training = train_sac(
            bundle.train_env,
            agent,
            config=SACTrainConfig(
                total_steps=total_steps,
                random_steps=min(random_steps, total_steps),
                batch_size=batch_size,
                replay_capacity=max(replay_capacity, batch_size),
                updates_per_step=1,
                seed=seed,
            ),
            post_step_observer=evaluation_observer,
        )
        recurrence_summary = (
            repeated_sequence_recurrence_summary(
                bundle.task_names,
                success_matrix.matrix.tolist(),
            )
            if benchmark == "CW20"
            else {}
        )

        return {
            "benchmark": benchmark,
            "protocol": "task_agnostic",
            "optimizer": optimizer,
            "seed": seed,
            "steps_per_task": steps_per_task,
            "total_steps": total_steps,
            "task_names": list(bundle.task_names),
            "training": training,
            "return_stage_labels": list(return_matrix.stage_labels),
            "return_matrix": return_matrix.matrix.tolist(),
            "success_stage_labels": list(success_matrix.stage_labels),
            "success_matrix": success_matrix.matrix.tolist(),
            "return_summary": return_matrix.summary(),
            "success_summary": success_matrix.summary(),
            "recurrence_summary": recurrence_summary,
            "information_access": {
                "receives_task_id": False,
                "receives_task_boundary": False,
                "receives_environment_context": False,
                "task_specific_heads": False,
                "replay_reset_on_task_change": False,
                "optimizer_reset_on_task_change": False,
                "consolidation_trigger": (
                    "fixed_optimizer_update_interval"
                    if optimizer
                    in {
                        "ewc",
                        "online_ewc",
                        "si",
                        "mas",
                    }
                    else "none"
                ),
                "consolidation_interval_updates": (
                    consolidation_interval_updates
                    if optimizer
                    in {
                        "ewc",
                        "online_ewc",
                        "si",
                        "mas",
                    }
                    else None
                ),
            },
        }

    finally:
        bundle.close()


def main() -> None:
    print(
        json.dumps(
            run_ta_continual_world_sac(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
