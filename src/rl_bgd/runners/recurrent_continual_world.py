"""3RL-style recurrent SAC/BGD on strict task-agnostic Continual World."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import torch

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.bgd_agent import BGDSACConfig
from rl_bgd.agents.sac.recurrent_agent import (
    RecurrentSACAgent,
    RecurrentSACConfig,
)
from rl_bgd.agents.sac.recurrent_bgd_agent import BGDRecurrentSACAgent
from rl_bgd.agents.sac.recurrent_train import (
    RecurrentSACTrainConfig,
    train_recurrent_sac,
)
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.envs.continual_world.evaluation import (
    PerformanceMatrixRecorder,
    TaskEvaluation,
    repeated_sequence_recurrence_summary,
)
from rl_bgd.envs.continual_world.metaworld import make_continual_world_protocol
from rl_bgd.envs.continual_world.stream import ContinualWorldBenchmark
from rl_bgd.envs.recurrent_context import PreviousTransitionContextEnv
from rl_bgd.replay.evidence_accounting import ReplayEvidenceConfig
from rl_bgd.surprise.base import RetentionMappingConfig
from rl_bgd.surprise.td import AdaptiveTDRetentionConfig
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import preserved_random_state, seed_everything

RecurrentCWOptimizer = Literal[
    "adam",
    "bgd",
    "adaptive_bgd",
]


@contextmanager
def preserve_recurrent_online_state(
    agent: RecurrentSACAgent,
) -> Iterator[None]:
    """Prevent evaluator-only rollouts from changing training memory state."""

    hidden = agent.actor_hidden.detach().clone()
    reset_count = agent.recurrent_reset_count
    try:
        yield
    finally:
        agent.actor_hidden = hidden
        agent.recurrent_reset_count = reset_count


@torch.no_grad()
def evaluate_recurrent_task(
    agent: RecurrentSACAgent,
    env: PreviousTransitionContextEnv,
    *,
    episodes: int,
    seed: int,
    max_episode_steps: int,
) -> TaskEvaluation:
    """Evaluate one task without exposing task identity to the policy."""

    if episodes < 1 or max_episode_steps < 1:
        raise ValueError("evaluation budgets must be positive")
    returns: list[float] = []
    successes: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        agent.reset_recurrent_state()
        episode_return = 0.0
        episode_success = 0.0
        saw_success = False
        for _ in range(max_episode_steps):
            action = agent.act_recurrent(
                observation,
                deterministic=True,
            )
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)
            episode_return += reward
            if "success" in info:
                success_value = info["success"]
                if isinstance(
                    success_value,
                    bool,
                ) or not isinstance(
                    success_value,
                    (int, float),
                ):
                    raise TypeError("recurrent Continual World success must be numeric")
                saw_success = True
                episode_success = max(
                    episode_success,
                    float(success_value),
                )
            if terminated or truncated:
                break
        returns.append(episode_return)
        if saw_success:
            successes.append(episode_success)

    return TaskEvaluation(
        mean_return=float(sum(returns) / len(returns)),
        success_rate=(float(sum(successes) / len(successes)) if successes else None),
    )


def run_recurrent_ta_continual_world_sac(
    *,
    benchmark: ContinualWorldBenchmark = "CW10",
    optimizer: RecurrentCWOptimizer = "adam",
    steps_per_task: int = 1_000_000,
    seed: int = 0,
    device: str = "auto",
    evaluation_episodes: int = 10,
    episode_horizon: int = 200,
    replay_capacity: int = 10_000_000,
    initial_random_steps: int = 10_000,
    sequence_batch_size: int = 128,
    history_length: int = 15,
    unroll: int = 1,
) -> dict[str, object]:
    """Run a 3RL-style strict task-agnostic CW10/CW20 experiment.

    The runner preserves the task-agnostic information contract and key
    history/replay settings of the reference implementation, but it does not
    claim architectural identity with the original 3RL code.
    """

    if (
        steps_per_task < 1
        or evaluation_episodes < 1
        or history_length < 1
        or unroll < 1
        or sequence_batch_size < 1
    ):
        raise ValueError("invalid recurrent Continual World budget")

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
    train_env = PreviousTransitionContextEnv(bundle.train_env)
    evaluation_envs = tuple(PreviousTransitionContextEnv(env) for env in bundle.evaluation_envs)

    observation_dim = int(train_env.observation_space.low.numel())
    action_dim = int(train_env.action_space.low.numel())
    sac_config = SACConfig(
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gradient_clip_norm=1.0,
    )
    recurrent_config = RecurrentSACConfig(
        recurrent_hidden_dim=30,
        encoder_hidden_dims=(400, 400),
        q_hidden_dims=(400, 400),
    )

    if optimizer == "adam":
        agent: RecurrentSACAgent = RecurrentSACAgent(
            observation_dim,
            action_dim,
            action_low=train_env.action_space.low,
            action_high=train_env.action_space.high,
            sac_config=sac_config,
            recurrent_config=recurrent_config,
            device=resolved,
        )
    elif optimizer in {"bgd", "adaptive_bgd"}:
        agent = BGDRecurrentSACAgent(
            observation_dim,
            action_dim,
            action_low=train_env.action_space.low,
            action_high=train_env.action_space.high,
            sac_config=sac_config,
            recurrent_config=recurrent_config,
            bgd_config=BGDSACConfig(
                bayesianization="actor_and_critic",
                posterior_std=0.1,
                replay_evidence=ReplayEvidenceConfig(mode="inverse_reuse_weight"),
                adaptive_td_retention=(
                    AdaptiveTDRetentionConfig(
                        mapping=RetentionMappingConfig(
                            lambda_min=0.5,
                            kappa=1.0,
                        )
                    )
                    if optimizer == "adaptive_bgd"
                    else None
                ),
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
    else:
        raise ValueError(f"unsupported recurrent CW optimizer: {optimizer}")

    return_matrix = PerformanceMatrixRecorder(bundle.task_names)
    success_matrix = PerformanceMatrixRecorder(bundle.task_names)

    def stage_evaluator(
        completed_steps: int,
        current_agent: RecurrentSACAgent,
    ) -> None:
        if completed_steps % steps_per_task != 0:
            return
        stage_index = completed_steps // steps_per_task - 1
        evaluations: list[TaskEvaluation] = []
        with preserved_random_state(), preserve_recurrent_online_state(current_agent):
            for task_index, env in enumerate(evaluation_envs):
                evaluations.append(
                    evaluate_recurrent_task(
                        current_agent,
                        env,
                        episodes=evaluation_episodes,
                        seed=(80_000 + stage_index * 100_000 + task_index * 10_000),
                        max_episode_steps=episode_horizon,
                    )
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
    try:
        training = train_recurrent_sac(
            train_env,
            agent,
            config=RecurrentSACTrainConfig(
                total_steps=total_steps,
                random_steps=min(
                    initial_random_steps,
                    max(0, total_steps - 1),
                ),
                sequence_batch_size=sequence_batch_size,
                burn_in=history_length,
                unroll=unroll,
                replay_capacity=max(
                    replay_capacity,
                    history_length + unroll,
                ),
                updates_per_step=1,
                seed=seed,
            ),
            post_step_observer=stage_evaluator,
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
            "protocol_label": "3RL-style",
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
                "posterior_reset_on_task_change": False,
                "recurrent_input_fields": [
                    "observation",
                    "previous_action",
                    "previous_reward",
                    "previous_done",
                ],
            },
            "three_rl_reference": {
                "paper_history_length": 15,
                "paper_actor_hidden_sizes": [400, 400],
                "paper_critic_hidden_sizes": [400, 400],
                "paper_context_dim": 30,
                "paper_learning_rate": 3e-4,
                "paper_batch_size": 1028,
                "paper_replay_size": 10_000_000,
                "paper_cw10_total_steps": 10_000_000,
                "paper_max_episode_steps": 200,
            },
            "implementation_deviations": [
                (
                    "Meta-World v3 adapter replaces the older Meta-World "
                    "stack used by the reference implementation"
                ),
                (
                    "GRU state is integrated into actor/critic sequence "
                    "models rather than a separate fixed-window context module"
                ),
                (
                    "sequence minibatches use explicit burn-in/unroll "
                    "semantics rather than flattened history-buffer features"
                ),
                (
                    "BGD and adaptive-BGD variants are RL-BGD extensions, "
                    "not methods reported in the 3RL paper"
                ),
            ],
        }
    finally:
        bundle.close()


def main() -> None:
    print(
        json.dumps(
            run_recurrent_ta_continual_world_sac(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
