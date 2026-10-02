"""Oracle-boundary UCL-PPO baseline on recurring LQR."""

from __future__ import annotations

import json

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.train import PPOTrainConfig, train_ppo
from rl_bgd.agents.ppo.ucl_agent import UCLPPOAgent, UCLPPOConfig
from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.envs.synthetic.nonstationary_lqr import ScheduledLQREnv
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


def run_ucl_ppo_recurring_lqr(
    *,
    phase_steps: int = 128,
    phases: int = 4,
    seed: int = 140,
    device: str = "auto",
) -> dict[str, object]:
    """Run UCL-PPO with evaluator-supplied true phase boundaries."""

    if phase_steps < 32 or phases < 2:
        raise ValueError("UCL recurring LQR requires nontrivial phases")
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="recurring",
            anchors=(
                {
                    "dynamics": 0.68,
                    "control_gain": 0.32,
                },
                {
                    "dynamics": 0.98,
                    "control_gain": 0.72,
                },
                {
                    "dynamics": 0.82,
                    "control_gain": 0.48,
                },
            ),
            phase_steps=phase_steps,
            seed=seed,
        )
    )
    env = ScheduledLQREnv(
        LinearQuadraticControlEnv(
            horizon=32,
            device=resolved,
        ),
        schedule,
    )
    agent = UCLPPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(32, 32),
        ppo_config=PPOConfig(
            actor_lr=3e-4,
            value_lr=1e-3,
            update_epochs=4,
            minibatch_size=32,
        ),
        ucl_config=UCLPPOConfig(),
        device=resolved,
    )

    phase_summaries: list[dict[str, object]] = []
    boundary_log: list[dict[str, object]] = []
    for phase_index in range(phases):
        phase_summaries.append(
            train_ppo(
                env,
                agent,
                config=PPOTrainConfig(
                    total_steps=phase_steps,
                    rollout_steps=min(64, phase_steps),
                    seed=seed + phase_index,
                ),
            )
        )
        if phase_index < phases - 1:
            agent.consolidate_boundary()
            boundary_log.append(
                {
                    "after_phase": phase_index + 1,
                    "environment_step": env.environment_step,
                    "boundary_count": agent.boundary_count,
                }
            )

    final_phase = phase_summaries[-1]
    final_phase_return = final_phase.get(
        "final_10_mean_return"
    )
    if not isinstance(
        final_phase_return,
        (int, float),
    ):
        raise TypeError(
            "UCL final phase did not expose final_10_mean_return"
        )

    return {
        "algorithm": "ucl_ppo",
        "final_phase_return": float(final_phase_return),
        "protocol": "oracle_boundary",
        "steps": phase_steps * phases,
        "phase_steps": phase_steps,
        "phases": phases,
        "phase_summaries": phase_summaries,
        "boundaries": boundary_log,
        "final_evaluation_context": env.evaluation_context,
        "information_access": {
            "receives_task_id": False,
            "receives_task_boundary": True,
            "task_specific_heads": False,
            "posterior_snapshot_trigger": "oracle_phase_boundary",
        },
        "source_alignment": {
            "optimizer": "adam",
            "bayesian_hidden_layers": True,
            "saved_previous_task_posterior": True,
            "ucl_mean_strength_regularizer": True,
            "ucl_sigma_regularizer": True,
            "deterministic_output_heads": True,
        },
    }


def main() -> None:
    print(
        json.dumps(
            run_ucl_ppo_recurring_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
