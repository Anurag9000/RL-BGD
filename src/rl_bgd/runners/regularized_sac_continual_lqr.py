"""Task-agnostic EWC/Online-EWC/SI/MAS SAC on recurring LQR."""

from __future__ import annotations

import json

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.regularized_agent import (
    RegularizationMethod,
    RegularizedSACAgent,
    RegularizedSACConfig,
)
from rl_bgd.agents.sac.train import SACTrainConfig, train_sac
from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.envs.synthetic.nonstationary_lqr import ScheduledLQREnv
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


def run_regularized_sac_recurring_lqr(
    *,
    method: RegularizationMethod = "ewc",
    steps: int = 192,
    seed: int = 93,
    device: str = "auto",
    consolidation_interval_updates: int = 8,
) -> dict[str, object]:
    """Run a fixed-update consolidation baseline without task-boundary input."""

    if steps < 32:
        raise ValueError("training horizon is too short")
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
                    "dynamics": 0.97,
                    "control_gain": 0.70,
                },
                {
                    "dynamics": 0.82,
                    "control_gain": 0.46,
                },
            ),
            phase_steps=40,
            seed=seed,
        )
    )
    env = ScheduledLQREnv(
        LinearQuadraticControlEnv(
            horizon=24,
            device=resolved,
        ),
        schedule,
    )
    agent = RegularizedSACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dims=(16, 16),
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        regularization_config=RegularizedSACConfig(
            method=method,
            target="actor_and_critic",
            strength=0.1,
            consolidation_interval_updates=consolidation_interval_updates,
            importance_samples=4,
            online_ewc_decay=0.95,
            si_damping=0.1,
        ),
        device=resolved,
    )
    training = train_sac(
        env,
        agent,
        config=SACTrainConfig(
            total_steps=steps,
            random_steps=24,
            batch_size=16,
            replay_capacity=max(512, steps),
            seed=seed,
        ),
    )
    return {
        "method": method,
        "steps": steps,
        "training": training,
        "consolidation_count": agent.consolidation_count,
        "final_evaluation_context": env.evaluation_context,
        "information_access": {
            "receives_task_id": False,
            "receives_task_boundary": False,
            "receives_context": False,
            "consolidation_trigger": "fixed_optimizer_update_interval",
            "consolidation_interval_updates": consolidation_interval_updates,
        },
    }


def main() -> None:
    print(
        json.dumps(
            run_regularized_sac_recurring_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
