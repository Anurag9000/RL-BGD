"""Oracle-boundary EWC/SI/MAS SAC baselines on recurring LQR."""

from __future__ import annotations

import json

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.regularized_agent import (
    RegularizationMethod,
    RegularizedSACAgent,
    RegularizedSACConfig,
)
from rl_bgd.agents.sac.regularized_train import (
    BoundaryRegularizedSACTrainConfig,
    train_boundary_regularized_sac,
)
from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.envs.synthetic.nonstationary_lqr import ScheduledLQREnv
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


def run_boundary_regularized_sac_recurring_lqr(
    *,
    method: RegularizationMethod = "ewc",
    steps: int = 384,
    phase_steps: int = 128,
    seed: int = 81,
    device: str = "auto",
) -> dict[str, object]:
    """Run oracle-boundary consolidation without task-ID or head routing."""

    if steps < 64 or phase_steps < 1:
        raise ValueError("invalid regularized SAC experiment horizon")
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="recurring",
            anchors=(
                {"dynamics": 0.72, "control_gain": 0.32},
                {"dynamics": 0.97, "control_gain": 0.70},
                {"dynamics": 0.84, "control_gain": 0.48},
            ),
            phase_steps=phase_steps,
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
        hidden_dims=(32, 32),
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        regularization_config=RegularizedSACConfig(
            method=method,
            target="actor_and_critic",
            strength=1.0,
            consolidation_interval_updates=1_000_000_000,
            importance_samples=16,
        ),
        device=resolved,
    )
    consolidation_steps = tuple(
        range(
            phase_steps,
            steps,
            phase_steps,
        )
    )
    training = train_boundary_regularized_sac(
        env,
        agent,
        config=BoundaryRegularizedSACTrainConfig(
            total_steps=steps,
            consolidation_steps=consolidation_steps,
            random_steps=16,
            batch_size=16,
            replay_capacity=max(2_000, steps),
            importance_batch_size=64,
            updates_per_step=1,
            seed=seed,
        ),
    )
    return {
        "method": method,
        "training": training,
        "information_access": {
            "receives_task_id": False,
            "receives_task_boundary": True,
            "receives_context": False,
            "task_specific_heads": False,
            "consolidation_trigger": "oracle_phase_boundary",
        },
    }


def main() -> None:
    print(
        json.dumps(
            run_boundary_regularized_sac_recurring_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
