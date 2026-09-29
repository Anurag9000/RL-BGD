"""Task-agnostic PPO validation on a recurring nonstationary LQR stream."""

from __future__ import annotations

import json
from typing import Literal

from rl_bgd.agents.ppo.agent import (
    PPOAgent,
    PPOConfig,
)
from rl_bgd.agents.ppo.bgd_agent import (
    BGDPPOAgent,
    BGDPPOConfig,
)
from rl_bgd.agents.ppo.train import (
    PPOTrainConfig,
    train_ppo,
)
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.continual.schedules import (
    ContextSchedule,
    ContextScheduleConfig,
)
from rl_bgd.envs.synthetic.lqr import (
    LinearQuadraticControlEnv,
)
from rl_bgd.envs.synthetic.nonstationary_lqr import (
    ScheduledLQREnv,
)
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything

OptimizerFamily = Literal["adam", "bgd"]


def run_ppo_recurring_lqr(
    *,
    steps: int = 384,
    seed: int = 31,
    device: str = "auto",
    optimizer: OptimizerFamily = "adam",
) -> dict[str, object]:
    seed_everything(
        seed,
        deterministic=True,
    )
    resolved = resolve_device(device)
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="recurring",
            anchors=(
                {
                    "dynamics": 0.75,
                    "control_gain": 0.35,
                },
                {
                    "dynamics": 0.95,
                    "control_gain": 0.7,
                },
                {
                    "dynamics": 0.85,
                    "control_gain": 0.5,
                },
            ),
            phase_steps=64,
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
    ppo_config = PPOConfig(
        actor_lr=1e-3,
        value_lr=1e-3,
        update_epochs=2,
        minibatch_size=32,
    )
    if optimizer == "adam":
        agent: PPOAgent = PPOAgent(
            1,
            1,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_dims=(32, 32),
            config=ppo_config,
            device=resolved,
        )
    elif optimizer == "bgd":
        agent = BGDPPOAgent(
            1,
            1,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_dims=(32, 32),
            ppo_config=ppo_config,
            bgd_config=BGDPPOConfig(
                bayesianization="actor_and_value",
                posterior_std=0.1,
                actor_bgd=BGDConfig(
                    eta=0.1,
                    mc_samples=2,
                    antithetic=True,
                ),
                value_bgd=BGDConfig(
                    eta=0.1,
                    mc_samples=2,
                    antithetic=True,
                ),
            ),
            device=resolved,
        )
    else:
        raise ValueError(
            f"unsupported optimizer family: {optimizer}"
        )

    summary = train_ppo(
        env,
        agent,
        config=PPOTrainConfig(
            total_steps=steps,
            rollout_steps=64,
            seed=seed,
        ),
    )
    return {
        "optimizer": optimizer,
        "steps": steps,
        "environment_steps": env.environment_step,
        "final_evaluation_context": (
            env.evaluation_context
        ),
        "training": summary,
    }


def main() -> None:
    print(
        json.dumps(
            run_ppo_recurring_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
