"""Matched recurrent Adam/BGD PPO on a hidden recurring LQR stream."""

from __future__ import annotations

import json
from typing import Literal

from rl_bgd.agents.ppo.agent import (
    PPOConfig,
)
from rl_bgd.agents.ppo.bgd_agent import (
    BGDPPOConfig,
)
from rl_bgd.agents.ppo.recurrent_agent import (
    RecurrentPPOAgent,
    RecurrentPPOConfig,
)
from rl_bgd.agents.ppo.recurrent_bgd_agent import (
    BGDRecurrentPPOAgent,
)
from rl_bgd.agents.ppo.recurrent_train import (
    RecurrentPPOTrainConfig,
    train_recurrent_ppo,
)
from rl_bgd.bayes.bgd import (
    BGDConfig,
)
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
from rl_bgd.utils.device import (
    resolve_device,
)
from rl_bgd.utils.randomness import (
    seed_everything,
)

OptimizerFamily = Literal[
    "adam",
    "bgd",
]


def run_recurrent_ppo_recurring_lqr(
    *,
    steps: int = 384,
    seed: int = 71,
    device: str = "auto",
    optimizer: OptimizerFamily = "adam",
) -> dict[str, object]:
    """Run recurrent PPO without exposing context values or switch times."""

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
                    "dynamics": 0.65,
                    "control_gain": 0.3,
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
    ppo_config = PPOConfig(
        actor_lr=1e-3,
        value_lr=1e-3,
        update_epochs=2,
        minibatch_size=16,
    )
    recurrent_config = RecurrentPPOConfig(
        recurrent_hidden_dim=24,
        sequence_length=16,
        encoder_hidden_dims=(24,),
    )
    if optimizer == "adam":
        agent: RecurrentPPOAgent = RecurrentPPOAgent(
            1,
            1,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            ppo_config=ppo_config,
            recurrent_config=recurrent_config,
            device=resolved,
        )
    elif optimizer == "bgd":
        agent = BGDRecurrentPPOAgent(
            1,
            1,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            ppo_config=ppo_config,
            recurrent_config=recurrent_config,
            bgd_config=BGDPPOConfig(
                bayesianization="actor_and_value",
                posterior_std=0.1,
                evidence_mode="first_epoch_only",
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
        raise ValueError(f"unsupported optimizer family: {optimizer}")

    summary = train_recurrent_ppo(
        env,
        agent,
        config=RecurrentPPOTrainConfig(
            total_steps=steps,
            rollout_steps=64,
            seed=seed,
        ),
    )
    recurrent_reset_count = summary.get(
        "recurrent_reset_count"
    )
    completed_episodes = summary.get(
        "episodes"
    )
    if (
        isinstance(
            recurrent_reset_count,
            bool,
        )
        or not isinstance(
            recurrent_reset_count,
            int,
        )
        or isinstance(
            completed_episodes,
            bool,
        )
        or not isinstance(
            completed_episodes,
            int,
        )
    ):
        raise TypeError(
            "recurrent PPO summary reset/episode counts must be integers"
        )

    return {
        "optimizer": optimizer,
        "steps": steps,
        "environment_steps": env.environment_step,
        "final_evaluation_context": (env.evaluation_context),
        "training": summary,
        "information_access": {
            "receives_task_id": False,
            "receives_task_boundary": False,
            "receives_context": False,
            "hidden_state_resets_only_on_episode_end": (
                recurrent_reset_count
                == completed_episodes + 1
            ),
        },
    }


def main() -> None:
    print(
        json.dumps(
            run_recurrent_ppo_recurring_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
