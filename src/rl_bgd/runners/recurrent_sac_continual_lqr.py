"""Matched recurrent Adam/BGD SAC on a hidden recurring LQR stream."""

from __future__ import annotations

import json
from typing import Literal

from rl_bgd.agents.sac.agent import (
    SACConfig,
)
from rl_bgd.agents.sac.bgd_agent import (
    BGDSACConfig,
)
from rl_bgd.agents.sac.recurrent_agent import (
    RecurrentSACAgent,
    RecurrentSACConfig,
)
from rl_bgd.agents.sac.recurrent_bgd_agent import (
    BGDRecurrentSACAgent,
)
from rl_bgd.agents.sac.recurrent_train import (
    RecurrentSACTrainConfig,
    train_recurrent_sac,
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
from rl_bgd.replay.evidence_accounting import (
    ReplayEvidenceConfig,
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


def run_recurrent_sac_recurring_lqr(
    *,
    steps: int = 256,
    seed: int = 86,
    device: str = "auto",
    optimizer: OptimizerFamily = "adam",
) -> dict[str, object]:
    """Run recurrent SAC without task IDs, context values, or switch callbacks."""

    seed_everything(
        seed,
        deterministic=True,
    )
    resolved = resolve_device(
        device
    )
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
    sac_config = SACConfig(
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
    )
    recurrent_config = (
        RecurrentSACConfig(
            recurrent_hidden_dim=16,
            encoder_hidden_dims=(16,),
            q_hidden_dims=(16,),
        )
    )
    if optimizer == "adam":
        agent: RecurrentSACAgent = (
            RecurrentSACAgent(
                1,
                1,
                action_low=env.action_space.low,
                action_high=env.action_space.high,
                sac_config=sac_config,
                recurrent_config=recurrent_config,
                device=resolved,
            )
        )
    elif optimizer == "bgd":
        agent = BGDRecurrentSACAgent(
            1,
            1,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            sac_config=sac_config,
            recurrent_config=recurrent_config,
            bgd_config=BGDSACConfig(
                bayesianization="actor_and_critic",
                posterior_std=0.1,
                replay_evidence=ReplayEvidenceConfig(
                    mode="inverse_reuse_weight"
                ),
                actor_bgd=BGDConfig(
                    eta=0.1,
                    mc_samples=2,
                    antithetic=True,
                ),
                critic_bgd=BGDConfig(
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

    summary = train_recurrent_sac(
        env,
        agent,
        config=RecurrentSACTrainConfig(
            total_steps=steps,
            random_steps=48,
            sequence_batch_size=4,
            burn_in=4,
            unroll=8,
            replay_capacity=max(
                512,
                steps,
            ),
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
        "information_access": {
            "receives_task_id": False,
            "receives_task_boundary": False,
            "receives_context": False,
            "hidden_state_resets_only_on_episode_end": (
                summary[
                    "recurrent_reset_count"
                ]
                == summary[
                    "episodes"
                ]
                + 1
            ),
        },
    }


def main() -> None:
    print(
        json.dumps(
            run_recurrent_sac_recurring_lqr(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
