"""Matched recurrent Adam/BGD SAC on a hidden recurring LQR stream."""

from __future__ import annotations

import json
from typing import Literal

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
from rl_bgd.envs.hidden_context_lqr import make_hidden_context_lqr_env
from rl_bgd.replay.evidence_accounting import ReplayEvidenceConfig
from rl_bgd.surprise.base import RetentionMappingConfig
from rl_bgd.surprise.td import AdaptiveTDRetentionConfig
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything

OptimizerFamily = Literal[
    "adam",
    "bgd",
    "adaptive_bgd",
]


def run_recurrent_sac_recurring_lqr(
    *,
    steps: int = 256,
    seed: int = 86,
    device: str = "auto",
    optimizer: OptimizerFamily = "adam",
) -> dict[str, object]:
    """Run recurrent SAC without task IDs, true context, or switch callbacks."""

    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    env = make_hidden_context_lqr_env(
        seed=seed,
        device=resolved,
    )
    observation_dim = int(env.observation_space.low.numel())
    action_dim = int(env.action_space.low.numel())
    sac_config = SACConfig(
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
    )
    recurrent_config = RecurrentSACConfig(
        recurrent_hidden_dim=16,
        encoder_hidden_dims=(16,),
        q_hidden_dims=(16,),
    )

    if optimizer == "adam":
        agent: RecurrentSACAgent = RecurrentSACAgent(
            observation_dim,
            action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            sac_config=sac_config,
            recurrent_config=recurrent_config,
            device=resolved,
        )
    elif optimizer in {"bgd", "adaptive_bgd"}:
        agent = BGDRecurrentSACAgent(
            observation_dim,
            action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            sac_config=sac_config,
            recurrent_config=recurrent_config,
            bgd_config=BGDSACConfig(
                bayesianization="actor_and_critic",
                posterior_std=0.1,
                replay_evidence=ReplayEvidenceConfig(mode="inverse_reuse_weight"),
                adaptive_td_retention=(
                    AdaptiveTDRetentionConfig(
                        mapping=RetentionMappingConfig(
                            lambda_min=0.6,
                            kappa=1.0,
                        )
                    )
                    if optimizer == "adaptive_bgd"
                    else None
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
        raise ValueError(f"unsupported optimizer family: {optimizer}")

    try:
        summary = train_recurrent_sac(
            env,
            agent,
            config=RecurrentSACTrainConfig(
                total_steps=steps,
                random_steps=min(48, max(0, steps - 1)),
                sequence_batch_size=4,
                burn_in=4,
                unroll=8,
                replay_capacity=max(512, steps),
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
                "recurrent SAC summary reset/episode counts must be integers"
            )

        return {
            "optimizer": optimizer,
            "steps": steps,
            "environment_steps": env.environment_step,
            "final_evaluation_context": env.evaluation_context,
            "training": summary,
            "information_access": {
                "receives_task_id": False,
                "receives_task_boundary": False,
                "receives_context": False,
                "receives_environment_context": False,
                "recurrent_input_fields": [
                    "observation",
                    "previous_action",
                    "previous_reward",
                    "previous_done",
                ],
                "hidden_state_resets_only_on_episode_end": (
                    recurrent_reset_count
                    == completed_episodes + 1
                ),
            },
        }
    finally:
        env.close()


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
