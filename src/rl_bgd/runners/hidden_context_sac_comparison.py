"""Five-way hidden-context SAC comparison required by the research plan."""

from __future__ import annotations

import json
from typing import Literal

from rl_bgd.agents.sac.agent import SACAgent, SACConfig
from rl_bgd.agents.sac.bgd_agent import BGDSACAgent, BGDSACConfig
from rl_bgd.agents.sac.train import SACTrainConfig, train_sac
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.envs.hidden_context_lqr import make_hidden_context_lqr_env
from rl_bgd.replay.evidence_accounting import ReplayEvidenceConfig
from rl_bgd.runners.recurrent_sac_continual_lqr import (
    run_recurrent_sac_recurring_lqr,
)
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything

FeedforwardOptimizer = Literal["adam", "bgd"]


def _run_feedforward(
    *,
    optimizer: FeedforwardOptimizer,
    steps: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    env = make_hidden_context_lqr_env(
        seed=seed,
        device=resolved,
    )
    observation_dim = int(env.observation_space.low.numel())
    action_dim = int(env.action_space.low.numel())
    config = SACConfig(
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
    )
    if optimizer == "adam":
        agent: SACAgent = SACAgent(
            observation_dim,
            action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_dims=(16, 16),
            config=config,
            device=resolved,
        )
    else:
        agent = BGDSACAgent(
            observation_dim,
            action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_dims=(16, 16),
            sac_config=config,
            bgd_config=BGDSACConfig(
                bayesianization="actor_and_critic",
                posterior_std=0.1,
                replay_evidence=ReplayEvidenceConfig(mode="inverse_reuse_weight"),
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

    batch_size = min(32, max(4, steps // 4))
    random_steps = min(32, max(0, steps - 1))
    try:
        training = train_sac(
            env,
            agent,
            config=SACTrainConfig(
                total_steps=steps,
                random_steps=random_steps,
                batch_size=batch_size,
                replay_capacity=max(256, steps),
                seed=seed,
            ),
        )
        return {
            "optimizer": optimizer,
            "recurrent": False,
            "training": training,
            "information_access": {
                "receives_task_id": False,
                "receives_task_boundary": False,
                "receives_environment_context": False,
                "input_fields": [
                    "observation",
                    "previous_action",
                    "previous_reward",
                    "previous_done",
                ],
            },
        }
    finally:
        env.close()


def run_hidden_context_sac_comparison(
    *,
    steps: int = 256,
    seed: int = 90,
    device: str = "auto",
) -> dict[str, object]:
    """Execute the matched feed-forward/recurrent continual-control suite."""

    if steps < 16:
        raise ValueError("comparison requires at least 16 environment steps")
    return {
        "benchmark": "hidden_recurring_lqr",
        "steps_per_variant": steps,
        "seed": seed,
        "variants": {
            "feedforward_adam": _run_feedforward(
                optimizer="adam",
                steps=steps,
                seed=seed,
                device=device,
            ),
            "feedforward_bgd": _run_feedforward(
                optimizer="bgd",
                steps=steps,
                seed=seed,
                device=device,
            ),
            "recurrent_adam": run_recurrent_sac_recurring_lqr(
                steps=steps,
                seed=seed,
                device=device,
                optimizer="adam",
            ),
            "recurrent_bgd": run_recurrent_sac_recurring_lqr(
                steps=steps,
                seed=seed,
                device=device,
                optimizer="bgd",
            ),
            "recurrent_adaptive_bgd": run_recurrent_sac_recurring_lqr(
                steps=steps,
                seed=seed,
                device=device,
                optimizer="adaptive_bgd",
            ),
        },
        "comparison_contract": {
            "same_stream": True,
            "same_per_step_inputs": True,
            "task_id_hidden": True,
            "task_boundary_hidden": True,
            "difference_of_interest": [
                "recurrent_memory",
                "bayesian_consolidation",
                "adaptive_replasticization",
            ],
        },
    }


def main() -> None:
    print(
        json.dumps(
            run_hidden_context_sac_comparison(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
