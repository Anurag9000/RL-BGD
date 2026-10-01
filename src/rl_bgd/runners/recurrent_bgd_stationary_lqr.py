"""Stationary learning acceptance runners for recurrent BGD-PPO/SAC."""

from __future__ import annotations

import json

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.bgd_agent import BGDPPOConfig
from rl_bgd.agents.ppo.recurrent_agent import RecurrentPPOConfig
from rl_bgd.agents.ppo.recurrent_bgd_agent import BGDRecurrentPPOAgent
from rl_bgd.agents.ppo.recurrent_train import (
    RecurrentPPOTrainConfig,
    evaluate_recurrent_ppo,
    train_recurrent_ppo,
)
from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.bgd_agent import BGDSACConfig
from rl_bgd.agents.sac.recurrent_agent import RecurrentSACConfig
from rl_bgd.agents.sac.recurrent_bgd_agent import BGDRecurrentSACAgent
from rl_bgd.agents.sac.recurrent_train import (
    RecurrentSACTrainConfig,
    evaluate_recurrent_sac,
    train_recurrent_sac,
)
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.replay.evidence_accounting import ReplayEvidenceConfig
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything


def run_recurrent_bgd_ppo_lqr(
    *,
    steps: int = 1_000,
    seed: int = 101,
    device: str = "auto",
) -> dict[str, object]:
    """Validate that recurrent actor-and-value BGD-PPO learns stationary control."""

    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    env = LinearQuadraticControlEnv(
        horizon=30,
        device=resolved,
    )
    agent = BGDRecurrentPPOAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        ppo_config=PPOConfig(
            actor_lr=1e-3,
            value_lr=1e-3,
            update_epochs=4,
            minibatch_size=64,
        ),
        recurrent_config=RecurrentPPOConfig(
            recurrent_hidden_dim=24,
            sequence_length=16,
            encoder_hidden_dims=(24,),
        ),
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
    pre_return = evaluate_recurrent_ppo(
        env,
        agent,
        episodes=5,
        seed=60_000,
    )
    training = train_recurrent_ppo(
        env,
        agent,
        config=RecurrentPPOTrainConfig(
            total_steps=steps,
            rollout_steps=128,
            seed=seed,
        ),
    )
    post_return = evaluate_recurrent_ppo(
        env,
        agent,
        episodes=5,
        seed=60_000,
    )
    return {
        "algorithm": "recurrent_bgd_ppo",
        "steps": steps,
        "pre_return": pre_return,
        "post_return": post_return,
        "improvement": post_return - pre_return,
        "training": training,
    }


def run_recurrent_bgd_sac_lqr(
    *,
    steps: int = 800,
    seed: int = 102,
    device: str = "auto",
) -> dict[str, object]:
    """Validate that recurrent critic-BGD SAC learns stationary control."""

    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    env = LinearQuadraticControlEnv(
        horizon=30,
        device=resolved,
    )
    agent = BGDRecurrentSACAgent(
        1,
        1,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        sac_config=SACConfig(
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
        ),
        recurrent_config=RecurrentSACConfig(
            recurrent_hidden_dim=16,
            encoder_hidden_dims=(16,),
            q_hidden_dims=(16,),
        ),
        bgd_config=BGDSACConfig(
            bayesianization="critic_only",
            posterior_std=0.1,
            replay_evidence=ReplayEvidenceConfig(
                mode="inverse_reuse_weight"
            ),
            critic_bgd=BGDConfig(
                eta=0.1,
                mc_samples=2,
                antithetic=True,
            ),
        ),
        device=resolved,
    )
    pre_return = evaluate_recurrent_sac(
        env,
        agent,
        episodes=5,
        seed=70_000,
    )
    training = train_recurrent_sac(
        env,
        agent,
        config=RecurrentSACTrainConfig(
            total_steps=steps,
            random_steps=64,
            sequence_batch_size=4,
            burn_in=4,
            unroll=8,
            replay_capacity=max(2_000, steps),
            seed=seed,
        ),
    )
    post_return = evaluate_recurrent_sac(
        env,
        agent,
        episodes=5,
        seed=70_000,
    )
    return {
        "algorithm": "recurrent_bgd_sac",
        "steps": steps,
        "pre_return": pre_return,
        "post_return": post_return,
        "improvement": post_return - pre_return,
        "training": training,
    }


def main() -> None:
    print(
        json.dumps(
            {
                "ppo": run_recurrent_bgd_ppo_lqr(),
                "sac": run_recurrent_bgd_sac_lqr(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
