"""Strict task-agnostic SAC/BGD experiments on CARL Pendulum."""

from __future__ import annotations

import json
from typing import Literal

from rl_bgd.agents.sac.agent import SACAgent, SACConfig
from rl_bgd.agents.sac.bgd_agent import BGDSACAgent, BGDSACConfig
from rl_bgd.agents.sac.train import SACTrainConfig, train_sac
from rl_bgd.bayes.bgd import BGDConfig
from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.carl.stream import make_carl_pendulum_stream
from rl_bgd.replay.evidence_accounting import ReplayEvidenceConfig
from rl_bgd.surprise.base import RetentionMappingConfig
from rl_bgd.surprise.td import AdaptiveTDRetentionConfig
from rl_bgd.utils.device import resolve_device
from rl_bgd.utils.randomness import seed_everything

CARLMode = Literal["abrupt", "smooth", "recurring"]
CARLOptimizer = Literal["adam", "bgd", "adaptive_bgd"]


def _schedule(
    mode: CARLMode,
    *,
    phase_steps: int,
    seed: int,
) -> ContextSchedule:
    anchors: tuple[
        dict[str, float],
        ...,
    ]
    if mode == "smooth":
        anchors = (
            {"g": 10.0, "m": 1.0, "l": 1.0, "dt": 0.05},
            {"g": 15.0, "m": 1.5, "l": 1.2, "dt": 0.05},
        )
    else:
        anchors = (
            {"g": 10.0, "m": 1.0, "l": 1.0, "dt": 0.05},
            {"g": 15.0, "m": 1.0, "l": 1.0, "dt": 0.05},
            {"g": 5.0, "m": 1.0, "l": 1.0, "dt": 0.05},
        )
    return ContextSchedule(
        ContextScheduleConfig(
            mode=mode,
            anchors=anchors,
            phase_steps=phase_steps,
            seed=seed,
        )
    )


def run_carl_pendulum_sac(
    *,
    mode: CARLMode = "recurring",
    optimizer: CARLOptimizer = "adam",
    steps: int = 150_000,
    phase_steps: int = 50_000,
    seed: int = 0,
    device: str = "auto",
) -> dict[str, object]:
    """Train SAC on CARL without exposing context values or change events."""

    if steps < 1 or phase_steps < 1:
        raise ValueError("CARL training budgets must be positive")
    seed_everything(seed, deterministic=True)
    resolved = resolve_device(device)
    env = make_carl_pendulum_stream(
        _schedule(
            mode,
            phase_steps=phase_steps,
            seed=seed,
        ),
        device=resolved,
        strict_task_agnostic=True,
        expose_evaluation_context=False,
    )
    observation_dim = int(env.observation_space.low.numel())
    action_dim = int(env.action_space.low.numel())
    sac_config = SACConfig(
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
    )

    if optimizer == "adam":
        agent: SACAgent = SACAgent(
            observation_dim,
            action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_dims=(256, 256),
            config=sac_config,
            device=resolved,
        )
    else:
        agent = BGDSACAgent(
            observation_dim,
            action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_dims=(256, 256),
            sac_config=sac_config,
            bgd_config=BGDSACConfig(
                bayesianization="critic_only",
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
                critic_bgd=BGDConfig(
                    eta=0.1,
                    mc_samples=4,
                    antithetic=True,
                ),
            ),
            device=resolved,
        )

    try:
        training = train_sac(
            env,
            agent,
            config=SACTrainConfig(
                total_steps=steps,
                random_steps=min(1_000, max(0, steps - 1)),
                batch_size=min(256, max(2, steps // 4)),
                replay_capacity=max(100_000, steps),
                seed=seed,
            ),
        )
        return {
            "environment": "CARLPendulum",
            "mode": mode,
            "optimizer": optimizer,
            "steps": steps,
            "phase_steps": phase_steps,
            "training": training,
            "information_access": {
                "receives_task_id": False,
                "receives_task_boundary": False,
                "receives_context": False,
                "context_hidden_by_adapter": True,
            },
        }
    finally:
        env.close()


def main() -> None:
    print(
        json.dumps(
            run_carl_pendulum_sac(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
