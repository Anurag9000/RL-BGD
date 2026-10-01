"""Shared hidden-context recurring LQR benchmark construction."""

from __future__ import annotations

import torch

from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.recurrent_context import PreviousTransitionContextEnv
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.envs.synthetic.nonstationary_lqr import ScheduledLQREnv


def make_hidden_context_lqr_env(
    *,
    seed: int,
    device: torch.device | str,
    phase_steps: int = 40,
    episode_horizon: int = 24,
) -> PreviousTransitionContextEnv:
    """Build the common task-agnostic A->B->C->A control stream."""

    if phase_steps < 1 or episode_horizon < 1:
        raise ValueError("hidden-context LQR horizons must be positive")
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
            phase_steps=phase_steps,
            seed=seed,
        )
    )
    return PreviousTransitionContextEnv(
        ScheduledLQREnv(
            LinearQuadraticControlEnv(
                horizon=episode_horizon,
                device=device,
            ),
            schedule,
        )
    )
