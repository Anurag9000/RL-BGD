"""Shared recurring-LQR profile for matched continual-learning baselines."""

from __future__ import annotations

import torch

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

BASELINE_RECURRING_LQR_PROFILE = "recurring_lqr_matched_v1"


def baseline_recurring_lqr_anchors() -> tuple[
    dict[str, float],
    ...,
]:
    """Return a fresh copy of the canonical matched-baseline context anchors."""

    return (
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
    )


def make_baseline_recurring_lqr(
    *,
    seed: int,
    device: torch.device | str,
    phase_steps: int,
    horizon: int,
) -> ScheduledLQREnv:
    """Build the exact recurring stream shared by baseline/control runners."""

    if phase_steps < 1:
        raise ValueError("baseline recurring LQR phase_steps must be positive")
    if horizon < 1:
        raise ValueError("baseline recurring LQR horizon must be positive")

    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="recurring",
            anchors=(baseline_recurring_lqr_anchors()),
            phase_steps=phase_steps,
            seed=seed,
        )
    )
    return ScheduledLQREnv(
        LinearQuadraticControlEnv(
            horizon=horizon,
            device=device,
        ),
        schedule,
    )
