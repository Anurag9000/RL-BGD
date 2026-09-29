import torch

from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.synthetic.lqr import LinearQuadraticControlEnv
from rl_bgd.envs.synthetic.nonstationary_lqr import ScheduledLQREnv


def test_scheduled_lqr_changes_dynamics_without_leaking_context() -> None:
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="abrupt",
            anchors=(
                {"dynamics": 0.9},
                {"dynamics": 0.2},
            ),
            phase_steps=1,
        )
    )
    base = LinearQuadraticControlEnv(
        horizon=10,
        process_noise=0.0,
    )
    env = ScheduledLQREnv(base, schedule)
    _, reset_info = env.reset(seed=0)
    assert reset_info == {}
    _, _, _, _, info = env.step(torch.zeros(1))
    assert info == {}
    env.step(torch.zeros(1))
    assert base.dynamics == 0.2
    assert env.evaluation_context["dynamics"] == 0.2
