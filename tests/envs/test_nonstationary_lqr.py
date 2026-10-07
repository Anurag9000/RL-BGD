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


def test_lqr_checkpoint_restores_simulator_and_local_rng() -> None:
    env = LinearQuadraticControlEnv(
        horizon=10,
        process_noise=0.05,
    )
    env.reset(seed=17)
    env.step(torch.tensor([0.25]))
    state = env.state_dict()
    expected = env.step(torch.tensor([-0.4]))

    restored = LinearQuadraticControlEnv(
        horizon=10,
        process_noise=0.05,
    )
    restored.load_state_dict(state)
    actual = restored.step(torch.tensor([-0.4]))

    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1:] == expected[1:]


def test_scheduled_lqr_checkpoint_restores_stream_position() -> None:
    config = ContextScheduleConfig(
        mode="recurring",
        anchors=(
            {"dynamics": 0.9},
            {"dynamics": 0.2},
        ),
        phase_steps=1,
    )
    env = ScheduledLQREnv(
        LinearQuadraticControlEnv(horizon=10),
        ContextSchedule(config),
    )
    env.reset(seed=9)
    env.step(torch.zeros(1))
    env.step(torch.zeros(1))
    state = env.state_dict()
    expected_context = env.evaluation_context
    expected = env.step(torch.zeros(1))

    restored = ScheduledLQREnv(
        LinearQuadraticControlEnv(horizon=10),
        ContextSchedule(config),
    )
    restored.load_state_dict(state)
    assert restored.environment_step == 2
    assert restored.evaluation_context == expected_context
    actual = restored.step(torch.zeros(1))
    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1:] == expected[1:]
