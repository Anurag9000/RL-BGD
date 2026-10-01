import pytest
import torch

from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.carl import make_carl_pendulum_stream

pytestmark = pytest.mark.benchmark


def test_live_carl_pendulum_strict_hidden_context_stream() -> None:
    base = {
        "g": 10.0,
        "m": 1.0,
        "l": 1.0,
        "dt": 0.05,
    }
    changed = {
        **base,
        "g": 12.0,
    }
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="abrupt",
            anchors=(base, changed),
            phase_steps=2,
        )
    )
    env = make_carl_pendulum_stream(
        schedule,
        device="cpu",
        strict_task_agnostic=True,
    )
    try:
        observation, info = env.reset(seed=0)
        assert observation.ndim == 1
        assert torch.isfinite(observation).all()
        assert "context_id" not in info
        assert "context" not in info

        action = (
            env.action_space.low
            + env.action_space.high
        ) / 2.0
        for _ in range(3):
            (
                observation,
                reward,
                terminated,
                truncated,
                step_info,
            ) = env.step(action)
            assert observation.ndim == 1
            assert torch.isfinite(observation).all()
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "context_id" not in step_info
            assert "context" not in step_info

        assert env.evaluation_context["g"] == pytest.approx(12.0)
    finally:
        env.close()
