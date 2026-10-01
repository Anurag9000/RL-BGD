import torch

from rl_bgd.envs.continual_bench import (
    ContinualBenchStreamConfig,
    make_continual_bench_stream,
)

import pytest

pytestmark = pytest.mark.benchmark


def test_live_continual_bench_two_task_hidden_stream() -> None:
    env = make_continual_bench_stream(
        config=ContinualBenchStreamConfig(
            task_sequence=(
                "button",
                "door",
            ),
            switch_mode="fixed_steps",
            steps_per_task=2,
            strict_task_agnostic=True,
        ),
        seed=0,
        device="cpu",
        render_mode=None,
    )
    try:
        observation, info = env.reset(seed=0)
        assert observation.ndim == 1
        assert torch.isfinite(observation).all()
        assert info == {}

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
                info,
            ) = env.step(action)
            assert observation.ndim == 1
            assert torch.isfinite(observation).all()
            assert torch.isfinite(
                torch.tensor(reward)
            )
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert info == {}
            if terminated or truncated:
                observation, info = env.reset()
                assert info == {}

        assert env.evaluation_context["task_name"] == "door"
    finally:
        env.close()
