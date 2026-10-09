from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.carl.stream import CARLContextStream


class FakeSpace:
    def __init__(self, low: list[float], high: list[float]) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape


class FakeContextSpace:
    def insert_defaults(
        self,
        context: dict[str, float],
    ) -> dict[str, float]:
        return {"g": 10.0, "m": 1.0, **context}


class FakeCARL:
    def __init__(self) -> None:
        self.action_space = FakeSpace([-2.0], [2.0])
        self.base_observation_space = FakeSpace(
            [-1.0, -1.0],
            [1.0, 1.0],
        )
        self.context: dict[str, float] = {
            "g": 10.0,
            "m": 1.0,
        }
        self.applied: list[dict[str, float]] = []
        self.reset_contexts: list[dict[str, float]] = []

    def get_context_space(self) -> FakeContextSpace:
        return FakeContextSpace()

    def _update_context(self) -> None:
        self.applied.append(dict(self.context))

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed
        self.reset_contexts.append(dict(self.context))
        return (
            {
                "obs": np.asarray(
                    [0.25, -0.25],
                    dtype=np.float32,
                ),
                "context": dict(self.context),
            },
            {"context_id": 0},
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        dict[str, Any],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        assert action.shape == (1,)
        return (
            {
                "obs": np.asarray(
                    [0.5, -0.5],
                    dtype=np.float32,
                ),
                "context": dict(self.context),
            },
            1.0,
            False,
            False,
            {"context_id": 0},
        )


def test_strict_carl_stream_hides_context_and_switch_metadata() -> None:
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="abrupt",
            anchors=({"g": 7.0}, {"g": 5.0}),
            phase_steps=1,
        )
    )
    base = FakeCARL()
    env = CARLContextStream(base, schedule)
    observation, info = env.reset(seed=0)
    assert observation.shape == (2,)
    assert base.reset_contexts[-1]["g"] == 7.0
    assert base.applied[-1]["g"] == 7.0
    assert "context_id" not in info
    assert "context" not in info
    _, _, _, _, step_info = env.step(torch.tensor([0.0]))
    assert "context_id" not in step_info
    env.step(torch.tensor([0.0]))
    assert base.applied[-1]["g"] == 5.0
    assert env.evaluation_context["g"] == 5.0


def test_evaluation_context_exposure_rejected_in_strict_mode() -> None:
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="abrupt",
            anchors=({"g": 10.0},),
        )
    )
    try:
        CARLContextStream(
            FakeCARL(),
            schedule,
            strict_task_agnostic=True,
            expose_evaluation_context=True,
        )
    except ValueError:
        return
    raise AssertionError("strict mode should reject exposed context")
