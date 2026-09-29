import numpy as np
import pytest
import torch

from rl_bgd.envs.continual_world import (
    MetaWorldTaskAdapter,
    make_continual_world_stream,
)


class FakeBox:
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
    ) -> None:
        self.low = low
        self.high = high


class FakeMetaWorldEnv:
    def __init__(self) -> None:
        self.action_space = FakeBox(
            np.full(
                (2,),
                -1.0,
                dtype=np.float32,
            ),
            np.full(
                (2,),
                1.0,
                dtype=np.float32,
            ),
        )
        self.observation_space = FakeBox(
            np.full(
                (3,),
                -5.0,
                dtype=np.float32,
            ),
            np.full(
                (3,),
                5.0,
                dtype=np.float32,
            ),
        )
        self._freeze_rand_vec = True
        self._step = 0
        self.task: object | None = None
        self.last_action: np.ndarray | None = None

    def set_task(
        self,
        task: object,
    ) -> None:
        self.task = task

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[
        np.ndarray,
        dict[str, object],
    ]:
        del seed
        self._step = 0
        return (
            np.array(
                [1.0, 2.0, 3.0],
                dtype=np.float32,
            ),
            {
                "task_id": 99,
            },
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, object],
    ]:
        self.last_action = action
        self._step += 1
        return (
            np.array(
                [4.0, 5.0, 6.0],
                dtype=np.float32,
            ),
            1.25,
            False,
            False,
            {
                "success": 1.0,
                "task_id": 99,
            },
        )


def test_metaworld_adapter_is_tensor_native_at_agent_boundary() -> None:
    raw_env = FakeMetaWorldEnv()
    task = object()
    env = MetaWorldTaskAdapter(
        raw_env,
        task,
        device="cpu",
        horizon=2,
    )
    assert raw_env.task is task
    assert not raw_env._freeze_rand_vec

    observation, info = env.reset(
        seed=7
    )
    assert info == {}
    torch.testing.assert_close(
        observation,
        torch.tensor(
            [1.0, 2.0, 3.0]
        ),
    )
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (
        3,
    )

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        torch.tensor([0.25, -0.5])
    )
    assert isinstance(
        raw_env.last_action,
        np.ndarray,
    )
    np.testing.assert_allclose(
        raw_env.last_action,
        np.array(
            [0.25, -0.5],
            dtype=np.float32,
        ),
    )
    torch.testing.assert_close(
        observation,
        torch.tensor(
            [4.0, 5.0, 6.0]
        ),
    )
    assert reward == pytest.approx(1.25)
    assert not terminated
    assert not truncated
    assert info == {
        "success": 1.0,
    }

    _, _, _, truncated, _ = env.step(
        torch.zeros(2)
    )
    assert truncated


@pytest.mark.benchmark
def test_modern_metaworld_cw10_factory_smoke() -> None:
    pytest.importorskip(
        "metaworld"
    )
    env = make_continual_world_stream(
        "CW10",
        steps_per_task=2,
        seed=13,
        device="cpu",
        episode_horizon=2,
    )
    observation, info = env.reset(
        seed=13
    )
    assert observation.shape == (
        env.observation_space.shape
    )
    assert info == {}
    action = torch.zeros(
        env.action_space.shape,
    )
    _, reward, _, _, info = env.step(
        action
    )
    assert np.isfinite(reward)
    assert info == {}
