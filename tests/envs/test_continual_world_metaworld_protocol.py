import pytest
import torch

from rl_bgd.envs.continual_world import (
    CanonicalContinualWorldStreamEnv,
    ContinualWorldStreamEnv,
    make_continual_world_protocol,
)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("protocol", "expected_type"),
    [
        ("task_agnostic", ContinualWorldStreamEnv),
        ("canonical", CanonicalContinualWorldStreamEnv),
    ],
)
def test_metaworld_protocol_bundle_has_separate_evaluation_envs(
    protocol: str,
    expected_type: type,
) -> None:
    pytest.importorskip("metaworld")
    bundle = make_continual_world_protocol(
        "CW10",
        protocol=protocol,  # type: ignore[arg-type]
        steps_per_task=1,
        seed=23,
        device="cpu",
        episode_horizon=1,
    )
    assert isinstance(bundle.train_env, expected_type)
    assert len(bundle.evaluation_envs) == 10
    assert len(bundle.task_names) == 10
    assert all(
        eval_env is not train_env
        for eval_env, train_env in zip(
            bundle.evaluation_envs,
            bundle.train_env.envs,
            strict=True,
        )
    )

    observation, info = bundle.train_env.reset(seed=23)
    if protocol == "task_agnostic":
        assert info == {}
        assert observation.shape == bundle.evaluation_envs[0].observation_space.shape
    else:
        assert info["seq_idx"] == 0
        assert observation.shape[-1] > 10
        torch.testing.assert_close(
            observation[-10:],
            torch.nn.functional.one_hot(
                torch.tensor(0),
                num_classes=10,
            ).float(),
        )
