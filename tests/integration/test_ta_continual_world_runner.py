import pytest

from rl_bgd.runners.continual_world_sac import run_ta_continual_world_sac


@pytest.mark.benchmark
def test_ta_cw10_runner_builds_complete_evaluation_matrix() -> None:
    pytest.importorskip("metaworld")
    result = run_ta_continual_world_sac(
        benchmark="CW10",
        optimizer="adam",
        steps_per_task=1,
        seed=101,
        device="cpu",
        evaluation_episodes=1,
        episode_horizon=1,
        hidden_dims=(16, 16),
        replay_capacity=16,
        batch_size=2,
        random_steps=10,
    )
    assert len(result["task_names"]) == 10
    assert len(result["return_matrix"]) == 10
    assert len(result["success_matrix"]) == 10
    assert all(len(row) == 10 for row in result["return_matrix"])
    assert all(len(row) == 10 for row in result["success_matrix"])
