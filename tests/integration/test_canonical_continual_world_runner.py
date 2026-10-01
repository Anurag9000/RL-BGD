import pytest

from rl_bgd.runners.canonical_continual_world_sac import (
    run_canonical_continual_world_sac,
)


@pytest.mark.benchmark
def test_canonical_cw10_runner_builds_full_performance_matrices() -> None:
    pytest.importorskip("metaworld")
    result = run_canonical_continual_world_sac(
        benchmark="CW10",
        steps_per_task=1,
        seed=37,
        device="cpu",
        evaluation_episodes=1,
        episode_horizon=1,
        hidden_dims=(8,),
        replay_capacity=16,
        batch_size=1,
        start_steps=0,
        update_after=1_000,
        update_every=1,
    )
    assert result["protocol"] == "canonical"
    assert result["architecture"] == "multihead"
    assert len(result["return_matrix"]) == 10
    assert all(len(row) == 10 for row in result["return_matrix"])
    assert len(result["success_matrix"]) == 10
    assert all(len(row) == 10 for row in result["success_matrix"])
    assert result["training"]["optimizer_resets"] == 10
