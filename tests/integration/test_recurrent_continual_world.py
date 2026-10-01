import pytest

from rl_bgd.runners.recurrent_continual_world import (
    run_recurrent_ta_continual_world_sac,
)


@pytest.mark.benchmark
def test_recurrent_ta_cw10_builds_full_matrices_without_task_id() -> None:
    pytest.importorskip("metaworld")
    result = run_recurrent_ta_continual_world_sac(
        benchmark="CW10",
        optimizer="adam",
        steps_per_task=1,
        seed=113,
        device="cpu",
        evaluation_episodes=1,
        episode_horizon=1,
        replay_capacity=64,
        initial_random_steps=10,
        sequence_batch_size=2,
        history_length=2,
        unroll=1,
    )
    assert result["protocol_label"] == "3RL-style"
    assert len(result["return_matrix"]) == 10
    assert len(result["success_matrix"]) == 10
    assert result["information_access"]["receives_task_id"] is False
    assert result["information_access"]["receives_task_boundary"] is False
