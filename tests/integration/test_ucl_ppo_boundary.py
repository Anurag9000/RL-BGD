from rl_bgd.runners.ucl_ppo_lqr import (
    run_ucl_ppo_recurring_lqr,
)


def test_ucl_runner_declares_oracle_boundary_access() -> None:
    result = run_ucl_ppo_recurring_lqr(
        phase_steps=32,
        phases=3,
        seed=144,
        device="cpu",
    )
    assert result["protocol"] == "oracle_boundary"
    assert len(result["boundaries"]) == 2
    access = result["information_access"]
    assert access["receives_task_id"] is False
    assert access["receives_task_boundary"] is True
    assert access["task_specific_heads"] is False
    assert access["posterior_snapshot_trigger"] == "oracle_phase_boundary"
