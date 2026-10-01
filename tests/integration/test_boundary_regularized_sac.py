from rl_bgd.runners.regularized_sac_lqr import (
    run_boundary_regularized_sac_recurring_lqr,
)


def test_boundary_regularized_runner_declares_oracle_boundary_access() -> None:
    result = run_boundary_regularized_sac_recurring_lqr(
        method="ewc",
        steps=96,
        phase_steps=48,
        seed=85,
        device="cpu",
    )
    assert len(result["training"]["consolidations"]) == 1
    access = result["information_access"]
    assert access["receives_task_boundary"] is True
    assert access["receives_task_id"] is False
    assert access["task_specific_heads"] is False
    assert access["consolidation_trigger"] == "oracle_phase_boundary"
