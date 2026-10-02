import math

from rl_bgd.runners.ucl_ppo_lqr import (
    run_adam_ppo_oracle_recurring_lqr_control,
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


def test_ucl_and_adam_control_share_exact_oracle_profile() -> None:
    control = run_adam_ppo_oracle_recurring_lqr_control(
        phase_steps=32,
        phases=3,
        seed=145,
        device="cpu",
        horizon=24,
    )
    ucl = run_ucl_ppo_recurring_lqr(
        phase_steps=32,
        phases=3,
        seed=145,
        device="cpu",
        horizon=24,
    )

    assert control["benchmark_profile"] == ucl["benchmark_profile"] == "recurring_lqr_matched_v1"
    assert control["phase_steps"] == ucl["phase_steps"] == 32
    assert control["phases"] == ucl["phases"] == 3
    assert control["horizon"] == ucl["horizon"] == 24
    assert control["final_evaluation_context"] == ucl["final_evaluation_context"]
    assert math.isfinite(float(control["final_phase_return"]))
    assert math.isfinite(float(ucl["final_phase_return"]))
    control_access = control["information_access"]
    assert control_access["receives_task_id"] is False
    assert control_access["receives_task_boundary"] is True
    assert control_access["receives_environment_context"] is False
