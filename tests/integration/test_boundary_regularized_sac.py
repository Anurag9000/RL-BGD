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


def test_oracle_zero_strength_sham_preserves_original_training_path() -> None:
    from unittest.mock import patch

    with patch(
        "rl_bgd.runners.regularized_sac_lqr.train_boundary_regularized_sac",
        return_value={"completed": True, "final_10_mean_return": 0.0},
    ) as run:
        result = run_boundary_regularized_sac_recurring_lqr(
            method="ewc",
            regularization_strength=0.0,
            steps=64,
            phase_steps=32,
            seed=82,
            device="cpu",
        )
    assert result["regularization_strength"] == 0.0
    assert result["information_access"]["receives_task_boundary"] is True
    assert run.call_args.args[1].regularization_config.strength == 0.0
    assert run.call_args.kwargs["config"].consolidation_steps == (32,)


def test_oracle_regularization_strength_rejects_negative_values() -> None:
    import pytest

    with pytest.raises(ValueError, match="regularization strength"):
        run_boundary_regularized_sac_recurring_lqr(
            regularization_strength=-0.01,
            steps=64,
            phase_steps=32,
            seed=82,
            device="cpu",
        )
