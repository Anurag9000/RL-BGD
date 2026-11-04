from rl_bgd.runners.quadratic import (
    QuadraticRunConfig,
    run_quadratic,
)


def test_quadratic_runner_executes() -> None:
    result = run_quadratic(
        QuadraticRunConfig(
            dimension=4,
            segment_steps=4,
            segments=2,
            mc_samples=2,
            seed=1,
            device="cpu",
        )
    )
    assert result["total_steps"] == 8
    assert len(result["boundaries"]) == 1
    assert result["sigma_mean"] > 0


def test_quadratic_suite_adapter_retains_structured_defaults() -> None:
    from dataclasses import asdict
    from unittest.mock import patch

    from rl_bgd.runners.quadratic import run_quadratic_suite

    with patch(
        "rl_bgd.runners.quadratic.run_quadratic",
        return_value={"final_loss": 2.0, "sigma_mean": 0.4},
    ) as original:
        output = run_quadratic_suite()
    original.assert_called_once()
    assert asdict(original.call_args.args[0]) == asdict(QuadraticRunConfig())
    assert output == {"final_loss": 2.0, "sigma_mean": 0.4}


def test_quadratic_suite_adapter_reuses_native_learning_path() -> None:
    from rl_bgd.runners.quadratic import run_quadratic_suite

    result = run_quadratic_suite(
        dimension=4,
        segment_steps=4,
        segments=2,
        mc_samples=2,
        seed=1,
        device="cpu",
    )
    assert result["total_steps"] == 8
    assert len(result["boundaries"]) == 1
    assert result["final_loss"] >= 0
    assert result["sigma_mean"] > 0
