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
