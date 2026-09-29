from rl_bgd.runners.smoke import run_smoke


def test_quadratic_smoke_cpu() -> None:
    result = run_smoke(steps=8, seed=0, device="cpu")
    assert result["final_abs_mean"] < result["initial_abs_mean"]
    assert result["diagnostics"]["sigma_mean"] > 0
