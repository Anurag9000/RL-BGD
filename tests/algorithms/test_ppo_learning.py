import pytest

from rl_bgd.runners.ppo_lqr import run_ppo_lqr


@pytest.mark.slow
def test_ppo_improves_stationary_lqr_return() -> None:
    result = run_ppo_lqr(
        steps=800,
        seed=19,
        device="cpu",
    )
    assert result["post_return"] > result["pre_return"] + 0.25
