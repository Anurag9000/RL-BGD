import pytest

from rl_bgd.runners.sac_lqr import (
    run_sac_lqr,
)


@pytest.mark.slow
def test_sac_improves_stationary_lqr_return() -> None:
    result = run_sac_lqr(
        steps=800,
        seed=7,
        device="cpu",
    )
    assert (
        result["post_return"]
        > result["pre_return"] + 0.75
    )
