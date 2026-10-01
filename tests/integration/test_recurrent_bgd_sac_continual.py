import torch

from rl_bgd.runners.recurrent_sac_continual_lqr import (
    run_recurrent_sac_recurring_lqr,
)


def test_recurrent_bgd_sac_recurring_stream_is_finite() -> None:
    result = run_recurrent_sac_recurring_lqr(
        steps=96,
        seed=88,
        device="cpu",
        optimizer="bgd",
    )
    metrics = result[
        "training"
    ][
        "last_update_metrics"
    ]
    assert metrics[
        "actor_sigma_mean"
    ] > 0.0
    assert metrics[
        "critic1_sigma_mean"
    ] > 0.0
    assert torch.isfinite(
        torch.tensor(
            metrics[
                "critic_loss"
            ]
        )
    )
