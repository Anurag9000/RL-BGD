from rl_bgd.runners.late_plasticity import (
    run_late_plasticity_quadratic,
)


def _run(
    retention: float,
) -> dict[str, object]:
    return run_late_plasticity_quadratic(
        consolidation_retention=retention,
        dimension=8,
        consolidation_steps=120,
        adaptation_steps=16,
        eta=0.15,
        prior_std=0.5,
        shift=1.0,
        curvature=1.0,
        mc_samples=16,
        seed=17,
        device="cpu",
    )


def test_late_consolidation_reduces_vanilla_post_shift_plasticity() -> None:
    vanilla = _run(
        1.0
    )
    tempered = _run(
        0.97
    )

    assert (
        float(
            tempered[
                "pre_shift_sigma_mean"
            ]
        )
        > 1.5
        * float(
            vanilla[
                "pre_shift_sigma_mean"
            ]
        )
    )
    assert (
        float(
            tempered[
                "first_step_mean_movement"
            ]
        )
        > 2.5
        * float(
            vanilla[
                "first_step_mean_movement"
            ]
        )
    )
    assert float(
        tempered[
            "post_shift_normalized_auc"
        ]
    ) < float(
        vanilla[
            "post_shift_normalized_auc"
        ]
    )
    assert float(
        tempered[
            "recovery_fraction"
        ]
    ) > float(
        vanilla[
            "recovery_fraction"
        ]
    )


def test_plasticity_runner_holds_post_shift_update_rule_fixed() -> None:
    result = _run(
        1.0
    )

    assert (
        result[
            "hypothesis_id"
        ]
        == "C"
    )
    assert (
        result[
            "adaptation_retention"
        ]
        == 1.0
    )
    curve = result[
        "post_shift_loss_curve"
    ]
    assert isinstance(
        curve,
        list,
    )
    assert len(
        curve
    ) == 17
    assert all(
        isinstance(
            value,
            float,
        )
        for value in curve
    )
