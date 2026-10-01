from rl_bgd.runners.hidden_context_sac_comparison import (
    run_hidden_context_sac_comparison,
)


def test_hidden_context_comparison_exposes_required_five_variants() -> None:
    result = run_hidden_context_sac_comparison(
        steps=56,
        seed=91,
        device="cpu",
    )
    variants = result["variants"]
    assert set(variants) == {
        "feedforward_adam",
        "feedforward_bgd",
        "recurrent_adam",
        "recurrent_bgd",
        "recurrent_adaptive_bgd",
    }
    assert result["comparison_contract"]["task_id_hidden"] is True
    assert result["comparison_contract"]["task_boundary_hidden"] is True
    assert (
        variants["recurrent_adaptive_bgd"]["training"]["last_update_metrics"]["retention_lambda"]
        <= 1.0
    )
