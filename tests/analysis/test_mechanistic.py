from pathlib import Path

from rl_bgd.analysis.mechanistic import (
    MechanisticAnalysisConfig,
    run_mechanistic_analysis,
)


def test_mechanistic_suite_writes_reproducible_artifacts(
    tmp_path: Path,
) -> None:
    summary = run_mechanistic_analysis(
        tmp_path,
        config=MechanisticAnalysisConfig(
            dimension=8,
            consolidation_steps=10,
            adaptation_steps=4,
            mc_samples=16,
            curvature_samples=64,
            eta=0.15,
            prior_std=0.5,
            seed=151,
            device="cpu",
        ),
    )
    expected = {
        "mechanistic_parameters.csv",
        "freezing_counterfactual.csv",
        "movement_vs_sigma.png",
        "perturbation_vs_precision.png",
        "freezing_counterfactual.png",
        "curvature_signal.png",
        "uncertainty_quality.png",
        "mechanistic_summary.json",
    }
    assert set(summary["artifacts"]) == expected
    for name in expected:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0
    assert summary["movement_sigma_spearman"] > 0.0
    assert summary["perturbation_precision_spearman"] > 0.0
    assert summary["perturbation_sigma_spearman"] < 0.0
    assert summary["curvature_signal_mean_relative_error"] < 0.5
