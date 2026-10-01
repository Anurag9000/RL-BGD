import pytest

from rl_bgd.analysis.statistics import (
    bootstrap_mean_ci,
    hierarchical_bootstrap_mean,
    paired_bootstrap_difference,
)


def test_bootstrap_mean_is_deterministic_and_reports_seed_count() -> None:
    first = bootstrap_mean_ci(
        [1.0, 2.0, 3.0, 4.0],
        resamples=500,
        seed=7,
    )
    second = bootstrap_mean_ci(
        [1.0, 2.0, 3.0, 4.0],
        resamples=500,
        seed=7,
    )
    assert first == second
    assert first.mean == pytest.approx(2.5)
    assert first.n == 4
    assert first.std > 0.0
    assert first.ci_low <= first.mean <= first.ci_high


def test_paired_bootstrap_uses_within_seed_differences() -> None:
    estimate = paired_bootstrap_difference(
        [3.0, 6.0, 9.0],
        [1.0, 4.0, 7.0],
        resamples=200,
        seed=11,
    )
    assert estimate.mean == pytest.approx(2.0)
    assert estimate.std == pytest.approx(0.0)
    assert estimate.ci_low == pytest.approx(2.0)
    assert estimate.ci_high == pytest.approx(2.0)


def test_hierarchical_bootstrap_weights_seeds_not_tasks_equally() -> None:
    estimate = hierarchical_bootstrap_mean(
        {
            0: [0.0, 2.0],
            1: [
                10.0,
                10.0,
                10.0,
                10.0,
            ],
        },
        resamples=300,
        seed=13,
    )
    # Seed means are 1 and 10, so the point estimate is 5.5 rather than
    # pooling six correlated task observations into a mean of 7.
    assert estimate.mean == pytest.approx(5.5)
    assert estimate.n == 2
    assert estimate.ci_low <= estimate.mean <= estimate.ci_high


def test_bootstrap_rejects_nonfinite_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0, float("nan")])
    with pytest.raises(ValueError):
        paired_bootstrap_difference(
            [1.0],
            [1.0, 2.0],
        )
    with pytest.raises(ValueError):
        hierarchical_bootstrap_mean({})
