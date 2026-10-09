import pytest
import torch

from rl_bgd.surprise.base import (
    EMANormalizerConfig,
    EMASurpriseNormalizer,
    RetentionMappingConfig,
    surprise_to_retention,
)
from rl_bgd.surprise.td import TDSurprise, TDSurpriseConfig


def test_retention_mapping_is_monotone_and_bounded() -> None:
    config = RetentionMappingConfig(lambda_min=0.4, kappa=2.0)
    values = [surprise_to_retention(x, config) for x in (0.0, 0.5, 2.0, 10.0)]
    assert values[0] == pytest.approx(1.0)
    assert values == sorted(values, reverse=True)
    assert all(0.4 <= value <= 1.0 for value in values)


def test_td_surprise_spike_increases_normalized_surprise() -> None:
    estimator = TDSurprise(
        TDSurpriseConfig(
            normalizer=EMANormalizerConfig(
                decay=0.9,
                smoothing_decay=0.0,
                initial_variance=0.01,
            )
        )
    )
    estimator.observe(torch.tensor([1.0, 1.0, 1.0]))
    stable = estimator.observe(torch.tensor([1.0, 1.0, 1.0]))
    spike = estimator.observe(torch.tensor([4.0, 4.0, 4.0]))
    assert stable.normalized == pytest.approx(0.0)
    assert spike.normalized > 5.0


def test_surprise_normalizer_checkpoint_rejects_config_mismatch() -> None:
    source = EMASurpriseNormalizer(
        EMANormalizerConfig(
            decay=0.9,
            smoothing_decay=0.5,
        )
    )
    source.observe(1.0)
    state = source.state_dict()

    restored = EMASurpriseNormalizer(
        EMANormalizerConfig(
            decay=0.8,
            smoothing_decay=0.5,
        )
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        restored.load_state_dict(state)


def test_td_surprise_checkpoint_rejects_aggregation_mismatch() -> None:
    source = TDSurprise(
        TDSurpriseConfig(
            aggregation="median_abs",
            normalizer=EMANormalizerConfig(decay=0.9),
        )
    )
    source.observe(torch.tensor([1.0, 2.0, 3.0]))
    state = source.state_dict()

    restored = TDSurprise(
        TDSurpriseConfig(
            aggregation="mean_abs",
            normalizer=EMANormalizerConfig(decay=0.9),
        )
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        restored.load_state_dict(state)


def test_td_surprise_checkpoint_round_trip_preserves_future_statistics() -> None:
    config = TDSurpriseConfig(
        aggregation="median_abs",
        normalizer=EMANormalizerConfig(
            decay=0.9,
            smoothing_decay=0.5,
        ),
    )
    source = TDSurprise(config)
    source.observe(torch.tensor([1.0, 2.0, 3.0]))
    state = source.state_dict()

    restored = TDSurprise(config)
    restored.load_state_dict(state)

    expected = source.observe(torch.tensor([4.0, 5.0, 6.0]))
    actual = restored.observe(torch.tensor([4.0, 5.0, 6.0]))
    assert actual == expected


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("initial_variance", float("nan")),
        ("epsilon", float("inf")),
        ("decay", True),
        ("smoothing_decay", "0.9"),
    ],
)
def test_surprise_configs_reject_nonfinite_and_coerced_values(
    field: str,
    invalid: object,
) -> None:
    kwargs = {field: invalid}
    with pytest.raises((TypeError, ValueError)):
        EMASurpriseNormalizer(EMANormalizerConfig(**kwargs))


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("kappa", float("nan")), ("lambda_min", float("inf")), ("kappa", "1")],
)
def test_retention_config_rejects_nonfinite_and_coerced_values(
    field: str,
    invalid: object,
) -> None:
    kwargs = {field: invalid}
    with pytest.raises((TypeError, ValueError)):
        RetentionMappingConfig(**kwargs).validate()


def test_surprise_normalizer_runtime_overflow_is_non_mutating() -> None:
    normalizer = EMASurpriseNormalizer()
    normalizer.observe(1e308)
    before = normalizer.state_dict()

    with pytest.raises(FloatingPointError, match="non-finite statistics"):
        normalizer.observe(-1e308)

    assert normalizer.state_dict() == before


@pytest.mark.parametrize("invalid", [True, "1.0"])
def test_surprise_normalizer_rejects_coerced_observations(invalid: object) -> None:
    normalizer = EMASurpriseNormalizer()
    before = normalizer.state_dict()
    with pytest.raises(TypeError, match="must be numeric"):
        normalizer.observe(invalid)
    assert normalizer.state_dict() == before
