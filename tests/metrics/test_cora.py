import pytest

from rl_bgd.metrics.cora import (
    CORA_ATARI_6_TASKS_5_CYCLES,
    CORATrace,
    cora_isolated_forgetting,
    cora_isolated_zero_shot_forward_transfer,
)


def test_cora_protocol_regions_match_cycle_order() -> None:
    protocol = CORA_ATARI_6_TASKS_5_CYCLES
    regions = protocol.training_regions(2)
    assert regions[0] == (100_000_000, 150_000_000)
    assert regions[1] == (400_000_000, 450_000_000)
    assert protocol.total_steps == 1_500_000_000


def test_cora_isolated_forgetting_uses_pre_stage_last_value() -> None:
    trace = CORATrace.from_sequences(
        [0, 5, 15, 25, 35],
        [0.0, 10.0, 7.0, 4.0, 3.0],
    )
    result = cora_isolated_forgetting(
        [trace],
        task_steps=10,
        task_id=0,
        num_tasks=3,
        num_cycles=1,
        return_scale=1.0,
    )
    assert result[1][0] == [pytest.approx(3.0)]
    assert result[2][0] == [pytest.approx(3.0)]


def test_cora_isolated_zero_shot_transfer_rebaselines_each_stage() -> None:
    trace = CORATrace.from_sequences(
        [0, 5, 15, 25],
        [2.0, 5.0, 9.0, 8.0],
    )
    result = cora_isolated_zero_shot_forward_transfer(
        [trace],
        task_steps=10,
        prior_task_ids=[0, 1],
        return_scale=1.0,
    )
    assert result[0] == [pytest.approx(3.0)]
    assert result[1] == [pytest.approx(4.0)]
