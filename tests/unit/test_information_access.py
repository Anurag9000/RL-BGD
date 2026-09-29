import pytest

from rl_bgd.continual.information_access import InformationAccessConfig


def test_strict_task_agnostic_default_is_valid() -> None:
    InformationAccessConfig().validate_strict_task_agnostic()


def test_strict_task_agnostic_rejects_task_id() -> None:
    with pytest.raises(ValueError, match="receives_task_id"):
        InformationAccessConfig(
            receives_task_id=True
        ).validate_strict_task_agnostic()


def test_replay_size_requires_replay() -> None:
    with pytest.raises(ValueError, match="replay_size"):
        InformationAccessConfig(
            replay=False, replay_size=10
        ).validate_strict_task_agnostic()
