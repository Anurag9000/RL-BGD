import pytest

from rl_bgd.continual.information_access import InformationAccessConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"receives_task_id": True},
        {"receives_task_boundary": True},
        {"task_specific_heads": True},
        {"uses_task_balanced_replay": True, "replay": True},
        {"context_available": True},
    ],
)
def test_privileged_channels_fail_strict_mode(
    kwargs: dict[str, bool],
) -> None:
    with pytest.raises(ValueError, match="privileged information"):
        InformationAccessConfig(**kwargs).validate_strict_task_agnostic()
