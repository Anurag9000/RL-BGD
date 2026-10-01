from rl_bgd.runners.recurrent_sac_continual_lqr import (
    run_recurrent_sac_recurring_lqr,
)


def test_recurrent_sac_hidden_state_ignores_context_switches() -> None:
    result = run_recurrent_sac_recurring_lqr(
        steps=112,
        seed=87,
        device="cpu",
        optimizer="adam",
    )
    access = result[
        "information_access"
    ]
    assert (
        access[
            "receives_task_id"
        ]
        is False
    )
    assert (
        access[
            "receives_task_boundary"
        ]
        is False
    )
    assert (
        access[
            "receives_context"
        ]
        is False
    )
    assert access["recurrent_input_fields"] == [
        "observation",
        "previous_action",
        "previous_reward",
        "previous_done",
    ]
    assert (
        access[
            "hidden_state_resets_only_on_episode_end"
        ]
        is True
    )
