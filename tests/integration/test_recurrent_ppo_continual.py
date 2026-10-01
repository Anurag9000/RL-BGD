from rl_bgd.runners.recurrent_ppo_continual_lqr import (
    run_recurrent_ppo_recurring_lqr,
)


def test_recurrent_hidden_state_does_not_reset_on_context_switches() -> None:
    result = run_recurrent_ppo_recurring_lqr(
        steps=128,
        seed=73,
        device="cpu",
        optimizer="adam",
    )
    access = result["information_access"]
    assert access["receives_task_id"] is False
    assert access["receives_task_boundary"] is False
    assert access["receives_context"] is False
    assert access["hidden_state_resets_only_on_episode_end"] is True
