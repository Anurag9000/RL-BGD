import pytest
import torch

from rl_bgd.continual.schedules import ContextSchedule, ContextScheduleConfig
from rl_bgd.envs.synthetic.bandit import ScheduledGaussianBandit


def _recurring_schedule() -> ContextSchedule:
    return ContextSchedule(
        ContextScheduleConfig(
            mode="recurring",
            anchors=(
                {
                    "arm_0": 1.0,
                    "arm_1": 0.0,
                    "arm_2": -1.0,
                },
                {
                    "arm_0": -0.5,
                    "arm_1": 2.0,
                    "arm_2": 0.5,
                },
            ),
            phase_steps=2,
        )
    )


def test_bandit_exact_reward_and_oracle_regret() -> None:
    env = ScheduledGaussianBandit(
        _recurring_schedule(),
        reward_std=0.0,
    )
    env.reset(seed=3)
    reward = env.pull(1)
    assert reward == pytest.approx(0.0)
    evaluation = env.last_evaluation
    assert evaluation.step == 0
    assert evaluation.action == 1
    assert evaluation.optimal_arm == 0
    assert evaluation.oracle_expected_reward == pytest.approx(1.0)
    assert evaluation.instantaneous_regret == pytest.approx(1.0)
    assert evaluation.cumulative_regret == pytest.approx(1.0)


def test_bandit_hidden_context_switch_is_evaluator_only() -> None:
    env = ScheduledGaussianBandit(
        _recurring_schedule(),
        reward_std=0.0,
    )
    env.reset(seed=4)
    rewards = [
        env.pull(0),
        env.pull(0),
        env.pull(0),
    ]
    assert rewards == pytest.approx([1.0, 1.0, -0.5])
    assert env.evaluation_context["optimal_arm"] == 1
    assert env.last_evaluation.optimal_arm == 1
    assert env.last_evaluation.instantaneous_regret == pytest.approx(2.5)


def test_bandit_optimal_action_has_zero_expected_regret() -> None:
    env = ScheduledGaussianBandit(
        _recurring_schedule(),
        reward_std=0.0,
    )
    env.reset(seed=5)
    env.pull(0)
    assert env.last_evaluation.instantaneous_regret == 0.0
    assert env.last_evaluation.cumulative_regret == 0.0


def test_bandit_stochastic_rewards_are_seed_reproducible() -> None:
    left = ScheduledGaussianBandit(
        _recurring_schedule(),
        reward_std=0.25,
    )
    right = ScheduledGaussianBandit(
        _recurring_schedule(),
        reward_std=0.25,
    )
    left.reset(seed=17)
    right.reset(seed=17)
    left_rewards = [left.pull(0) for _ in range(6)]
    right_rewards = [right.pull(0) for _ in range(6)]
    assert left_rewards == pytest.approx(right_rewards)


def test_bandit_supports_smooth_context_schedule() -> None:
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="smooth",
            anchors=(
                {"arm_0": 0.0, "arm_1": 1.0},
                {"arm_0": 2.0, "arm_1": 0.0},
            ),
            phase_steps=4,
        )
    )
    env = ScheduledGaussianBandit(schedule)
    env.reset(seed=1)
    env.pull(torch.tensor(0))
    env.pull(0)
    env.pull(0)
    assert env.last_evaluation.expected_reward == pytest.approx(1.0)
    assert env.last_evaluation.instantaneous_regret == 0.0


def test_bandit_rejects_noncontiguous_arm_keys() -> None:
    schedule = ContextSchedule(
        ContextScheduleConfig(
            mode="abrupt",
            anchors=({"arm_0": 0.0, "arm_2": 1.0},),
        )
    )
    with pytest.raises(ValueError):
        ScheduledGaussianBandit(schedule)
