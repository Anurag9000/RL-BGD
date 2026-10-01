"""Task-label-free nonstationary Gaussian multi-armed bandit."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_bgd.continual.schedules import ContextSchedule


@dataclass(frozen=True)
class BanditEvaluationStep:
    """Evaluator-only oracle information for one completed interaction."""

    step: int
    action: int
    expected_reward: float
    oracle_expected_reward: float
    optimal_arm: int
    instantaneous_regret: float
    cumulative_regret: float


class ScheduledGaussianBandit:
    """K-armed Gaussian bandit driven by the generic hidden-context schedule.

    Schedule contexts must use contiguous keys arm_0, arm_1, and so on.
    Learner-facing pull returns only the sampled reward. Oracle means,
    optimal-arm identity, and regret remain available only through explicitly
    evaluator-facing properties.
    """

    def __init__(
        self,
        schedule: ContextSchedule,
        *,
        reward_std: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        if reward_std < 0:
            raise ValueError("reward_std cannot be negative")
        self.schedule = schedule
        self.reward_std = float(reward_std)
        self.device = torch.device(device)
        self._arm_keys = self._validate_arm_keys(
            tuple(schedule.config.anchors[0])
        )
        self.num_arms = len(self._arm_keys)
        self._generator = torch.Generator(device=self.device)
        self.environment_step = 0
        self.cumulative_regret = 0.0
        self._last_evaluation: BanditEvaluationStep | None = None

    @staticmethod
    def _validate_arm_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
        expected = tuple(
            f"arm_{index}"
            for index in range(len(keys))
        )
        if len(keys) < 2 or set(keys) != set(expected):
            raise ValueError(
                "bandit contexts require contiguous arm_0...arm_K-1 keys"
            )
        return expected

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            self._generator.manual_seed(seed)
        self.environment_step = 0
        self.cumulative_regret = 0.0
        self._last_evaluation = None

    def _means(
        self,
        step: int,
    ) -> Tensor:
        context = self.schedule.context_at(step)
        return torch.tensor(
            [context[key] for key in self._arm_keys],
            device=self.device,
            dtype=torch.float32,
        )

    def _action_index(
        self,
        action: int | Tensor,
    ) -> int:
        if isinstance(action, Tensor):
            if action.numel() != 1:
                raise ValueError("bandit action tensor must contain one index")
            index = int(action.detach().item())
        else:
            index = int(action)
        if not 0 <= index < self.num_arms:
            raise ValueError(
                f"bandit action {index} is outside [0, {self.num_arms})"
            )
        return index

    def pull(
        self,
        action: int | Tensor,
    ) -> float:
        """Sample one reward without exposing hidden context or oracle regret."""

        index = self._action_index(action)
        step = self.environment_step
        means = self._means(step)
        expected_reward = means[index]
        oracle_expected_reward, optimal_arm = torch.max(means, dim=0)
        regret = oracle_expected_reward - expected_reward
        noise = torch.zeros((), device=self.device)
        if self.reward_std:
            noise = self.reward_std * torch.randn(
                (),
                device=self.device,
                generator=self._generator,
            )
        reward = expected_reward + noise
        self.cumulative_regret += float(regret.item())
        self._last_evaluation = BanditEvaluationStep(
            step=step,
            action=index,
            expected_reward=float(expected_reward.item()),
            oracle_expected_reward=float(oracle_expected_reward.item()),
            optimal_arm=int(optimal_arm.item()),
            instantaneous_regret=float(regret.item()),
            cumulative_regret=self.cumulative_regret,
        )
        self.environment_step += 1
        return float(reward.item())

    @property
    def evaluation_context(self) -> dict[str, object]:
        """Current hidden context for evaluator/debug use only."""

        means = self._means(self.environment_step)
        oracle_expected_reward, optimal_arm = torch.max(means, dim=0)
        return {
            "environment_step": self.environment_step,
            "arm_means": means.clone(),
            "optimal_arm": int(optimal_arm.item()),
            "oracle_expected_reward": float(oracle_expected_reward.item()),
            "cumulative_regret": self.cumulative_regret,
        }

    @property
    def last_evaluation(self) -> BanditEvaluationStep:
        """Return oracle statistics for the most recent completed pull."""

        if self._last_evaluation is None:
            raise RuntimeError("bandit has no completed interaction")
        return self._last_evaluation
