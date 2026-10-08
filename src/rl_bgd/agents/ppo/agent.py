"""Inspectable continuous-action PPO baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.agents.ppo.rollout import (
    PPORolloutBatch,
    RolloutBuffer,
)
from rl_bgd.utils.checkpoint_transaction import transactional_state_load
from rl_bgd.models.ppo import (
    PPOSquashedGaussianActor,
    ValueNetwork,
)


@dataclass(frozen=True)
class PPOConfig:
    """PPO optimization and estimator controls."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: float | None = 0.2
    actor_lr: float = 3e-4
    value_lr: float = 1e-3
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 64
    gradient_clip_norm: float = 0.5
    normalize_advantages: bool = True
    target_kl: float | None = None

    def validate(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must lie in (0, 1)")
        if self.value_clip_ratio is not None and not 0.0 < self.value_clip_ratio < 1.0:
            raise ValueError("value_clip_ratio must lie in (0, 1)")
        if (
            min(
                self.actor_lr,
                self.value_lr,
                self.value_coef,
                self.gradient_clip_norm,
            )
            <= 0
            or self.entropy_coef < 0
        ):
            raise ValueError("invalid PPO optimization coefficients")
        if self.update_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("update_epochs and minibatch_size must be positive")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be positive")


class PPOAgent:
    """Clipped-surrogate PPO with GAE and separate value optimization."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: tuple[int, ...] = (256, 256),
        config: PPOConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or PPOConfig()
        self.config.validate()
        self.device = torch.device(device)

        self.actor = PPOSquashedGaussianActor(
            observation_dim,
            action_dim,
            action_low=action_low.to(self.device),
            action_high=action_high.to(self.device),
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.value = ValueNetwork(
            observation_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(),
            lr=self.config.value_lr,
        )
        self.update_count = 0

    @torch.no_grad()
    def act(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor:
        obs = observation.to(
            self.device,
            dtype=torch.float32,
        )
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(obs)
        else:
            action, _, _ = self.actor.sample(obs)
        return action.squeeze(0) if squeeze else action

    @torch.no_grad()
    def sample_action(
        self,
        observation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs = observation.to(
            self.device,
            dtype=torch.float32,
        )
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        action, log_prob, _ = self.actor.sample(obs)
        value = self.value(obs)
        if squeeze:
            return (
                action.squeeze(0),
                log_prob.squeeze(0),
                value.squeeze(0),
            )
        return action, log_prob, value

    @torch.no_grad()
    def value_of(self, observation: Tensor) -> Tensor:
        obs = observation.to(
            self.device,
            dtype=torch.float32,
        )
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        value = self.value(obs)
        return value.squeeze(0) if squeeze else value

    def _actor_loss(
        self,
        batch: PPORolloutBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        log_prob, entropy = self.actor.evaluate_actions(
            batch.observations,
            batch.actions,
        )
        log_ratio = log_prob - batch.old_log_probs
        ratio = log_ratio.exp()
        surrogate1 = ratio * batch.advantages
        surrogate2 = (
            ratio.clamp(
                1.0 - self.config.clip_ratio,
                1.0 + self.config.clip_ratio,
            )
            * batch.advantages
        )
        policy_loss = -torch.minimum(
            surrogate1,
            surrogate2,
        ).mean()
        entropy_mean = entropy.mean()
        actor_loss = policy_loss - self.config.entropy_coef * entropy_mean
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > self.config.clip_ratio).float().mean()
        return (
            actor_loss,
            policy_loss,
            entropy_mean,
            approx_kl,
            clip_fraction,
        )

    def _value_loss(
        self,
        batch: PPORolloutBatch,
    ) -> Tensor:
        prediction = self.value(batch.observations)
        if self.config.value_clip_ratio is None:
            return 0.5 * torch.nn.functional.mse_loss(
                prediction,
                batch.returns,
            )
        clipped = batch.old_values + (prediction - batch.old_values).clamp(
            -self.config.value_clip_ratio,
            self.config.value_clip_ratio,
        )
        plain_loss = (prediction - batch.returns).square()
        clipped_loss = (clipped - batch.returns).square()
        return (
            0.5
            * torch.maximum(
                plain_loss,
                clipped_loss,
            ).mean()
        )

    def update(
        self,
        rollout: RolloutBuffer,
    ) -> dict[str, float]:
        rollout.compute_gae(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize_advantages=self.config.normalize_advantages,
        )
        generator = torch.Generator(device=self.device).manual_seed(self.update_count + 12_345)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        minibatches = 0
        stop_early = False
        for _ in range(self.config.update_epochs):
            for batch in rollout.batches(
                self.config.minibatch_size,
                generator=generator,
            ):
                (
                    actor_loss,
                    policy_loss,
                    entropy,
                    approx_kl,
                    clip_fraction,
                ) = self._actor_loss(batch)
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.actor_optimizer.step()

                value_loss = self._value_loss(batch)
                self.value_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.value.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.value_optimizer.step()

                for name, value in (
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy),
                    ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction),
                ):
                    totals[name] += float(value.detach().item())
                minibatches += 1

                if (
                    self.config.target_kl is not None
                    and float(approx_kl.detach().item()) > self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        if minibatches == 0:
            raise RuntimeError("PPO update produced no minibatches")
        self.update_count += 1
        metrics = {name: value / minibatches for name, value in totals.items()}
        metrics["epochs_early_stopped"] = float(stop_early)
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("nonfinite PPO update metric")
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_version": 2,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "actor_optimizer": (self.actor_optimizer.state_dict()),
            "value_optimizer": (self.value_optimizer.state_dict()),
            "update_count": self.update_count,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if state.get("checkpoint_version") != 2:
            raise ValueError("unsupported PPO checkpoint version")
        if state.get("config") != asdict(self.config):
            raise ValueError("PPO checkpoint configuration mismatch")

        def apply(payload: dict[str, Any]) -> None:
            self.actor.load_state_dict(payload["actor"])
            self.value.load_state_dict(payload["value"])
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.value_optimizer.load_state_dict(payload["value_optimizer"])
            self.update_count = int(payload["update_count"])

        transactional_state_load(
            state,
            current_state=self.state_dict,
            apply=apply,
        )
