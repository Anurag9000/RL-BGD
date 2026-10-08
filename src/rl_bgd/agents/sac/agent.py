"""Inspectable Soft Actor-Critic implementation used as the RL baseline."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.utils.checkpoint_transaction import transactional_state_load
from rl_bgd.models.actor import SquashedGaussianActor
from rl_bgd.models.critic import QNetwork
from rl_bgd.replay.buffer import ReplayBatch


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    target_entropy: float | None = None
    gradient_clip_norm: float | None = None

    def validate(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must lie in (0, 1]")
        if (
            min(
                self.actor_lr,
                self.critic_lr,
                self.alpha_lr,
                self.initial_alpha,
            )
            <= 0
        ):
            raise ValueError("learning rates and alpha must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


class SACAgent:
    """Twin-critic SAC with optional automatic entropy tuning."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: tuple[int, ...] = (256, 256),
        config: SACConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or SACConfig()
        self.config.validate()
        self.device = torch.device(device)
        low = action_low.to(self.device, dtype=torch.float32)
        high = action_high.to(self.device, dtype=torch.float32)
        self.actor = SquashedGaussianActor(
            observation_dim,
            action_dim,
            action_low=low,
            action_high=high,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic1 = QNetwork(
            observation_dim,
            action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic2 = QNetwork(
            observation_dim,
            action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.target1 = copy.deepcopy(self.critic1).eval()
        self.target2 = copy.deepcopy(self.critic2).eval()
        for target in (self.target1, self.target2):
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
        )
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters,
            lr=self.config.critic_lr,
        )
        self.target_entropy = (
            float(self.config.target_entropy)
            if self.config.target_entropy is not None
            else -float(action_dim)
        )
        initial_log_alpha = torch.log(
            torch.tensor(
                self.config.initial_alpha,
                device=self.device,
            )
        )
        self.log_alpha = nn.Parameter(initial_log_alpha)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.config.alpha_lr,
        )
        self.update_count = 0

    @property
    def alpha(self) -> Tensor:
        if self.config.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return torch.tensor(
            self.config.initial_alpha,
            device=self.device,
        )

    @torch.no_grad()
    def act(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor:
        obs = observation.to(self.device, dtype=torch.float32)
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(obs)
        else:
            action, _, _ = self.actor.sample(obs)
        return action.squeeze(0) if squeeze else action

    def _clip_gradients(self, parameters: list[nn.Parameter]) -> None:
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                self.config.gradient_clip_norm,
            )

    @torch.no_grad()
    def _target_values(self, batch: ReplayBatch) -> Tensor:
        next_action, next_log_prob, _ = self.actor.sample(batch.next_observations)
        target_q = torch.minimum(
            self.target1(
                batch.next_observations,
                next_action,
            ),
            self.target2(
                batch.next_observations,
                next_action,
            ),
        )
        target_q = target_q - self.alpha.detach() * next_log_prob
        return batch.rewards + self.config.gamma * batch.bootstrap_mask * target_q

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        target = self._target_values(batch)
        q1 = self.critic1(batch.observations, batch.actions)
        q2 = self.critic2(batch.observations, batch.actions)
        critic1_loss = torch.nn.functional.mse_loss(q1, target)
        critic2_loss = torch.nn.functional.mse_loss(q2, target)
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self._clip_gradients(critic_params)
        self.critic_optimizer.step()

        sampled_action, log_prob, _ = self.actor.sample(batch.observations)
        q_pi = torch.minimum(
            self.critic1(batch.observations, sampled_action),
            self.critic2(batch.observations, sampled_action),
        )
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip_gradients(list(self.actor.parameters()))
        self.actor_optimizer.step()

        alpha_loss = torch.zeros((), device=self.device)
        if self.config.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._polyak_update()
        self.update_count += 1
        metrics = {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
            "target_q_mean": float(target.detach().mean().item()),
            "policy_entropy_estimate": float((-log_prob.detach()).mean().item()),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("nonfinite SAC update metric")
        return metrics

    @torch.no_grad()
    def _polyak_update(self) -> None:
        for source, target in zip(
            self.critic1.parameters(),
            self.target1.parameters(),
            strict=True,
        ):
            target.lerp_(source, self.config.tau)
        for source, target in zip(
            self.critic2.parameters(),
            self.target2.parameters(),
            strict=True,
        ):
            target.lerp_(source, self.config.tau)

    def state_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_version": 2,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(),
            "actor_optimizer": (self.actor_optimizer.state_dict()),
            "critic_optimizer": (self.critic_optimizer.state_dict()),
            "log_alpha": (self.log_alpha.detach().clone()),
            "alpha_optimizer": (self.alpha_optimizer.state_dict()),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("checkpoint_version") != 2:
            raise ValueError("unsupported SAC checkpoint version")
        if state.get("config") != asdict(self.config):
            raise ValueError("SAC checkpoint configuration mismatch")

        def apply(payload: dict[str, Any]) -> None:
            for name in (
                "actor",
                "critic1",
                "critic2",
                "target1",
                "target2",
            ):
                getattr(self, name).load_state_dict(payload[name])
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
            self.update_count = int(payload["update_count"])

        transactional_state_load(
            state,
            current_state=self.state_dict,
            apply=apply,
        )
