"""Task-aware multi-head SAC for canonical Continual World."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.utils.checkpoint_transaction import transactional_state_load
from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.models.task_aware_sac import (
    TaskAwareQNetwork,
    TaskAwareSquashedGaussianActor,
)
from rl_bgd.replay.buffer import ReplayBatch


class TaskAwareSACAgent:
    """Canonical SAC with shared bodies and occurrence-specific heads."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        num_tasks: int,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: tuple[int, ...] = (256, 256, 256, 256),
        config: SACConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or SACConfig()
        self.config.validate()
        self.device = torch.device(device)
        self.num_tasks = num_tasks
        self.actor = TaskAwareSquashedGaussianActor(
            observation_dim,
            action_dim,
            num_tasks=num_tasks,
            action_low=action_low.to(self.device),
            action_high=action_high.to(self.device),
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic1 = TaskAwareQNetwork(
            observation_dim,
            action_dim,
            num_tasks=num_tasks,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.critic2 = TaskAwareQNetwork(
            observation_dim,
            action_dim,
            num_tasks=num_tasks,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.target1 = copy.deepcopy(self.critic1).eval()
        self.target2 = copy.deepcopy(self.critic2).eval()
        for target in (self.target1, self.target2):
            for parameter in target.parameters():
                parameter.requires_grad_(False)

        self.log_alpha = nn.Parameter(
            torch.log(
                torch.full(
                    (num_tasks,),
                    self.config.initial_alpha,
                    device=self.device,
                )
            )
        )
        self.target_entropy = (
            float(self.config.target_entropy)
            if self.config.target_entropy is not None
            else -float(action_dim)
        )
        self.update_count = 0
        self.optimizer_reset_count = 0
        self._create_optimizers()

    def _create_optimizers(self) -> None:
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.config.critic_lr,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.config.alpha_lr,
        )

    def reset_optimizers(self) -> None:
        self._create_optimizers()
        self.optimizer_reset_count += 1

    def _task_weights(self, observation: Tensor) -> Tensor:
        return observation[..., -self.num_tasks :]

    def _log_alpha_for(self, observation: Tensor) -> Tensor:
        return (self._task_weights(observation) * self.log_alpha).sum(dim=-1, keepdim=True)

    def _alpha_for(self, observation: Tensor) -> Tensor:
        if self.config.automatic_entropy_tuning:
            return self._log_alpha_for(observation).exp()
        return torch.full(
            (*observation.shape[:-1], 1),
            self.config.initial_alpha,
            device=observation.device,
            dtype=observation.dtype,
        )

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

    def _clip_gradients(
        self,
        parameters: list[nn.Parameter],
    ) -> None:
        if self.config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                parameters,
                self.config.gradient_clip_norm,
            )

    @torch.no_grad()
    def _target_values(
        self,
        batch: ReplayBatch,
    ) -> Tensor:
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
        target_q -= self._alpha_for(batch.next_observations) * next_log_prob
        return batch.rewards + self.config.gamma * batch.bootstrap_mask * target_q

    def update(
        self,
        batch: ReplayBatch,
    ) -> dict[str, float]:
        target = self._target_values(batch)
        q1 = self.critic1(
            batch.observations,
            batch.actions,
        )
        q2 = self.critic2(
            batch.observations,
            batch.actions,
        )
        critic_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(
            q2, target
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self._clip_gradients(critic_parameters)
        self.critic_optimizer.step()

        sampled_action, log_prob, _ = self.actor.sample(batch.observations)
        q_pi = torch.minimum(
            self.critic1(
                batch.observations,
                sampled_action,
            ),
            self.critic2(
                batch.observations,
                sampled_action,
            ),
        )
        actor_loss = (self._alpha_for(batch.observations).detach() * log_prob - q_pi).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip_gradients(list(self.actor.parameters()))
        self.actor_optimizer.step()

        alpha_loss = torch.zeros(
            (),
            device=self.device,
        )
        if self.config.automatic_entropy_tuning:
            alpha_loss = -(
                self._log_alpha_for(batch.observations) * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._polyak_update()
        self.update_count += 1
        metrics = {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(self._alpha_for(batch.observations).detach().mean().item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
            "target_q_mean": float(target.detach().mean().item()),
            "policy_entropy_estimate": float((-log_prob.detach()).mean().item()),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("nonfinite task-aware SAC update metric")
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
            "num_tasks": self.num_tasks,
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
            "optimizer_reset_count": self.optimizer_reset_count,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if state.get("checkpoint_version") != 2:
            raise ValueError("unsupported task-aware SAC checkpoint version")
        if state.get("config") != asdict(self.config):
            raise ValueError("task-aware SAC checkpoint configuration mismatch")
        if state.get("num_tasks") != self.num_tasks:
            raise ValueError("task-aware SAC checkpoint task-count mismatch")

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
            self.optimizer_reset_count = int(payload.get("optimizer_reset_count", 0))

        transactional_state_load(
            state,
            current_state=self.state_dict,
            apply=apply,
        )
