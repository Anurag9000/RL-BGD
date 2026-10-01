"""Boundary-triggered EWC, Online-EWC, SI, and MAS SAC baselines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from rl_bgd.agents.sac.agent import SACAgent, SACConfig
from rl_bgd.baselines.ewc import EWCRegularizer, OnlineEWCRegularizer
from rl_bgd.baselines.importance import empirical_fisher_diagonal, mas_importance
from rl_bgd.baselines.mas import MASRegularizer
from rl_bgd.baselines.si import SynapticIntelligence
from rl_bgd.replay.buffer import ReplayBatch

RegularizerName = Literal["ewc", "online_ewc", "si", "mas"]
Regularizer = EWCRegularizer | OnlineEWCRegularizer | SynapticIntelligence | MASRegularizer


@dataclass(frozen=True)
class RegularizedSACConfig:
    """Continual-learning regularizer controls for SAC."""

    method: RegularizerName = "ewc"
    strength: float = 1.0
    online_ewc_decay: float = 1.0
    si_damping: float = 0.1
    importance_samples: int = 32
    regularize_actor: bool = True
    regularize_critics: bool = True

    def validate(self) -> None:
        if self.method not in {"ewc", "online_ewc", "si", "mas"}:
            raise ValueError(f"unsupported SAC regularizer: {self.method}")
        if self.strength < 0:
            raise ValueError("regularizer strength must be non-negative")
        if not 0.0 <= self.online_ewc_decay <= 1.0:
            raise ValueError("online_ewc_decay must lie in [0, 1]")
        if self.si_damping <= 0:
            raise ValueError("si_damping must be positive")
        if self.importance_samples < 1:
            raise ValueError("importance_samples must be positive")
        if not (self.regularize_actor or self.regularize_critics):
            raise ValueError("at least one SAC module must be regularized")


class RegularizedSACAgent(SACAgent):
    """SAC with externally triggered parameter-importance consolidation.

    A caller that supplies true task boundaries is an oracle/boundary-aware
    baseline. No task ID or task-specific routing is used by this agent.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: tuple[int, ...] = (256, 256),
        sac_config: SACConfig | None = None,
        regularizer_config: RegularizedSACConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dims=hidden_dims,
            config=sac_config,
            device=device,
        )
        self.regularizer_config = regularizer_config or RegularizedSACConfig()
        self.regularizer_config.validate()
        self.regularizers: dict[str, Regularizer] = {}
        if self.regularizer_config.regularize_actor:
            self.regularizers["actor"] = self._make_regularizer(self.actor)
        if self.regularizer_config.regularize_critics:
            self.regularizers["critic1"] = self._make_regularizer(self.critic1)
            self.regularizers["critic2"] = self._make_regularizer(self.critic2)
        self.consolidation_count = 0

    def _make_regularizer(self, module: nn.Module) -> Regularizer:
        config = self.regularizer_config
        if config.method == "ewc":
            return EWCRegularizer(strength=config.strength)
        if config.method == "online_ewc":
            return OnlineEWCRegularizer(
                strength=config.strength,
                decay=config.online_ewc_decay,
            )
        if config.method == "si":
            return SynapticIntelligence(
                module,
                strength=config.strength,
                damping=config.si_damping,
            )
        return MASRegularizer(strength=config.strength)

    def _regularizer_penalty(self, name: str, module: nn.Module) -> Tensor:
        regularizer = self.regularizers.get(name)
        if regularizer is None:
            return torch.zeros((), device=self.device)
        return regularizer.penalty(module)

    def _capture_si(
        self,
        name: str,
        module: nn.Module,
    ) -> dict[str, Tensor] | None:
        regularizer = self.regularizers.get(name)
        if not isinstance(regularizer, SynapticIntelligence):
            return None
        return regularizer.capture_gradients(module)

    def _accumulate_si(
        self,
        name: str,
        module: nn.Module,
        gradients: dict[str, Tensor] | None,
    ) -> None:
        if gradients is None:
            return
        regularizer = self.regularizers[name]
        if not isinstance(regularizer, SynapticIntelligence):
            raise RuntimeError("SI gradients attached to a non-SI regularizer")
        regularizer.accumulate_after_step(module, gradients)

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        target = self._target_values(batch)
        q1 = self.critic1(batch.observations, batch.actions)
        q2 = self.critic2(batch.observations, batch.actions)
        critic_task_loss = (
            torch.nn.functional.mse_loss(q1, target)
            + torch.nn.functional.mse_loss(q2, target)
        )
        critic1_penalty = self._regularizer_penalty("critic1", self.critic1)
        critic2_penalty = self._regularizer_penalty("critic2", self.critic2)
        critic_loss = critic_task_loss + critic1_penalty + critic2_penalty
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic1_si = self._capture_si("critic1", self.critic1)
        critic2_si = self._capture_si("critic2", self.critic2)
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self._clip_gradients(critic_params)
        self.critic_optimizer.step()
        self._accumulate_si("critic1", self.critic1, critic1_si)
        self._accumulate_si("critic2", self.critic2, critic2_si)

        sampled_action, log_prob, _ = self.actor.sample(batch.observations)
        q_pi = torch.minimum(
            self.critic1(batch.observations, sampled_action),
            self.critic2(batch.observations, sampled_action),
        )
        actor_task_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        actor_penalty = self._regularizer_penalty("actor", self.actor)
        actor_loss = actor_task_loss + actor_penalty
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_si = self._capture_si("actor", self.actor)
        self._clip_gradients(list(self.actor.parameters()))
        self.actor_optimizer.step()
        self._accumulate_si("actor", self.actor, actor_si)

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
            "critic_task_loss": float(critic_task_loss.detach().item()),
            "critic_regularizer_penalty": float(
                (critic1_penalty + critic2_penalty).detach().item()
            ),
            "actor_loss": float(actor_loss.detach().item()),
            "actor_task_loss": float(actor_task_loss.detach().item()),
            "actor_regularizer_penalty": float(actor_penalty.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
            "target_q_mean": float(target.detach().mean().item()),
            "policy_entropy_estimate": float((-log_prob.detach()).mean().item()),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("nonfinite regularized SAC update metric")
        return metrics

    def _sample_indices(self, batch: ReplayBatch) -> range:
        count = min(
            self.regularizer_config.importance_samples,
            int(batch.observations.shape[0]),
        )
        return range(count)

    def _fisher_closures(
        self,
        name: str,
        batch: ReplayBatch,
        target: Tensor,
    ) -> list[Callable[[], Tensor]]:
        closures: list[Callable[[], Tensor]] = []
        if name in {"critic1", "critic2"}:
            critic = getattr(self, name)
            for index in self._sample_indices(batch):

                def closure(
                    sample_index: int = index,
                    module: nn.Module = critic,
                ) -> Tensor:
                    prediction = module(
                        batch.observations[sample_index : sample_index + 1],
                        batch.actions[sample_index : sample_index + 1],
                    )
                    error = prediction - target[sample_index : sample_index + 1]
                    return 0.5 * error.square().sum()

                closures.append(closure)
            return closures

        for index in self._sample_indices(batch):

            def actor_closure(sample_index: int = index) -> Tensor:
                observation = batch.observations[sample_index : sample_index + 1]
                action, log_prob, _ = self.actor.sample(observation)
                q_value = torch.minimum(
                    self.critic1(observation, action),
                    self.critic2(observation, action),
                )
                return (self.alpha.detach() * log_prob - q_value).mean()

            closures.append(actor_closure)
        return closures

    def _mas_closures(
        self,
        name: str,
        batch: ReplayBatch,
    ) -> list[Callable[[], Tensor]]:
        closures: list[Callable[[], Tensor]] = []
        if name == "actor":
            for index in self._sample_indices(batch):

                def actor_output(sample_index: int = index) -> Tensor:
                    mean, log_std = self.actor.distribution_parameters(
                        batch.observations[sample_index : sample_index + 1]
                    )
                    return torch.cat((mean, log_std), dim=-1)

                closures.append(actor_output)
            return closures

        critic = getattr(self, name)
        for index in self._sample_indices(batch):

            def critic_output(
                sample_index: int = index,
                module: nn.Module = critic,
            ) -> Tensor:
                return module(
                    batch.observations[sample_index : sample_index + 1],
                    batch.actions[sample_index : sample_index + 1],
                )

            closures.append(critic_output)
        return closures

    def consolidate(self, batch: ReplayBatch) -> dict[str, float]:
        """Consolidate configured modules using a boundary-local replay batch."""

        method = self.regularizer_config.method
        target = (
            self._target_values(batch).detach()
            if method in {"ewc", "online_ewc"}
            else None
        )
        importance_means: dict[str, float] = {}
        for name, regularizer in self.regularizers.items():
            module = getattr(self, name)
            if isinstance(regularizer, SynapticIntelligence):
                regularizer.consolidate(module)
                importance = regularizer.importance
            elif isinstance(
                regularizer,
                (EWCRegularizer, OnlineEWCRegularizer),
            ):
                assert target is not None
                importance = empirical_fisher_diagonal(
                    module,
                    self._fisher_closures(name, batch, target),
                )
                regularizer.consolidate(module, importance)
            elif isinstance(regularizer, MASRegularizer):
                importance = mas_importance(
                    module,
                    self._mas_closures(name, batch),
                )
                regularizer.consolidate(module, importance)
            else:
                raise RuntimeError("unknown SAC regularizer instance")

            flattened = torch.cat(
                [value.detach().float().reshape(-1) for value in importance.values()]
            )
            importance_means[f"{name}_importance_mean"] = float(
                flattened.mean().item()
            )

        self.consolidation_count += 1
        importance_means["consolidation_count"] = float(self.consolidation_count)
        return importance_means

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["regularized_sac_version"] = 1
        state["regularizer_method"] = self.regularizer_config.method
        state["regularizers"] = {
            name: regularizer.state_dict()
            for name, regularizer in self.regularizers.items()
        }
        state["consolidation_count"] = self.consolidation_count
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("regularized_sac_version") != 1:
            raise ValueError("unsupported regularized SAC checkpoint version")
        if state.get("regularizer_method") != self.regularizer_config.method:
            raise ValueError("regularized SAC method mismatch")
        super().load_state_dict(state)
        regularizer_states = state["regularizers"]
        if set(regularizer_states) != set(self.regularizers):
            raise ValueError("regularized SAC module set mismatch")
        for name, regularizer in self.regularizers.items():
            regularizer.load_state_dict(regularizer_states[name])
        self.consolidation_count = int(state["consolidation_count"])
