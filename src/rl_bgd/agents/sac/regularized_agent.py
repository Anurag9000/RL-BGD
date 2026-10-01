"""SAC baselines with task-agnostic parameter-importance regularization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from rl_bgd.agents.sac.agent import SACAgent, SACConfig
from rl_bgd.baselines.ewc import EWCRegularizer, OnlineEWCRegularizer
from rl_bgd.baselines.importance import (
    ParameterState,
    empirical_fisher_diagonal,
    mas_importance,
    trainable_parameters,
)
from rl_bgd.baselines.mas import MASRegularizer
from rl_bgd.baselines.si import SynapticIntelligence
from rl_bgd.replay.buffer import ReplayBatch
from rl_bgd.utils.randomness import preserved_random_state

RegularizationMethod = Literal["ewc", "online_ewc", "si", "mas"]
RegularizationTarget = Literal["actor_only", "critic_only", "actor_and_critic"]
Regularizer = EWCRegularizer | OnlineEWCRegularizer | MASRegularizer | SynapticIntelligence


@dataclass(frozen=True)
class RegularizedSACConfig:
    """Parameter-importance baseline controls.

    Consolidation is update-count driven by default. Calling the explicit
    consolidation method from a true task boundary is allowed for a labelled
    oracle baseline, but strict task-agnostic runners must not do so.
    """

    method: RegularizationMethod = "ewc"
    target: RegularizationTarget = "actor_and_critic"
    strength: float = 1.0
    consolidation_interval_updates: int = 100
    importance_samples: int = 8
    online_ewc_decay: float = 0.95
    si_damping: float = 0.1

    def validate(self) -> None:
        if self.strength < 0:
            raise ValueError("regularization strength must be non-negative")
        if self.consolidation_interval_updates < 1:
            raise ValueError("consolidation interval must be positive")
        if self.importance_samples < 1:
            raise ValueError("importance_samples must be positive")
        if not 0.0 <= self.online_ewc_decay <= 1.0:
            raise ValueError("Online-EWC decay must lie in [0, 1]")
        if self.si_damping <= 0:
            raise ValueError("SI damping must be positive")


def _loss_gradient_state(
    module: nn.Module,
    loss: Tensor,
) -> ParameterState:
    parameters = trainable_parameters(module)
    names = list(parameters)
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters[name] for name in names),
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )
    return {
        name: gradient.detach().float().clone()
        for name, gradient in zip(names, gradients, strict=True)
    }


class RegularizedSACAgent(SACAgent):
    """SAC with EWC, Online-EWC, SI, or MAS actor/critic penalties."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: tuple[int, ...] = (256, 256),
        sac_config: SACConfig | None = None,
        regularization_config: RegularizedSACConfig | None = None,
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
        self.regularization_config = regularization_config or RegularizedSACConfig()
        self.regularization_config.validate()
        self.critic_pair = nn.ModuleDict(
            {
                "critic1": self.critic1,
                "critic2": self.critic2,
            }
        )
        target = self.regularization_config.target
        self.actor_regularizer = (
            self._make_regularizer(self.actor)
            if target in {"actor_only", "actor_and_critic"}
            else None
        )
        self.critic_regularizer = (
            self._make_regularizer(self.critic_pair)
            if target in {"critic_only", "actor_and_critic"}
            else None
        )
        self.consolidation_count = 0

    def _make_regularizer(self, module: nn.Module) -> Regularizer:
        config = self.regularization_config
        if config.method == "ewc":
            return EWCRegularizer(config.strength)
        if config.method == "online_ewc":
            return OnlineEWCRegularizer(
                config.strength,
                decay=config.online_ewc_decay,
            )
        if config.method == "mas":
            return MASRegularizer(config.strength)
        if config.method == "si":
            return SynapticIntelligence(
                module,
                strength=config.strength,
                damping=config.si_damping,
            )
        raise ValueError(f"unsupported regularization method: {config.method}")

    def _penalty(
        self,
        regularizer: Regularizer | None,
        module: nn.Module,
    ) -> Tensor:
        if regularizer is None:
            return torch.zeros((), device=self.device)
        return regularizer.penalty(module)

    def _importance_sample_count(self, batch: ReplayBatch) -> int:
        return min(
            self.regularization_config.importance_samples,
            int(batch.observations.shape[0]),
        )

    def _actor_importance(
        self,
        batch: ReplayBatch,
    ) -> ParameterState:
        count = self._importance_sample_count(batch)
        if self.regularization_config.method == "mas":
            closures = []
            for index in range(count):
                observation = batch.observations[index : index + 1]

                def output_closure(
                    observation: Tensor = observation,
                ) -> Tensor:
                    mean, log_std = self.actor.distribution_parameters(observation)
                    return torch.cat(
                        [mean, log_std],
                        dim=-1,
                    )

                closures.append(output_closure)
            return mas_importance(self.actor, closures)

        closures = []
        for index in range(count):
            observation = batch.observations[index : index + 1]

            def loss_closure(
                observation: Tensor = observation,
            ) -> Tensor:
                action, log_prob, _ = self.actor.sample(observation)
                q_value = torch.minimum(
                    self.critic1(observation, action),
                    self.critic2(observation, action),
                )
                return (self.alpha.detach() * log_prob - q_value).mean()

            closures.append(loss_closure)
        return empirical_fisher_diagonal(self.actor, closures)

    def _critic_importance(
        self,
        batch: ReplayBatch,
        target: Tensor,
    ) -> ParameterState:
        count = self._importance_sample_count(batch)
        if self.regularization_config.method == "mas":
            closures = []
            for index in range(count):
                observation = batch.observations[index : index + 1]
                action = batch.actions[index : index + 1]

                def output_closure(
                    observation: Tensor = observation,
                    action: Tensor = action,
                ) -> Tensor:
                    return torch.cat(
                        [
                            self.critic1(observation, action),
                            self.critic2(observation, action),
                        ],
                        dim=-1,
                    )

                closures.append(output_closure)
            return mas_importance(self.critic_pair, closures)

        closures = []
        for index in range(count):
            observation = batch.observations[index : index + 1]
            action = batch.actions[index : index + 1]
            target_value = target[index : index + 1]

            def loss_closure(
                observation: Tensor = observation,
                action: Tensor = action,
                target_value: Tensor = target_value,
            ) -> Tensor:
                q1 = self.critic1(observation, action)
                q2 = self.critic2(observation, action)
                return 0.5 * (
                    torch.nn.functional.mse_loss(q1, target_value)
                    + torch.nn.functional.mse_loss(q2, target_value)
                )

            closures.append(loss_closure)
        return empirical_fisher_diagonal(self.critic_pair, closures)

    def consolidate_from_batch(
        self,
        batch: ReplayBatch,
        *,
        critic_target: Tensor | None = None,
    ) -> None:
        """Consolidate using learner-visible replay data only."""

        method = self.regularization_config.method
        if method == "si":
            if isinstance(self.actor_regularizer, SynapticIntelligence):
                self.actor_regularizer.consolidate(self.actor)
            if isinstance(self.critic_regularizer, SynapticIntelligence):
                self.critic_regularizer.consolidate(self.critic_pair)
            self.consolidation_count += 1
            return

        with preserved_random_state():
            torch.manual_seed(100_000 + self.update_count)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(100_000 + self.update_count)

            if self.actor_regularizer is not None:
                actor_importance = self._actor_importance(batch)
                self.actor_regularizer.consolidate(
                    self.actor,
                    actor_importance,
                )

            if self.critic_regularizer is not None:
                target = (
                    critic_target.detach()
                    if critic_target is not None
                    else self._target_values(batch).detach()
                )
                critic_importance = self._critic_importance(batch, target)
                self.critic_regularizer.consolidate(
                    self.critic_pair,
                    critic_importance,
                )
        self.consolidation_count += 1

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        target = self._target_values(batch)
        q1 = self.critic1(batch.observations, batch.actions)
        q2 = self.critic2(batch.observations, batch.actions)
        critic_base_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(
            q2, target
        )
        critic_penalty = self._penalty(
            self.critic_regularizer,
            self.critic_pair,
        )
        critic_total_loss = critic_base_loss + critic_penalty

        critic_si_gradients: ParameterState | None = None
        if isinstance(self.critic_regularizer, SynapticIntelligence):
            critic_si_gradients = _loss_gradient_state(
                self.critic_pair,
                critic_base_loss,
            )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_total_loss.backward()
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self._clip_gradients(critic_parameters)
        self.critic_optimizer.step()
        if (
            isinstance(self.critic_regularizer, SynapticIntelligence)
            and critic_si_gradients is not None
        ):
            self.critic_regularizer.accumulate_after_step(
                self.critic_pair,
                critic_si_gradients,
            )

        sampled_action, log_prob, _ = self.actor.sample(batch.observations)
        q_pi = torch.minimum(
            self.critic1(batch.observations, sampled_action),
            self.critic2(batch.observations, sampled_action),
        )
        actor_base_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        actor_penalty = self._penalty(
            self.actor_regularizer,
            self.actor,
        )
        actor_total_loss = actor_base_loss + actor_penalty

        actor_si_gradients: ParameterState | None = None
        if isinstance(self.actor_regularizer, SynapticIntelligence):
            actor_si_gradients = _loss_gradient_state(
                self.actor,
                actor_base_loss,
            )

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_total_loss.backward()
        self._clip_gradients(list(self.actor.parameters()))
        self.actor_optimizer.step()
        if (
            isinstance(self.actor_regularizer, SynapticIntelligence)
            and actor_si_gradients is not None
        ):
            self.actor_regularizer.accumulate_after_step(
                self.actor,
                actor_si_gradients,
            )

        alpha_loss = torch.zeros((), device=self.device)
        if self.config.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._polyak_update()
        self.update_count += 1
        if self.update_count % self.regularization_config.consolidation_interval_updates == 0:
            self.consolidate_from_batch(
                batch,
                critic_target=target,
            )

        metrics = {
            "critic_loss": float(critic_base_loss.detach().item()),
            "critic_regularization_penalty": float(critic_penalty.detach().item()),
            "critic_total_loss": float(critic_total_loss.detach().item()),
            "actor_loss": float(actor_base_loss.detach().item()),
            "actor_regularization_penalty": float(actor_penalty.detach().item()),
            "actor_total_loss": float(actor_total_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
            "target_q_mean": float(target.detach().mean().item()),
            "policy_entropy_estimate": float((-log_prob.detach()).mean().item()),
            "consolidation_count": float(self.consolidation_count),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("nonfinite regularized SAC update metric")
        return metrics

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["regularized_sac"] = {
            "version": 1,
            "config": asdict(self.regularization_config),
            "consolidation_count": self.consolidation_count,
            "actor_regularizer": (
                None if self.actor_regularizer is None else self.actor_regularizer.state_dict()
            ),
            "critic_regularizer": (
                None if self.critic_regularizer is None else self.critic_regularizer.state_dict()
            ),
        }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        regularized = state.get("regularized_sac")
        if not isinstance(regularized, dict) or regularized.get("version") != 1:
            raise ValueError("missing or unsupported regularized SAC state")
        if regularized.get("config") != asdict(self.regularization_config):
            raise ValueError("regularized SAC checkpoint configuration mismatch")
        self.consolidation_count = int(regularized["consolidation_count"])
        for key, regularizer in (
            ("actor_regularizer", self.actor_regularizer),
            ("critic_regularizer", self.critic_regularizer),
        ):
            saved = regularized[key]
            if regularizer is None:
                if saved is not None:
                    raise ValueError(f"unexpected checkpoint state for {key}")
            else:
                if not isinstance(saved, dict):
                    raise TypeError(f"checkpoint {key} must be a mapping")
                regularizer.load_state_dict(saved)
