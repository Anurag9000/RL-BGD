"""Bayesian Gradient Descent variants of recurrent SAC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.func import functional_call

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.recurrent_agent import (
    RecurrentSACAgent,
    RecurrentSACConfig,
)
from rl_bgd.bayes.bgd import (
    BGDConfig,
    BGDLoss,
    BGDStepResult,
    BGDUpdater,
)
from rl_bgd.bayes.diagonal_gaussian import (
    DiagonalGaussianPosterior,
    PosteriorBounds,
)
from rl_bgd.replay.evidence_accounting import ReplayEvidenceConfig
from rl_bgd.replay.sequence_buffer import SequenceReplayBatch
from rl_bgd.replay.sequence_evidence import (
    SequenceReplayEvidenceSummary,
    sequence_replay_evidence_weights,
    weighted_sequence_evidence_mean,
)
from rl_bgd.surprise.base import SurpriseObservation, surprise_to_retention
from rl_bgd.surprise.td import AdaptiveTDRetentionConfig, TDSurprise

RecurrentBayesianizationMode = Literal[
    "critic_only",
    "actor_only",
    "actor_and_critic",
]


@dataclass(frozen=True)
class RecurrentBGDSACConfig:
    """Posterior, evidence, and adaptive-retention controls."""

    bayesianization: RecurrentBayesianizationMode = "actor_and_critic"
    posterior_std: float = 0.1
    sigma_min: float = 1e-6
    sigma_max: float = 10.0
    replay_evidence: ReplayEvidenceConfig = field(
        default_factory=ReplayEvidenceConfig
    )
    adaptive_td_retention: AdaptiveTDRetentionConfig | None = None
    actor_bgd: BGDConfig = field(
        default_factory=lambda: BGDConfig(
            eta=0.1,
            mc_samples=4,
            antithetic=True,
        )
    )
    critic_bgd: BGDConfig = field(
        default_factory=lambda: BGDConfig(
            eta=0.1,
            mc_samples=4,
            antithetic=True,
        )
    )

    def validate(self) -> None:
        if self.bayesianization not in {
            "critic_only",
            "actor_only",
            "actor_and_critic",
        }:
            raise ValueError(
                f"unsupported recurrent Bayesianization mode: {self.bayesianization}"
            )
        if self.posterior_std <= 0:
            raise ValueError("posterior_std must be positive")
        PosteriorBounds(
            self.sigma_min,
            self.sigma_max,
        ).validate()
        self.replay_evidence.validate()
        if self.adaptive_td_retention is not None:
            self.adaptive_td_retention.validate()
        self.actor_bgd.validate()
        self.critic_bgd.validate()


class RecurrentBGDSACAgent(RecurrentSACAgent):
    """Recurrent SAC with BGD on the actor, critics, or both."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        sac_config: SACConfig | None = None,
        recurrent_config: RecurrentSACConfig | None = None,
        bgd_config: RecurrentBGDSACConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            action_low=action_low,
            action_high=action_high,
            sac_config=sac_config,
            recurrent_config=recurrent_config,
            device=device,
        )
        self.bgd_config = bgd_config or RecurrentBGDSACConfig()
        self.bgd_config.validate()
        bounds = PosteriorBounds(
            sigma_min=self.bgd_config.sigma_min,
            sigma_max=self.bgd_config.sigma_max,
        )
        mode = self.bgd_config.bayesianization

        self.actor_posterior: DiagonalGaussianPosterior | None = None
        self.critic1_posterior: DiagonalGaussianPosterior | None = None
        self.critic2_posterior: DiagonalGaussianPosterior | None = None
        self.actor_bgd: BGDUpdater | None = None
        self.critic1_bgd: BGDUpdater | None = None
        self.critic2_bgd: BGDUpdater | None = None
        self.td_surprise: TDSurprise | None = None

        if self.bgd_config.adaptive_td_retention is not None:
            self.td_surprise = TDSurprise(
                self.bgd_config.adaptive_td_retention.surprise
            )

        if mode in {"actor_only", "actor_and_critic"}:
            self.actor_posterior = DiagonalGaussianPosterior.from_module(
                self.actor,
                prior_std=self.bgd_config.posterior_std,
                bounds=bounds,
            )
            self.actor_bgd = BGDUpdater(
                self.actor_posterior,
                self.bgd_config.actor_bgd,
            )

        if mode in {"critic_only", "actor_and_critic"}:
            self.critic1_posterior = DiagonalGaussianPosterior.from_module(
                self.critic1,
                prior_std=self.bgd_config.posterior_std,
                bounds=bounds,
            )
            self.critic2_posterior = DiagonalGaussianPosterior.from_module(
                self.critic2,
                prior_std=self.bgd_config.posterior_std,
                bounds=bounds,
            )
            self.critic1_bgd = BGDUpdater(
                self.critic1_posterior,
                self.bgd_config.critic_bgd,
            )
            self.critic2_bgd = BGDUpdater(
                self.critic2_posterior,
                self.bgd_config.critic_bgd,
            )

    def _evidence(
        self,
        batch: SequenceReplayBatch,
    ) -> SequenceReplayEvidenceSummary:
        return sequence_replay_evidence_weights(
            batch,
            self.bgd_config.replay_evidence,
        )

    def _adaptive_retention(
        self,
        batch: SequenceReplayBatch,
        target: Tensor,
    ) -> tuple[float | None, SurpriseObservation | None]:
        config = self.bgd_config.adaptive_td_retention
        if config is None or self.td_surprise is None:
            return None, None
        with torch.no_grad():
            q1, q2 = self._critic_predictions(batch)
            td_errors = torch.cat(
                [
                    target - q1,
                    target - q2,
                ],
                dim=0,
            )
        observation = self.td_surprise.observe(td_errors)
        return (
            surprise_to_retention(
                observation.smoothed,
                config.mapping,
            ),
            observation,
        )

    def _update_critic_bgd(
        self,
        module: nn.Module,
        updater: BGDUpdater,
        batch: SequenceReplayBatch,
        target: Tensor,
        evidence: SequenceReplayEvidenceSummary,
        *,
        generator: torch.Generator,
        retention: float | None,
    ) -> BGDStepResult:
        selector = batch.unroll_slice

        def mean_loss(output: Tensor) -> Tensor:
            return torch.nn.functional.mse_loss(
                output[:, selector],
                target,
            )

        def uncertainty_loss(output: Tensor) -> Tensor:
            per_transition = torch.nn.functional.mse_loss(
                output[:, selector],
                target,
                reduction="none",
            )
            return weighted_sequence_evidence_mean(
                per_transition,
                evidence.weights,
            )

        return updater.step_module(
            module,
            mean_loss,
            batch.observations,
            batch.actions,
            batch.episode_starts,
            uncertainty_loss_fn=uncertainty_loss,
            generator=generator,
            retention=retention,
        )

    def _sampled_actor_loss(
        self,
        params: dict[str, Tensor],
        batch: SequenceReplayBatch,
        evidence: SequenceReplayEvidenceSummary,
    ) -> BGDLoss:
        actor_buffers = dict(self.actor.named_buffers())
        (
            sampled_actions,
            log_prob,
            _,
            _,
            _,
        ) = functional_call(
            self.actor,
            (params, actor_buffers),
            (
                batch.observations,
                batch.episode_starts,
            ),
        )
        with torch.no_grad():
            critic1_hidden, _ = self.critic1.hidden_sequence(
                batch.observations,
                batch.episode_starts,
            )
            critic2_hidden, _ = self.critic2.hidden_sequence(
                batch.observations,
                batch.episode_starts,
            )
        q1_pi = self.critic1.q_from_hidden(
            critic1_hidden,
            sampled_actions,
        )
        q2_pi = self.critic2.q_from_hidden(
            critic2_hidden,
            sampled_actions,
        )
        selector = batch.unroll_slice
        per_transition = (
            self.alpha.detach()
            * log_prob[:, selector]
            - torch.minimum(
                q1_pi[:, selector],
                q2_pi[:, selector],
            )
        )
        return BGDLoss(
            mean=per_transition.mean(),
            uncertainty=weighted_sequence_evidence_mean(
                per_transition,
                evidence.weights,
            ),
        )

    def _update_actor_bgd(
        self,
        batch: SequenceReplayBatch,
        evidence: SequenceReplayEvidenceSummary,
        *,
        generator: torch.Generator,
        retention: float | None,
    ) -> BGDStepResult:
        if self.actor_bgd is None or self.actor_posterior is None:
            raise RuntimeError("recurrent actor BGD is not configured")

        def objective(params: dict[str, Tensor]) -> BGDLoss:
            return self._sampled_actor_loss(
                params,
                batch,
                evidence,
            )

        result = self.actor_bgd.step(
            objective,
            generator=generator,
            retention=retention,
        )
        self.actor_posterior.sync_module(self.actor)
        return result

    def update(
        self,
        batch: SequenceReplayBatch,
    ) -> dict[str, float]:
        mode = self.bgd_config.bayesianization
        target = self._target_values(batch)
        evidence = self._evidence(batch)
        retention, surprise = self._adaptive_retention(
            batch,
            target,
        )
        generator = torch.Generator(
            device=self.device
        ).manual_seed(
            91_000 + self.update_count
        )

        critic1_result: BGDStepResult | None = None
        critic2_result: BGDStepResult | None = None
        if mode in {"critic_only", "actor_and_critic"}:
            if self.critic1_bgd is None or self.critic2_bgd is None:
                raise RuntimeError("recurrent critic BGD is not configured")
            critic1_result = self._update_critic_bgd(
                self.critic1,
                self.critic1_bgd,
                batch,
                target,
                evidence,
                generator=generator,
                retention=retention,
            )
            critic2_result = self._update_critic_bgd(
                self.critic2,
                self.critic2_bgd,
                batch,
                target,
                evidence,
                generator=generator,
                retention=retention,
            )
            q1, q2 = self._critic_predictions(batch)
            critic_loss_value = (
                torch.nn.functional.mse_loss(q1, target)
                + torch.nn.functional.mse_loss(q2, target)
            ).detach()
        else:
            q1, q2 = self._critic_predictions(batch)
            critic_loss = (
                torch.nn.functional.mse_loss(q1, target)
                + torch.nn.functional.mse_loss(q2, target)
            )
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_parameters = (
                list(self.critic1.parameters())
                + list(self.critic2.parameters())
            )
            self._clip_gradients(critic_parameters)
            self.critic_optimizer.step()
            critic_loss_value = critic_loss.detach()

        actor_result: BGDStepResult | None = None
        if mode in {"actor_only", "actor_and_critic"}:
            actor_result = self._update_actor_bgd(
                batch,
                evidence,
                generator=generator,
                retention=retention,
            )
            with torch.no_grad():
                actor_terms, log_prob, _ = self._actor_terms(batch)
                actor_loss_value = actor_terms.mean()
        else:
            self._set_critics_trainable(False)
            actor_terms, log_prob, _ = self._actor_terms(batch)
            actor_loss = actor_terms.mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip_gradients(list(self.actor.parameters()))
            self.actor_optimizer.step()
            self._set_critics_trainable(True)
            actor_loss_value = actor_loss.detach()

        if mode in {"actor_only", "actor_and_critic"}:
            with torch.no_grad():
                _, alpha_log_prob, _ = self._actor_terms(batch)
        else:
            alpha_log_prob = log_prob

        alpha_loss = torch.zeros((), device=self.device)
        if self.config.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha
                * (
                    alpha_log_prob.detach()
                    + self.target_entropy
                )
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._polyak_update()
        self.update_count += 1
        metrics: dict[str, float] = {
            "critic_loss": float(critic_loss_value.item()),
            "actor_loss": float(actor_loss_value.item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
            "target_q_mean": float(target.detach().mean().item()),
            "policy_entropy_estimate": float(
                (-alpha_log_prob.detach()).mean().item()
            ),
            "evidence_weight_mean": evidence.mean_weight,
            "evidence_weight_min": evidence.min_weight,
            "evidence_weight_max": evidence.max_weight,
            "evidence_fresh_fraction": evidence.fresh_fraction,
            "evidence_mean_usage_count": evidence.mean_usage_count,
            "evidence_effective_sample_size": evidence.effective_sample_size,
        }
        if surprise is not None and retention is not None:
            metrics.update(
                {
                    "surprise_raw": surprise.raw,
                    "surprise_center": surprise.center,
                    "surprise_scale": surprise.scale,
                    "surprise_normalized": surprise.normalized,
                    "surprise_smoothed": surprise.smoothed,
                    "retention_lambda": retention,
                }
            )
        if actor_result is not None:
            metrics.update(
                {
                    "actor_sigma_mean": actor_result.diagnostics["sigma_mean"],
                    "actor_effective_lr_mean": actor_result.diagnostics[
                        "effective_lr_mean"
                    ],
                    "actor_gradient_norm": actor_result.gradient_norm,
                    "actor_c_norm": actor_result.c_norm,
                }
            )
        if critic1_result is not None and critic2_result is not None:
            metrics.update(
                {
                    "critic1_sigma_mean": critic1_result.diagnostics["sigma_mean"],
                    "critic2_sigma_mean": critic2_result.diagnostics["sigma_mean"],
                    "critic1_effective_lr_mean": critic1_result.diagnostics[
                        "effective_lr_mean"
                    ],
                    "critic2_effective_lr_mean": critic2_result.diagnostics[
                        "effective_lr_mean"
                    ],
                    "critic1_gradient_norm": critic1_result.gradient_norm,
                    "critic2_gradient_norm": critic2_result.gradient_norm,
                }
            )
        if not all(
            torch.isfinite(torch.tensor(value))
            for value in metrics.values()
        ):
            raise FloatingPointError("nonfinite recurrent BGD-SAC metric")
        return metrics

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["recurrent_bgd_sac_version"] = 1
        state["bayesianization"] = self.bgd_config.bayesianization
        if self.actor_bgd is not None:
            state["actor_bgd"] = self.actor_bgd.state_dict()
        if self.critic1_bgd is not None and self.critic2_bgd is not None:
            state["critic1_bgd"] = self.critic1_bgd.state_dict()
            state["critic2_bgd"] = self.critic2_bgd.state_dict()
        if self.td_surprise is not None:
            state["td_surprise"] = self.td_surprise.state_dict()
        return state

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if state.get("recurrent_bgd_sac_version") != 1:
            raise ValueError(
                "unsupported recurrent BGD-SAC checkpoint version"
            )
        if state.get("bayesianization") != self.bgd_config.bayesianization:
            raise ValueError(
                "recurrent BGD-SAC checkpoint Bayesianization mode mismatch"
            )
        super().load_state_dict(state)
        if self.actor_bgd is not None:
            self.actor_bgd.load_state_dict(state["actor_bgd"])
            assert self.actor_posterior is not None
            self.actor_posterior.sync_module(self.actor)
        if self.critic1_bgd is not None and self.critic2_bgd is not None:
            self.critic1_bgd.load_state_dict(state["critic1_bgd"])
            self.critic2_bgd.load_state_dict(state["critic2_bgd"])
            assert self.critic1_posterior is not None
            assert self.critic2_posterior is not None
            self.critic1_posterior.sync_module(self.critic1)
            self.critic2_posterior.sync_module(self.critic2)
        if self.td_surprise is not None and "td_surprise" in state:
            self.td_surprise.load_state_dict(state["td_surprise"])
