"""Soft Actor-Critic variants whose actor and/or critics use BGD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor
from torch.func import functional_call

from rl_bgd.agents.sac.agent import SACAgent, SACConfig
from rl_bgd.bayes.bgd import BGDConfig, BGDLoss, BGDStepResult, BGDUpdater
from rl_bgd.bayes.diagonal_gaussian import (
    DiagonalGaussianPosterior,
    PosteriorBounds,
)
from rl_bgd.replay.buffer import ReplayBatch
from rl_bgd.replay.evidence_accounting import (
    ReplayEvidenceConfig,
    ReplayEvidenceSummary,
    replay_evidence_weights,
    weighted_evidence_mean,
)
from rl_bgd.surprise.base import SurpriseObservation, surprise_to_retention
from rl_bgd.surprise.ensemble import (
    AdaptiveEnsembleRetentionConfig,
    EnsembleDisagreementSurprise,
)
from rl_bgd.surprise.predictive import (
    AdaptivePredictiveRetentionConfig,
    GaussianTransitionModel,
    PredictiveSurprise,
)
from rl_bgd.surprise.td import AdaptiveTDRetentionConfig, TDSurprise

BayesianizationMode = Literal[
    "critic_only",
    "actor_only",
    "actor_and_critic",
]


@dataclass(frozen=True)
class BGDSACConfig:
    bayesianization: BayesianizationMode = "critic_only"
    posterior_std: float = 0.1
    sigma_min: float = 1e-6
    sigma_max: float = 10.0
    replay_evidence: ReplayEvidenceConfig = field(default_factory=ReplayEvidenceConfig)
    adaptive_td_retention: AdaptiveTDRetentionConfig | None = None
    adaptive_ensemble_retention: AdaptiveEnsembleRetentionConfig | None = None
    adaptive_predictive_retention: AdaptivePredictiveRetentionConfig | None = None
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
            raise ValueError(f"unsupported Bayesianization mode: {self.bayesianization}")
        if self.posterior_std <= 0:
            raise ValueError("posterior_std must be positive")
        PosteriorBounds(
            self.sigma_min,
            self.sigma_max,
        ).validate()
        self.replay_evidence.validate()
        adaptive_sources = sum(
            source is not None
            for source in (
                self.adaptive_td_retention,
                self.adaptive_ensemble_retention,
                self.adaptive_predictive_retention,
            )
        )
        if adaptive_sources > 1:
            raise ValueError("configure at most one adaptive retention surprise source")
        if self.adaptive_td_retention is not None:
            self.adaptive_td_retention.validate()
        if self.adaptive_ensemble_retention is not None:
            self.adaptive_ensemble_retention.validate()
        if self.adaptive_predictive_retention is not None:
            self.adaptive_predictive_retention.validate()
        self.actor_bgd.validate()
        self.critic_bgd.validate()


class BGDSACAgent(SACAgent):
    """SAC with selectable BGD Bayesianization of actor and/or twin critics."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: tuple[int, ...] = (256, 256),
        sac_config: SACConfig | None = None,
        bgd_config: BGDSACConfig | None = None,
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
        self.bgd_config = bgd_config or BGDSACConfig()
        self.bgd_config.validate()
        bounds = PosteriorBounds(
            sigma_min=self.bgd_config.sigma_min,
            sigma_max=self.bgd_config.sigma_max,
        )
        mode = self.bgd_config.bayesianization
        self.actor_posterior: DiagonalGaussianPosterior | None = None
        self.actor_bgd: BGDUpdater | None = None
        self.critic1_posterior: DiagonalGaussianPosterior | None = None
        self.critic2_posterior: DiagonalGaussianPosterior | None = None
        self.critic1_bgd: BGDUpdater | None = None
        self.critic2_bgd: BGDUpdater | None = None
        self.td_surprise: TDSurprise | None = None
        self.ensemble_surprise: EnsembleDisagreementSurprise | None = None
        self.predictive_surprise: PredictiveSurprise | None = None
        self.predictive_model: GaussianTransitionModel | None = None
        self.predictive_optimizer: torch.optim.Optimizer | None = None

        if self.bgd_config.adaptive_td_retention is not None:
            self.td_surprise = TDSurprise(self.bgd_config.adaptive_td_retention.surprise)
        if self.bgd_config.adaptive_ensemble_retention is not None:
            self.ensemble_surprise = EnsembleDisagreementSurprise(
                self.bgd_config.adaptive_ensemble_retention.normalizer
            )
        predictive_config = self.bgd_config.adaptive_predictive_retention
        if predictive_config is not None:
            self.predictive_surprise = PredictiveSurprise(
                predictive_config.normalizer
            )
            self.predictive_model = GaussianTransitionModel(
                observation_dim,
                action_dim,
                hidden_dims=predictive_config.hidden_dims,
                min_log_std=predictive_config.min_log_std,
                max_log_std=predictive_config.max_log_std,
            ).to(self.device)
            self.predictive_optimizer = torch.optim.Adam(
                self.predictive_model.parameters(),
                lr=predictive_config.learning_rate,
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

    def _evidence(self, batch: ReplayBatch) -> ReplayEvidenceSummary:
        return replay_evidence_weights(batch, self.bgd_config.replay_evidence)

    def _adaptive_retention(
        self,
        batch: ReplayBatch,
        target: Tensor,
    ) -> tuple[float | None, SurpriseObservation | None]:
        td_config = self.bgd_config.adaptive_td_retention
        if td_config is not None and self.td_surprise is not None:
            with torch.no_grad():
                q1 = self.critic1(batch.observations, batch.actions)
                q2 = self.critic2(batch.observations, batch.actions)
                td_errors = torch.cat((target - q1, target - q2), dim=0)
            observation = self.td_surprise.observe(td_errors)
            retention = surprise_to_retention(
                observation.smoothed,
                td_config.mapping,
            )
            return retention, observation

        ensemble_config = self.bgd_config.adaptive_ensemble_retention
        if ensemble_config is not None and self.ensemble_surprise is not None:
            with torch.no_grad():
                predictions = torch.stack(
                    (
                        self.critic1(
                            batch.observations,
                            batch.actions,
                        ),
                        self.critic2(
                            batch.observations,
                            batch.actions,
                        ),
                    ),
                    dim=0,
                )
            observation = self.ensemble_surprise.observe(predictions)
            retention = surprise_to_retention(
                observation.smoothed,
                ensemble_config.mapping,
            )
            return retention, observation

        predictive_config = self.bgd_config.adaptive_predictive_retention
        if (
            predictive_config is not None
            and self.predictive_surprise is not None
            and self.predictive_model is not None
        ):
            with torch.no_grad():
                nll = self.predictive_model.negative_log_likelihood(
                    batch.observations,
                    batch.actions,
                    batch.next_observations,
                    batch.rewards,
                )
            observation = self.predictive_surprise.observe_nll(nll)
            retention = surprise_to_retention(
                observation.smoothed,
                predictive_config.mapping,
            )
            return retention, observation

        return None, None

    def _update_predictive_model(
        self,
        batch: ReplayBatch,
    ) -> float | None:
        config = self.bgd_config.adaptive_predictive_retention
        if (
            config is None
            or self.predictive_model is None
            or self.predictive_optimizer is None
        ):
            return None
        nll = self.predictive_model.negative_log_likelihood(
            batch.observations,
            batch.actions,
            batch.next_observations,
            batch.rewards,
        )
        loss = nll.mean()
        self.predictive_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.predictive_model.parameters(),
                config.gradient_clip_norm,
            )
        self.predictive_optimizer.step()
        return float(loss.detach().item())

    def _update_critics_bgd(
        self,
        batch: ReplayBatch,
        target: Tensor,
        evidence: ReplayEvidenceSummary,
        retention: float | None,
    ) -> tuple[BGDStepResult, BGDStepResult]:
        if self.critic1_bgd is None or self.critic2_bgd is None:
            raise RuntimeError("critic BGD is not configured")

        def critic_loss(output: Tensor) -> Tensor:
            return torch.nn.functional.mse_loss(output, target)

        def critic_uncertainty_loss(output: Tensor) -> Tensor:
            per_item = torch.nn.functional.mse_loss(
                output,
                target,
                reduction="none",
            )
            return weighted_evidence_mean(per_item, evidence.weights)

        result1 = self.critic1_bgd.step_module(
            self.critic1,
            critic_loss,
            batch.observations,
            batch.actions,
            uncertainty_loss_fn=critic_uncertainty_loss,
            retention=retention,
        )
        result2 = self.critic2_bgd.step_module(
            self.critic2,
            critic_loss,
            batch.observations,
            batch.actions,
            uncertainty_loss_fn=critic_uncertainty_loss,
            retention=retention,
        )
        return result1, result2

    def _update_actor_bgd(
        self,
        batch: ReplayBatch,
        evidence: ReplayEvidenceSummary,
        retention: float | None,
    ) -> BGDStepResult:
        if self.actor_bgd is None or self.actor_posterior is None:
            raise RuntimeError("actor BGD is not configured")
        actor_buffers = dict(self.actor.named_buffers())

        def objective(params: dict[str, Tensor]) -> BGDLoss:
            sampled_action, log_prob, _ = functional_call(
                self.actor,
                (dict(params), actor_buffers),
                (batch.observations,),
            )
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
            per_item = self.alpha.detach() * log_prob - q_pi
            return BGDLoss(
                mean=per_item.mean(),
                uncertainty=weighted_evidence_mean(per_item, evidence.weights),
            )

        result = self.actor_bgd.step(
            objective,
            retention=retention,
        )
        self.actor_posterior.sync_module(self.actor)
        return result

    def update(
        self,
        batch: ReplayBatch,
    ) -> dict[str, float]:
        mode = self.bgd_config.bayesianization
        target = self._target_values(batch)
        evidence = self._evidence(batch)
        retention, surprise = self._adaptive_retention(batch, target)
        predictive_model_loss = self._update_predictive_model(batch)

        critic1_result: BGDStepResult | None = None
        critic2_result: BGDStepResult | None = None
        if mode in {"critic_only", "actor_and_critic"}:
            critic1_result, critic2_result = self._update_critics_bgd(
                batch,
                target,
                evidence,
                retention,
            )
            q1 = self.critic1(
                batch.observations,
                batch.actions,
            )
            q2 = self.critic2(
                batch.observations,
                batch.actions,
            )
            critic_loss_value = (
                torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
            ).detach()
        else:
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
            critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
            self._clip_gradients(critic_params)
            self.critic_optimizer.step()
            critic_loss_value = critic_loss.detach()

        actor_result: BGDStepResult | None = None
        if mode in {"actor_only", "actor_and_critic"}:
            actor_result = self._update_actor_bgd(
                batch,
                evidence,
                retention,
            )
            with torch.no_grad():
                (
                    sampled_action,
                    log_prob,
                    _,
                ) = self.actor.sample(batch.observations)
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
                actor_loss_value = (self.alpha.detach() * log_prob - q_pi).mean()
        else:
            (
                sampled_action,
                log_prob,
                _,
            ) = self.actor.sample(batch.observations)
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
            actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip_gradients(list(self.actor.parameters()))
            self.actor_optimizer.step()
            actor_loss_value = actor_loss.detach()

        if mode in {"actor_only", "actor_and_critic"}:
            _, alpha_log_prob, _ = self.actor.sample(batch.observations)
        else:
            alpha_log_prob = log_prob

        alpha_loss = torch.zeros((), device=self.device)
        if self.config.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (alpha_log_prob.detach() + self.target_entropy)).mean()
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
            "policy_entropy_estimate": float((-alpha_log_prob.detach()).mean().item()),
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
        if predictive_model_loss is not None:
            metrics["predictive_model_loss"] = predictive_model_loss
        if critic1_result is not None and critic2_result is not None:
            metrics.update(
                {
                    "critic1_sigma_mean": (critic1_result.diagnostics["sigma_mean"]),
                    "critic2_sigma_mean": (critic2_result.diagnostics["sigma_mean"]),
                    "critic1_effective_lr_mean": (critic1_result.diagnostics["effective_lr_mean"]),
                    "critic2_effective_lr_mean": (critic2_result.diagnostics["effective_lr_mean"]),
                    "critic1_uncertainty_gradient_norm": (critic1_result.uncertainty_gradient_norm),
                    "critic2_uncertainty_gradient_norm": (critic2_result.uncertainty_gradient_norm),
                }
            )
        if actor_result is not None:
            metrics.update(
                {
                    "actor_sigma_mean": (actor_result.diagnostics["sigma_mean"]),
                    "actor_effective_lr_mean": (actor_result.diagnostics["effective_lr_mean"]),
                    "actor_uncertainty_gradient_norm": (actor_result.uncertainty_gradient_norm),
                }
            )
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("nonfinite BGD-SAC update metric")
        return metrics

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["bgd_sac_version"] = 1
        state["bayesianization"] = self.bgd_config.bayesianization
        state["replay_evidence_mode"] = self.bgd_config.replay_evidence.mode
        state["adaptive_td_retention"] = self.bgd_config.adaptive_td_retention is not None
        state["adaptive_ensemble_retention"] = (
            self.bgd_config.adaptive_ensemble_retention is not None
        )
        state["adaptive_predictive_retention"] = (
            self.bgd_config.adaptive_predictive_retention is not None
        )
        if self.td_surprise is not None:
            state["td_surprise"] = self.td_surprise.state_dict()
        if self.ensemble_surprise is not None:
            state["ensemble_surprise"] = self.ensemble_surprise.state_dict()
        if (
            self.predictive_surprise is not None
            and self.predictive_model is not None
            and self.predictive_optimizer is not None
        ):
            state["predictive_surprise"] = self.predictive_surprise.state_dict()
            state["predictive_model"] = self.predictive_model.state_dict()
            state["predictive_optimizer"] = self.predictive_optimizer.state_dict()
        if self.actor_bgd is not None:
            state["actor_bgd"] = self.actor_bgd.state_dict()
        if self.critic1_bgd is not None and self.critic2_bgd is not None:
            state["critic1_bgd"] = self.critic1_bgd.state_dict()
            state["critic2_bgd"] = self.critic2_bgd.state_dict()
        return state

    def load_state_dict(
        self,
        state: dict[str, object],
    ) -> None:
        if state.get("bgd_sac_version") != 1:
            raise ValueError("unsupported BGD-SAC checkpoint version")
        if state.get("bayesianization") != self.bgd_config.bayesianization:
            raise ValueError("BGD-SAC checkpoint Bayesianization mode mismatch")
        if state.get("replay_evidence_mode", "all_replay") != self.bgd_config.replay_evidence.mode:
            raise ValueError("BGD-SAC checkpoint replay evidence mode mismatch")
        expected_td_adaptive = self.bgd_config.adaptive_td_retention is not None
        if bool(state.get("adaptive_td_retention", False)) != expected_td_adaptive:
            raise ValueError("BGD-SAC checkpoint TD adaptive-retention configuration mismatch")
        expected_ensemble_adaptive = self.bgd_config.adaptive_ensemble_retention is not None
        if bool(state.get("adaptive_ensemble_retention", False)) != expected_ensemble_adaptive:
            raise ValueError(
                "BGD-SAC checkpoint ensemble adaptive-retention configuration mismatch"
            )
        expected_predictive_adaptive = self.bgd_config.adaptive_predictive_retention is not None
        if bool(state.get("adaptive_predictive_retention", False)) != expected_predictive_adaptive:
            raise ValueError(
                "BGD-SAC checkpoint predictive adaptive-retention configuration mismatch"
            )
        super().load_state_dict(state)  # type: ignore[arg-type]
        if self.td_surprise is not None:
            payload = state["td_surprise"]
            if not isinstance(payload, dict):
                raise TypeError("TD-surprise checkpoint state must be a dictionary")
            self.td_surprise.load_state_dict(payload)
        if self.ensemble_surprise is not None:
            payload = state["ensemble_surprise"]
            if not isinstance(payload, dict):
                raise TypeError("ensemble-surprise checkpoint state must be a dictionary")
            self.ensemble_surprise.load_state_dict(payload)
        if (
            self.predictive_surprise is not None
            and self.predictive_model is not None
            and self.predictive_optimizer is not None
        ):
            payload = state["predictive_surprise"]
            if not isinstance(payload, dict):
                raise TypeError("predictive-surprise checkpoint state must be a dictionary")
            self.predictive_surprise.load_state_dict(payload)
            model_state = state["predictive_model"]
            optimizer_state = state["predictive_optimizer"]
            if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
                raise TypeError("predictive-model checkpoint state must be dictionaries")
            self.predictive_model.load_state_dict(model_state)
            self.predictive_optimizer.load_state_dict(optimizer_state)
        if self.actor_bgd is not None:
            self.actor_bgd.load_state_dict(
                state["actor_bgd"]  # type: ignore[arg-type]
            )
            assert self.actor_posterior is not None
            self.actor_posterior.sync_module(self.actor)
        if self.critic1_bgd is not None and self.critic2_bgd is not None:
            self.critic1_bgd.load_state_dict(
                state["critic1_bgd"]  # type: ignore[arg-type]
            )
            self.critic2_bgd.load_state_dict(
                state["critic2_bgd"]  # type: ignore[arg-type]
            )
            assert self.critic1_posterior is not None
            assert self.critic2_posterior is not None
            self.critic1_posterior.sync_module(self.critic1)
            self.critic2_posterior.sync_module(self.critic2)
