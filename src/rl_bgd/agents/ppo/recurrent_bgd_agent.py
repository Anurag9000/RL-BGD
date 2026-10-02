"""BGD variant of recurrent PPO with matched hidden-state architecture."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.bgd_agent import BGDPPOConfig
from rl_bgd.agents.ppo.recurrent_agent import (
    RecurrentPPOAgent,
    RecurrentPPOConfig,
)
from rl_bgd.agents.ppo.recurrent_rollout import (
    RecurrentPPORolloutBatch,
    RecurrentRolloutBuffer,
)
from rl_bgd.bayes.bgd import (
    BGDLoss,
    BGDStepResult,
    BGDUpdater,
)
from rl_bgd.bayes.diagonal_gaussian import (
    DiagonalGaussianPosterior,
)


class BGDRecurrentPPOAgent(RecurrentPPOAgent):
    """Recurrent PPO with BGD on actor, value network, or both."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        ppo_config: PPOConfig | None = None,
        recurrent_config: RecurrentPPOConfig | None = None,
        bgd_config: BGDPPOConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            observation_dim,
            action_dim,
            action_low=action_low,
            action_high=action_high,
            ppo_config=ppo_config,
            recurrent_config=recurrent_config,
            device=device,
        )
        self.bgd_config = bgd_config or BGDPPOConfig()
        self.bgd_config.validate()

        mode = self.bgd_config.bayesianization
        self.actor_posterior: DiagonalGaussianPosterior | None = None
        self.value_posterior: DiagonalGaussianPosterior | None = None
        self.actor_bgd: BGDUpdater | None = None
        self.value_bgd: BGDUpdater | None = None

        if mode in {
            "actor_only",
            "actor_and_value",
        }:
            self.actor_posterior = DiagonalGaussianPosterior.from_module(
                self.actor,
                prior_std=self.bgd_config.posterior_std,
            )
            self.actor_bgd = BGDUpdater(
                self.actor_posterior,
                self.bgd_config.actor_bgd,
            )

        if mode in {
            "value_only",
            "actor_and_value",
        }:
            self.value_posterior = DiagonalGaussianPosterior.from_module(
                self.value,
                prior_std=self.bgd_config.posterior_std,
            )
            self.value_bgd = BGDUpdater(
                self.value_posterior,
                self.bgd_config.value_bgd,
            )

    def _uncertainty_evidence_weight(
        self,
        epoch_index: int,
    ) -> float:
        if not (0 <= epoch_index < self.config.update_epochs):
            raise ValueError("epoch_index is outside the configured PPO update range")
        mode = self.bgd_config.evidence_mode
        if mode == "all_epochs":
            return 1.0
        if mode == "first_epoch_only":
            return 1.0 if epoch_index == 0 else 0.0
        return 1.0 / float(self.config.update_epochs)

    def _sampled_actor_loss(
        self,
        params: Mapping[str, Tensor],
        batch: RecurrentPPORolloutBatch,
    ) -> Tensor:
        (
            log_prob,
            entropy,
            _,
        ) = functional_call(
            self.actor,
            (
                dict(params),
                dict(self.actor.named_buffers()),
            ),
            (
                batch.observations,
                batch.actions,
                batch.initial_actor_hidden,
                batch.episode_starts,
            ),
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
        return policy_loss - self.config.entropy_coef * entropy.mean()

    def _sampled_value_loss(
        self,
        params: Mapping[str, Tensor],
        batch: RecurrentPPORolloutBatch,
    ) -> Tensor:
        (
            prediction,
            _,
        ) = functional_call(
            self.value,
            (
                dict(params),
                dict(self.value.named_buffers()),
            ),
            (
                batch.observations,
                batch.initial_value_hidden,
                batch.episode_starts,
            ),
        )
        if self.config.value_clip_ratio is None:
            value_loss = 0.5 * torch.nn.functional.mse_loss(
                prediction,
                batch.returns,
            )
        else:
            clipped = batch.old_values + (prediction - batch.old_values).clamp(
                -self.config.value_clip_ratio,
                self.config.value_clip_ratio,
            )
            plain_loss = (prediction - batch.returns).square()
            clipped_loss = (clipped - batch.returns).square()
            value_loss = (
                0.5
                * torch.maximum(
                    plain_loss,
                    clipped_loss,
                ).mean()
            )
        return self.config.value_coef * value_loss

    def _measure_batch(
        self,
        batch: RecurrentPPORolloutBatch,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
    ]:
        with torch.no_grad():
            (
                _,
                policy_loss,
                entropy,
                approx_kl,
                clip_fraction,
            ) = self._actor_loss(batch)
            value_loss = self._value_loss(batch)
        return (
            float(policy_loss.item()),
            float(value_loss.item()),
            float(entropy.item()),
            float(approx_kl.item()),
            float(clip_fraction.item()),
        )

    def update(
        self,
        rollout: RecurrentRolloutBuffer,
    ) -> dict[str, float]:
        rollout.compute_gae(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize_advantages=self.config.normalize_advantages,
        )
        generator = torch.Generator(device=self.device).manual_seed(self.update_count + 98_765)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        bayes_totals: dict[
            str,
            float,
        ] = {}
        evidence_weight_total = 0.0
        sequences = 0
        stop_early = False
        mode = self.bgd_config.bayesianization

        for epoch_index in range(self.config.update_epochs):
            evidence_weight = self._uncertainty_evidence_weight(epoch_index)
            for batch in rollout.sequence_batches(
                self.recurrent_config.sequence_length,
                generator=generator,
            ):
                actor_result: BGDStepResult | None = None
                value_result: BGDStepResult | None = None

                if mode in {
                    "actor_only",
                    "actor_and_value",
                }:
                    assert self.actor_bgd is not None
                    assert self.actor_posterior is not None

                    def actor_objective(
                        params: Mapping[str, Tensor],
                        current_batch: RecurrentPPORolloutBatch = batch,
                        current_evidence_weight: float = evidence_weight,
                    ) -> BGDLoss:
                        loss = self._sampled_actor_loss(
                            params,
                            current_batch,
                        )
                        return BGDLoss(
                            mean=loss,
                            uncertainty=(current_evidence_weight * loss),
                        )

                    actor_result = self.actor_bgd.step(
                        actor_objective,
                        generator=generator,
                    )
                    self.actor_posterior.sync_module(self.actor)
                else:
                    (
                        actor_loss,
                        _,
                        _,
                        _,
                        _,
                    ) = self._actor_loss(batch)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        self.config.gradient_clip_norm,
                    )
                    self.actor_optimizer.step()

                if mode in {
                    "value_only",
                    "actor_and_value",
                }:
                    assert self.value_bgd is not None
                    assert self.value_posterior is not None

                    def value_objective(
                        params: Mapping[str, Tensor],
                        current_batch: RecurrentPPORolloutBatch = batch,
                        current_evidence_weight: float = evidence_weight,
                    ) -> BGDLoss:
                        loss = self._sampled_value_loss(
                            params,
                            current_batch,
                        )
                        return BGDLoss(
                            mean=loss,
                            uncertainty=(current_evidence_weight * loss),
                        )

                    value_result = self.value_bgd.step(
                        value_objective,
                        generator=generator,
                    )
                    self.value_posterior.sync_module(self.value)
                else:
                    value_loss = self._value_loss(batch)
                    self.value_optimizer.zero_grad(set_to_none=True)
                    (self.config.value_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(
                        self.value.parameters(),
                        self.config.gradient_clip_norm,
                    )
                    self.value_optimizer.step()

                (
                    policy_value,
                    value_value,
                    entropy_value,
                    kl_value,
                    clip_value,
                ) = self._measure_batch(batch)
                totals["policy_loss"] += policy_value
                totals["value_loss"] += value_value
                totals["entropy"] += entropy_value
                totals["approx_kl"] += kl_value
                totals["clip_fraction"] += clip_value
                evidence_weight_total += evidence_weight
                sequences += 1

                if actor_result is not None:
                    for (
                        name,
                        value,
                    ) in {
                        "actor_sigma_mean": actor_result.diagnostics["sigma_mean"],
                        "actor_effective_lr_mean": actor_result.diagnostics["effective_lr_mean"],
                        "actor_gradient_norm": actor_result.gradient_norm,
                        "actor_c_norm": actor_result.c_norm,
                    }.items():
                        bayes_totals[name] = bayes_totals.get(
                            name,
                            0.0,
                        ) + float(value)

                if value_result is not None:
                    for (
                        name,
                        value,
                    ) in {
                        "value_sigma_mean": value_result.diagnostics["sigma_mean"],
                        "value_effective_lr_mean": value_result.diagnostics["effective_lr_mean"],
                        "value_gradient_norm": value_result.gradient_norm,
                        "value_c_norm": value_result.c_norm,
                    }.items():
                        bayes_totals[name] = bayes_totals.get(
                            name,
                            0.0,
                        ) + float(value)

                if self.config.target_kl is not None and kl_value > self.config.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        if sequences == 0:
            raise RuntimeError("BGD recurrent PPO update produced no sequence chunks")
        self.update_count += 1
        metrics = {
            name: value / sequences
            for (
                name,
                value,
            ) in totals.items()
        }
        metrics["epochs_early_stopped"] = float(stop_early)
        metrics["uncertainty_evidence_weight_mean"] = evidence_weight_total / sequences
        metrics["sequence_chunks"] = float(sequences)
        for (
            name,
            value,
        ) in bayes_totals.items():
            metrics[name] = value / sequences

        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("nonfinite BGD recurrent PPO update metric")
        return metrics

    def state_dict(
        self,
    ) -> dict[str, Any]:
        state = super().state_dict()
        state["bgd_recurrent_ppo_version"] = 1
        state["bayesianization"] = self.bgd_config.bayesianization
        state["evidence_mode"] = self.bgd_config.evidence_mode
        if self.actor_bgd is not None:
            state["actor_bgd"] = self.actor_bgd.state_dict()
        if self.value_bgd is not None:
            state["value_bgd"] = self.value_bgd.state_dict()
        return state

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if state.get("bgd_recurrent_ppo_version") != 1:
            raise ValueError("unsupported BGD recurrent PPO checkpoint version")
        if state.get("bayesianization") != self.bgd_config.bayesianization:
            raise ValueError("BGD recurrent PPO Bayesianization mode mismatch")
        if state.get("evidence_mode") != self.bgd_config.evidence_mode:
            raise ValueError("BGD recurrent PPO evidence mode mismatch")
        super().load_state_dict(state)
        if self.actor_bgd is not None:
            self.actor_bgd.load_state_dict(state["actor_bgd"])
            assert self.actor_posterior is not None
            self.actor_posterior.sync_module(self.actor)
        if self.value_bgd is not None:
            self.value_bgd.load_state_dict(state["value_bgd"])
            assert self.value_posterior is not None
            self.value_posterior.sync_module(self.value)
