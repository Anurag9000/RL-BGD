"""Bayesian Gradient Descent for recurrent sequence-replay SAC."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.func import functional_call

from rl_bgd.agents.sac.agent import SACConfig
from rl_bgd.agents.sac.bgd_agent import BGDSACConfig
from rl_bgd.agents.sac.recurrent_agent import (
    RecurrentSACAgent,
    RecurrentSACConfig,
)
from rl_bgd.bayes.bgd import (
    BGDLoss,
    BGDStepResult,
    BGDUpdater,
)
from rl_bgd.bayes.diagonal_gaussian import (
    DiagonalGaussianPosterior,
    PosteriorBounds,
)
from rl_bgd.replay.sequence_buffer import (
    SequenceReplayBatch,
)
from rl_bgd.replay.sequence_evidence import (
    SequenceReplayEvidenceSummary,
    sequence_replay_evidence_weights,
    weighted_sequence_evidence_mean,
)
from rl_bgd.surprise.base import (
    SurpriseObservation,
    surprise_to_retention,
)
from rl_bgd.surprise.td import TDSurprise


class BGDRecurrentSACAgent(RecurrentSACAgent):
    """Recurrent SAC with selectable BGD actor and/or critic posteriors."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        sac_config: SACConfig | None = None,
        recurrent_config: RecurrentSACConfig | None = None,
        bgd_config: BGDSACConfig | None = None,
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
        self.bgd_config = bgd_config or BGDSACConfig()
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

        if mode in {
            "actor_only",
            "actor_and_critic",
        }:
            self.actor_posterior = DiagonalGaussianPosterior.from_module(
                self.actor,
                prior_std=self.bgd_config.posterior_std,
                bounds=bounds,
            )
            self.actor_bgd = BGDUpdater(
                self.actor_posterior,
                self.bgd_config.actor_bgd,
            )

        if mode in {
            "critic_only",
            "actor_and_critic",
        }:
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
    ) -> tuple[
        float | None,
        SurpriseObservation | None,
    ]:
        config = self.bgd_config.adaptive_td_retention
        if config is None or self.td_surprise is None:
            return None, None
        with torch.no_grad():
            q1, q2 = self._critic_predictions(
                batch
            )
            td_errors = torch.cat(
                [
                    target - q1,
                    target - q2,
                ],
                dim=0,
            )
        observation = self.td_surprise.observe(
            td_errors
        )
        retention = surprise_to_retention(
            observation.smoothed,
            config.mapping,
        )
        return retention, observation

    def _sampled_critic_loss(
        self,
        module: torch.nn.Module,
        params: dict[str, Tensor],
        batch: SequenceReplayBatch,
        target: Tensor,
        evidence: SequenceReplayEvidenceSummary,
    ) -> BGDLoss:
        q = functional_call(
            module,
            (
                params,
                dict(
                    module.named_buffers()
                ),
            ),
            (
                batch.observations,
                batch.actions,
                batch.episode_starts,
            ),
        )[
            :,
            batch.unroll_slice,
        ]
        per_transition = torch.nn.functional.mse_loss(
            q,
            target,
            reduction="none",
        )
        return BGDLoss(
            mean=per_transition.mean(),
            uncertainty=weighted_sequence_evidence_mean(
                per_transition,
                evidence.weights,
            ),
        )

    def _update_critics_bgd(
        self,
        batch: SequenceReplayBatch,
        target: Tensor,
        evidence: SequenceReplayEvidenceSummary,
        retention: float | None,
    ) -> tuple[
        BGDStepResult,
        BGDStepResult,
    ]:
        if (
            self.critic1_bgd is None
            or self.critic2_bgd is None
            or self.critic1_posterior is None
            or self.critic2_posterior is None
        ):
            raise RuntimeError(
                "recurrent critic BGD is not configured"
            )

        def objective1(
            params: dict[str, Tensor],
        ) -> BGDLoss:
            return self._sampled_critic_loss(
                self.critic1,
                params,
                batch,
                target,
                evidence,
            )

        def objective2(
            params: dict[str, Tensor],
        ) -> BGDLoss:
            return self._sampled_critic_loss(
                self.critic2,
                params,
                batch,
                target,
                evidence,
            )

        result1 = self.critic1_bgd.step(
            objective1,
            retention=retention,
        )
        result2 = self.critic2_bgd.step(
            objective2,
            retention=retention,
        )
        self.critic1_posterior.sync_module(
            self.critic1
        )
        self.critic2_posterior.sync_module(
            self.critic2
        )
        return result1, result2

    def _update_actor_bgd(
        self,
        batch: SequenceReplayBatch,
        evidence: SequenceReplayEvidenceSummary,
        retention: float | None,
    ) -> BGDStepResult:
        if (
            self.actor_bgd is None
            or self.actor_posterior is None
        ):
            raise RuntimeError(
                "recurrent actor BGD is not configured"
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
            critic1_hidden = critic1_hidden.detach()
            critic2_hidden = critic2_hidden.detach()
        self._set_critics_trainable(
            False
        )

        def objective(
            params: dict[str, Tensor],
        ) -> BGDLoss:
            (
                sampled_actions,
                log_prob,
                _,
                _,
                _,
            ) = functional_call(
                self.actor,
                (
                    params,
                    dict(
                        self.actor.named_buffers()
                    ),
                ),
                (
                    batch.observations,
                    batch.episode_starts,
                    None,
                ),
            )
            q1 = self.critic1.q_from_hidden(
                critic1_hidden,
                sampled_actions,
            )
            q2 = self.critic2.q_from_hidden(
                critic2_hidden,
                sampled_actions,
            )
            selector = batch.unroll_slice
            per_transition = (
                self.alpha.detach()
                * log_prob[
                    :,
                    selector,
                ]
                - torch.minimum(
                    q1[
                        :,
                        selector,
                    ],
                    q2[
                        :,
                        selector,
                    ],
                )
            )
            return BGDLoss(
                mean=per_transition.mean(),
                uncertainty=weighted_sequence_evidence_mean(
                    per_transition,
                    evidence.weights,
                ),
            )

        try:
            result = self.actor_bgd.step(
                objective,
                retention=retention,
            )
        finally:
            self._set_critics_trainable(
                True
            )
        self.actor_posterior.sync_module(
            self.actor
        )
        return result

    def update(
        self,
        batch: SequenceReplayBatch,
    ) -> dict[str, float]:
        mode = self.bgd_config.bayesianization
        target = self._target_values(
            batch
        )
        evidence = self._evidence(
            batch
        )
        retention, surprise = (
            self._adaptive_retention(
                batch,
                target,
            )
        )

        critic1_result: BGDStepResult | None = None
        critic2_result: BGDStepResult | None = None
        if mode in {
            "critic_only",
            "actor_and_critic",
        }:
            (
                critic1_result,
                critic2_result,
            ) = self._update_critics_bgd(
                batch,
                target,
                evidence,
                retention,
            )
            q1, q2 = self._critic_predictions(
                batch
            )
            critic_loss_value = (
                torch.nn.functional.mse_loss(
                    q1,
                    target,
                )
                + torch.nn.functional.mse_loss(
                    q2,
                    target,
                )
            ).detach()
        else:
            q1, q2 = self._critic_predictions(
                batch
            )
            critic_loss = (
                torch.nn.functional.mse_loss(
                    q1,
                    target,
                )
                + torch.nn.functional.mse_loss(
                    q2,
                    target,
                )
            )
            self.critic_optimizer.zero_grad(
                set_to_none=True
            )
            critic_loss.backward()
            parameters = (
                list(
                    self.critic1.parameters()
                )
                + list(
                    self.critic2.parameters()
                )
            )
            self._clip_gradients(
                parameters
            )
            self.critic_optimizer.step()
            critic_loss_value = critic_loss.detach()

        actor_result: BGDStepResult | None = None
        if mode in {
            "actor_only",
            "actor_and_critic",
        }:
            actor_result = self._update_actor_bgd(
                batch,
                evidence,
                retention,
            )
            with torch.no_grad():
                (
                    actor_terms,
                    log_prob,
                    _,
                ) = self._actor_terms(
                    batch
                )
                actor_loss_value = actor_terms.mean()
        else:
            self._set_critics_trainable(
                False
            )
            try:
                (
                    actor_terms,
                    log_prob,
                    _,
                ) = self._actor_terms(
                    batch
                )
                actor_loss = actor_terms.mean()
                self.actor_optimizer.zero_grad(
                    set_to_none=True
                )
                actor_loss.backward()
                self._clip_gradients(
                    list(
                        self.actor.parameters()
                    )
                )
                self.actor_optimizer.step()
                actor_loss_value = actor_loss.detach()
            finally:
                self._set_critics_trainable(
                    True
                )

        if mode in {
            "actor_only",
            "actor_and_critic",
        }:
            with torch.no_grad():
                _, alpha_log_prob, _ = self._actor_terms(
                    batch
                )
        else:
            alpha_log_prob = log_prob

        alpha_loss = torch.zeros(
            (),
            device=self.device,
        )
        if self.config.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha
                * (
                    alpha_log_prob.detach()
                    + self.target_entropy
                )
            ).mean()
            self.alpha_optimizer.zero_grad(
                set_to_none=True
            )
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._polyak_update()
        self.update_count += 1
        metrics: dict[str, float] = {
            "critic_loss": float(
                critic_loss_value.item()
            ),
            "actor_loss": float(
                actor_loss_value.item()
            ),
            "alpha_loss": float(
                alpha_loss.detach().item()
            ),
            "alpha": float(
                self.alpha.detach().item()
            ),
            "q1_mean": float(
                q1.detach().mean().item()
            ),
            "q2_mean": float(
                q2.detach().mean().item()
            ),
            "target_q_mean": float(
                target.detach().mean().item()
            ),
            "policy_entropy_estimate": float(
                (
                    -alpha_log_prob.detach()
                ).mean().item()
            ),
            "evidence_weight_mean": evidence.mean_weight,
            "evidence_weight_min": evidence.min_weight,
            "evidence_weight_max": evidence.max_weight,
            "evidence_fresh_fraction": evidence.fresh_fraction,
            "evidence_mean_usage_count": evidence.mean_usage_count,
            "evidence_effective_sample_size": evidence.effective_sample_size,
        }
        if (
            surprise is not None
            and retention is not None
        ):
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
        if (
            critic1_result is not None
            and critic2_result is not None
        ):
            metrics.update(
                {
                    "critic1_sigma_mean": critic1_result.diagnostics[
                        "sigma_mean"
                    ],
                    "critic2_sigma_mean": critic2_result.diagnostics[
                        "sigma_mean"
                    ],
                    "critic1_effective_lr_mean": critic1_result.diagnostics[
                        "effective_lr_mean"
                    ],
                    "critic2_effective_lr_mean": critic2_result.diagnostics[
                        "effective_lr_mean"
                    ],
                }
            )
        if actor_result is not None:
            metrics.update(
                {
                    "actor_sigma_mean": actor_result.diagnostics[
                        "sigma_mean"
                    ],
                    "actor_effective_lr_mean": actor_result.diagnostics[
                        "effective_lr_mean"
                    ],
                }
            )
        if not all(
            torch.isfinite(
                torch.tensor(
                    value
                )
            )
            for value in metrics.values()
        ):
            raise FloatingPointError(
                "nonfinite BGD recurrent SAC metric"
            )
        return metrics

    def state_dict(
        self,
    ) -> dict[str, Any]:
        state = super().state_dict()
        state[
            "bgd_recurrent_sac_version"
        ] = 1
        state[
            "bayesianization"
        ] = self.bgd_config.bayesianization
        state[
            "replay_evidence_mode"
        ] = self.bgd_config.replay_evidence.mode
        state[
            "adaptive_td_retention"
        ] = (
            self.bgd_config.adaptive_td_retention
            is not None
        )
        if self.td_surprise is not None:
            state[
                "td_surprise"
            ] = self.td_surprise.state_dict()
        if self.actor_bgd is not None:
            state[
                "actor_bgd"
            ] = self.actor_bgd.state_dict()
        if (
            self.critic1_bgd is not None
            and self.critic2_bgd is not None
        ):
            state[
                "critic1_bgd"
            ] = self.critic1_bgd.state_dict()
            state[
                "critic2_bgd"
            ] = self.critic2_bgd.state_dict()
        return state

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if state.get(
            "bgd_recurrent_sac_version"
        ) != 1:
            raise ValueError(
                "unsupported BGD recurrent SAC checkpoint version"
            )
        if (
            state.get(
                "bayesianization"
            )
            != self.bgd_config.bayesianization
        ):
            raise ValueError(
                "BGD recurrent SAC Bayesianization mode mismatch"
            )
        if (
            state.get(
                "replay_evidence_mode"
            )
            != self.bgd_config.replay_evidence.mode
        ):
            raise ValueError(
                "BGD recurrent SAC evidence mode mismatch"
            )
        expected_adaptive = (
            self.bgd_config.adaptive_td_retention
            is not None
        )
        if (
            bool(
                state.get(
                    "adaptive_td_retention",
                    False,
                )
            )
            != expected_adaptive
        ):
            raise ValueError(
                "BGD recurrent SAC adaptive-retention mismatch"
            )
        super().load_state_dict(
            state
        )
        if self.td_surprise is not None:
            payload = state[
                "td_surprise"
            ]
            if not isinstance(
                payload,
                dict,
            ):
                raise TypeError(
                    "TD-surprise state must be a dictionary"
                )
            self.td_surprise.load_state_dict(
                payload
            )
        if self.actor_bgd is not None:
            self.actor_bgd.load_state_dict(
                state[
                    "actor_bgd"
                ]
            )
            assert (
                self.actor_posterior
                is not None
            )
            self.actor_posterior.sync_module(
                self.actor
            )
        if (
            self.critic1_bgd is not None
            and self.critic2_bgd is not None
        ):
            self.critic1_bgd.load_state_dict(
                state[
                    "critic1_bgd"
                ]
            )
            self.critic2_bgd.load_state_dict(
                state[
                    "critic2_bgd"
                ]
            )
            assert (
                self.critic1_posterior
                is not None
                and self.critic2_posterior
                is not None
            )
            self.critic1_posterior.sync_module(
                self.critic1
            )
            self.critic2_posterior.sync_module(
                self.critic2
            )
