"""Matched recurrent PPO baseline for hidden-context continual RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.recurrent_rollout import (
    RecurrentPPORolloutBatch,
    RecurrentRolloutBuffer,
)
from rl_bgd.models.recurrent_ppo import (
    RecurrentPPOSquashedGaussianActor,
    RecurrentValueNetwork,
)


@dataclass(frozen=True)
class RecurrentPPOConfig:
    """Architecture and truncated-BPTT controls shared by Adam and BGD."""

    recurrent_hidden_dim: int = 64
    sequence_length: int = 32
    encoder_hidden_dims: tuple[int, ...] = (64,)

    def validate(self) -> None:
        if (
            self.recurrent_hidden_dim < 1
            or self.sequence_length < 1
        ):
            raise ValueError(
                "recurrent hidden dimension and sequence length must be positive"
            )
        if any(
            width < 1
            for width in self.encoder_hidden_dims
        ):
            raise ValueError(
                "recurrent encoder widths must be positive"
            )


@dataclass(frozen=True)
class RecurrentActionStep:
    """Behavior-policy outputs plus hidden state before/after one observation."""

    action: Tensor
    log_prob: Tensor
    value: Tensor
    actor_hidden_before: Tensor
    value_hidden_before: Tensor
    value_hidden_after: Tensor


class RecurrentPPOAgent:
    """Clipped PPO with GRU actor/value networks and explicit recurrent state."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        ppo_config: PPOConfig | None = None,
        recurrent_config: RecurrentPPOConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = ppo_config or PPOConfig()
        self.config.validate()
        self.recurrent_config = (
            recurrent_config
            or RecurrentPPOConfig()
        )
        self.recurrent_config.validate()
        self.device = torch.device(
            device
        )
        recurrent_hidden_dim = (
            self.recurrent_config.recurrent_hidden_dim
        )
        encoder_hidden_dims = (
            self.recurrent_config.encoder_hidden_dims
        )
        self.actor = (
            RecurrentPPOSquashedGaussianActor(
                observation_dim,
                action_dim,
                action_low=action_low.to(
                    self.device
                ),
                action_high=action_high.to(
                    self.device
                ),
                recurrent_hidden_dim=recurrent_hidden_dim,
                encoder_hidden_dims=encoder_hidden_dims,
            ).to(
                self.device
            )
        )
        self.value = RecurrentValueNetwork(
            observation_dim,
            recurrent_hidden_dim=recurrent_hidden_dim,
            encoder_hidden_dims=encoder_hidden_dims,
        ).to(
            self.device
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(),
            lr=self.config.value_lr,
        )
        self.update_count = 0
        self.recurrent_reset_count = 0
        self.actor_hidden = (
            self.actor.initial_state()
        )
        self.value_hidden = (
            self.value.initial_state()
        )

    def reset_recurrent_state(
        self,
    ) -> None:
        self.actor_hidden = (
            self.actor.initial_state()
        )
        self.value_hidden = (
            self.value.initial_state()
        )
        self.recurrent_reset_count += 1

    @torch.no_grad()
    def sample_action_recurrent(
        self,
        observation: Tensor,
    ) -> RecurrentActionStep:
        obs = observation.to(
            self.device,
            dtype=torch.float32,
        ).reshape(
            1,
            -1,
        )
        actor_hidden_before = (
            self.actor_hidden.detach().clone()
        )
        value_hidden_before = (
            self.value_hidden.detach().clone()
        )
        (
            action,
            log_prob,
            _,
            actor_hidden_after,
        ) = self.actor.sample_step(
            obs,
            self.actor_hidden,
        )
        value, value_hidden_after = (
            self.value.step(
                obs,
                self.value_hidden,
            )
        )
        self.actor_hidden = (
            actor_hidden_after.detach()
        )
        self.value_hidden = (
            value_hidden_after.detach()
        )
        return RecurrentActionStep(
            action=action.squeeze(
                0
            ),
            log_prob=log_prob.squeeze(
                0
            ),
            value=value.squeeze(
                0
            ),
            actor_hidden_before=actor_hidden_before.squeeze(
                0
            ),
            value_hidden_before=value_hidden_before.squeeze(
                0
            ),
            value_hidden_after=value_hidden_after.detach().squeeze(
                0
            ),
        )

    @torch.no_grad()
    def value_from_hidden(
        self,
        observation: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        obs = observation.to(
            self.device,
            dtype=torch.float32,
        ).reshape(
            1,
            -1,
        )
        value, _ = self.value.step(
            obs,
            hidden.to(
                self.device,
                dtype=torch.float32,
            ).reshape(
                1,
                -1,
            ),
        )
        return value.squeeze(
            0
        )

    @torch.no_grad()
    def act_recurrent(
        self,
        observation: Tensor,
        *,
        deterministic: bool = True,
    ) -> Tensor:
        obs = observation.to(
            self.device,
            dtype=torch.float32,
        ).reshape(
            1,
            -1,
        )
        if deterministic:
            action, next_hidden = (
                self.actor.deterministic_step(
                    obs,
                    self.actor_hidden,
                )
            )
        else:
            (
                action,
                _,
                _,
                next_hidden,
            ) = self.actor.sample_step(
                obs,
                self.actor_hidden,
            )
        self.actor_hidden = (
            next_hidden.detach()
        )
        return action.squeeze(
            0
        )

    def _actor_loss(
        self,
        batch: RecurrentPPORolloutBatch,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        (
            log_prob,
            entropy,
            _,
        ) = (
            self.actor.evaluate_actions_sequence(
                batch.observations,
                batch.actions,
                batch.initial_actor_hidden,
                batch.episode_starts,
            )
        )
        log_ratio = (
            log_prob
            - batch.old_log_probs
        )
        ratio = log_ratio.exp()
        surrogate1 = (
            ratio
            * batch.advantages
        )
        surrogate2 = (
            ratio.clamp(
                1.0
                - self.config.clip_ratio,
                1.0
                + self.config.clip_ratio,
            )
            * batch.advantages
        )
        policy_loss = -torch.minimum(
            surrogate1,
            surrogate2,
        ).mean()
        entropy_mean = entropy.mean()
        actor_loss = (
            policy_loss
            - self.config.entropy_coef
            * entropy_mean
        )
        approx_kl = (
            (ratio - 1.0)
            - log_ratio
        ).mean()
        clip_fraction = (
            torch.abs(
                ratio - 1.0
            )
            > self.config.clip_ratio
        ).float().mean()
        return (
            actor_loss,
            policy_loss,
            entropy_mean,
            approx_kl,
            clip_fraction,
        )

    def _value_loss(
        self,
        batch: RecurrentPPORolloutBatch,
    ) -> Tensor:
        prediction, _ = self.value(
            batch.observations,
            batch.initial_value_hidden,
            batch.episode_starts,
        )
        if (
            self.config.value_clip_ratio
            is None
        ):
            return (
                0.5
                * torch.nn.functional.mse_loss(
                    prediction,
                    batch.returns,
                )
            )
        clipped = (
            batch.old_values
            + (
                prediction
                - batch.old_values
            ).clamp(
                -self.config.value_clip_ratio,
                self.config.value_clip_ratio,
            )
        )
        plain_loss = (
            prediction
            - batch.returns
        ).square()
        clipped_loss = (
            clipped
            - batch.returns
        ).square()
        return (
            0.5
            * torch.maximum(
                plain_loss,
                clipped_loss,
            ).mean()
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
        generator = torch.Generator(
            device=self.device
        ).manual_seed(
            self.update_count
            + 87_654
        )
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        sequences = 0
        stop_early = False
        for _ in range(
            self.config.update_epochs
        ):
            for batch in rollout.sequence_batches(
                self.recurrent_config.sequence_length,
                generator=generator,
            ):
                (
                    actor_loss,
                    policy_loss,
                    entropy,
                    approx_kl,
                    clip_fraction,
                ) = self._actor_loss(
                    batch
                )
                self.actor_optimizer.zero_grad(
                    set_to_none=True
                )
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.actor_optimizer.step()

                value_loss = (
                    self._value_loss(
                        batch
                    )
                )
                self.value_optimizer.zero_grad(
                    set_to_none=True
                )
                (
                    self.config.value_coef
                    * value_loss
                ).backward()
                nn.utils.clip_grad_norm_(
                    self.value.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.value_optimizer.step()

                for name, value in (
                    (
                        "policy_loss",
                        policy_loss,
                    ),
                    (
                        "value_loss",
                        value_loss,
                    ),
                    (
                        "entropy",
                        entropy,
                    ),
                    (
                        "approx_kl",
                        approx_kl,
                    ),
                    (
                        "clip_fraction",
                        clip_fraction,
                    ),
                ):
                    totals[
                        name
                    ] += float(
                        value.detach().item()
                    )
                sequences += 1
                if (
                    self.config.target_kl
                    is not None
                    and float(
                        approx_kl.detach().item()
                    )
                    > self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break
        if sequences == 0:
            raise RuntimeError(
                "recurrent PPO update produced no sequence chunks"
            )
        self.update_count += 1
        metrics = {
            name: value
            / sequences
            for name, value in totals.items()
        }
        metrics[
            "epochs_early_stopped"
        ] = float(
            stop_early
        )
        metrics[
            "sequence_chunks"
        ] = float(
            sequences
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
                "nonfinite recurrent PPO update metric"
            )
        return metrics

    @torch.no_grad()
    def refresh_recurrent_state(
        self,
        rollout: RecurrentRolloutBuffer,
    ) -> None:
        """Recompute current hidden state under updated parameters.

        The chunk-start hidden at the beginning of a rollout is behavior-policy
        state. Replaying the rollout after each update prevents additional
        stale-state drift across rollout boundaries while preserving the
        truncated-BPTT approximation at the first observation.
        """

        if len(
            rollout
        ) == 0:
            return
        _, actor_hidden = (
            self.actor._sequence_hidden(
                rollout.observations[
                    : rollout.size
                ],
                rollout.actor_hiddens[
                    0
                ],
                rollout.episode_starts[
                    : rollout.size
                ],
            )
        )
        _, value_hidden = self.value(
            rollout.observations[
                : rollout.size
            ],
            rollout.value_hiddens[
                0
            ],
            rollout.episode_starts[
                : rollout.size
            ],
        )
        self.actor_hidden = (
            actor_hidden.detach()
        )
        self.value_hidden = (
            value_hidden.detach()
        )

    def state_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "checkpoint_version": 1,
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "actor_hidden": self.actor_hidden.detach().clone(),
            "value_hidden": self.value_hidden.detach().clone(),
            "update_count": self.update_count,
            "recurrent_reset_count": self.recurrent_reset_count,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if state.get(
            "checkpoint_version"
        ) != 1:
            raise ValueError(
                "unsupported recurrent PPO checkpoint version"
            )
        self.actor.load_state_dict(
            state[
                "actor"
            ]
        )
        self.value.load_state_dict(
            state[
                "value"
            ]
        )
        self.actor_optimizer.load_state_dict(
            state[
                "actor_optimizer"
            ]
        )
        self.value_optimizer.load_state_dict(
            state[
                "value_optimizer"
            ]
        )
        self.actor_hidden = state[
            "actor_hidden"
        ].to(
            self.device,
            dtype=torch.float32,
        )
        self.value_hidden = state[
            "value_hidden"
        ].to(
            self.device,
            dtype=torch.float32,
        )
        self.update_count = int(
            state[
                "update_count"
            ]
        )
        self.recurrent_reset_count = int(
            state[
                "recurrent_reset_count"
            ]
        )
