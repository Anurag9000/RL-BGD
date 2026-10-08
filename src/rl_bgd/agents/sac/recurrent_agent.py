"""Recurrent Soft Actor-Critic for hidden-context continual control."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.utils.checkpoint_transaction import transactional_state_load
from rl_bgd.agents.sac.agent import (
    SACConfig,
)
from rl_bgd.models.recurrent_sac import (
    RecurrentQNetwork,
    RecurrentSACActor,
)
from rl_bgd.replay.sequence_buffer import (
    SequenceReplayBatch,
)


@dataclass(frozen=True)
class RecurrentSACConfig:
    """Matched recurrent architecture controls."""

    recurrent_hidden_dim: int = 64
    encoder_hidden_dims: tuple[
        int,
        ...,
    ] = (64,)
    q_hidden_dims: tuple[
        int,
        ...,
    ] = (
        64,
        64,
    )

    def validate(self) -> None:
        if self.recurrent_hidden_dim < 1:
            raise ValueError("recurrent_hidden_dim must be positive")
        if any(
            width < 1
            for width in (
                *self.encoder_hidden_dims,
                *self.q_hidden_dims,
            )
        ):
            raise ValueError("recurrent SAC hidden widths must be positive")


class RecurrentSACAgent:
    """Twin-critic SAC using observation-history GRU state."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        sac_config: SACConfig | None = None,
        recurrent_config: RecurrentSACConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = sac_config or SACConfig()
        self.config.validate()
        self.recurrent_config = recurrent_config or RecurrentSACConfig()
        self.recurrent_config.validate()
        self.device = torch.device(device)
        hidden_dim = self.recurrent_config.recurrent_hidden_dim
        encoder_dims = self.recurrent_config.encoder_hidden_dims
        q_dims = self.recurrent_config.q_hidden_dims
        low = action_low.to(
            self.device,
            dtype=torch.float32,
        )
        high = action_high.to(
            self.device,
            dtype=torch.float32,
        )
        self.actor = RecurrentSACActor(
            observation_dim,
            action_dim,
            action_low=low,
            action_high=high,
            recurrent_hidden_dim=hidden_dim,
            encoder_hidden_dims=encoder_dims,
        ).to(self.device)
        self.critic1 = RecurrentQNetwork(
            observation_dim,
            action_dim,
            recurrent_hidden_dim=hidden_dim,
            encoder_hidden_dims=encoder_dims,
            q_hidden_dims=q_dims,
        ).to(self.device)
        self.critic2 = RecurrentQNetwork(
            observation_dim,
            action_dim,
            recurrent_hidden_dim=hidden_dim,
            encoder_hidden_dims=encoder_dims,
            q_hidden_dims=q_dims,
        ).to(self.device)
        self.target1 = copy.deepcopy(self.critic1).eval()
        self.target2 = copy.deepcopy(self.critic2).eval()
        for target in (
            self.target1,
            self.target2,
        ):
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
        self.log_alpha = nn.Parameter(
            torch.log(
                torch.tensor(
                    self.config.initial_alpha,
                    device=self.device,
                )
            )
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.config.alpha_lr,
        )
        self.update_count = 0
        self.recurrent_reset_count = 0
        self.actor_hidden = self.actor.initial_state()

    @property
    def alpha(self) -> Tensor:
        if self.config.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return torch.tensor(
            self.config.initial_alpha,
            device=self.device,
        )

    def reset_recurrent_state(
        self,
    ) -> None:
        self.actor_hidden = self.actor.initial_state()
        self.recurrent_reset_count += 1

    @torch.no_grad()
    def advance_actor_hidden(
        self,
        observation: Tensor,
    ) -> None:
        obs = observation.to(
            self.device,
            dtype=torch.float32,
        ).reshape(
            1,
            -1,
        )
        self.actor_hidden = self.actor.step_hidden(
            obs,
            self.actor_hidden,
        ).detach()

    @torch.no_grad()
    def act_recurrent(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor:
        obs = observation.to(
            self.device,
            dtype=torch.float32,
        ).reshape(
            1,
            -1,
        )
        if deterministic:
            action, hidden = self.actor.deterministic_step(
                obs,
                self.actor_hidden,
            )
        else:
            (
                action,
                _,
                _,
                hidden,
            ) = self.actor.sample_step(
                obs,
                self.actor_hidden,
            )
        self.actor_hidden = hidden.detach()
        return action.squeeze(0)

    @torch.no_grad()
    def rebuild_actor_hidden(
        self,
        observations: Sequence[Tensor],
    ) -> None:
        """Recompute online hidden state under the updated actor."""

        if not observations:
            self.actor_hidden = self.actor.initial_state()
            return
        sequence = torch.stack(
            [
                observation.to(
                    self.device,
                    dtype=torch.float32,
                ).reshape(-1)
                for observation in observations
            ],
            dim=0,
        ).unsqueeze(0)
        starts = torch.zeros(
            1,
            sequence.shape[1],
            1,
            device=self.device,
            dtype=torch.bool,
        )
        starts[
            0,
            0,
            0,
        ] = True
        _, hidden = self.actor.hidden_sequence(
            sequence,
            starts,
        )
        self.actor_hidden = hidden.detach()

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
        batch: SequenceReplayBatch,
    ) -> Tensor:
        (
            actor_hidden,
            _,
        ) = self.actor.hidden_sequence(
            batch.observations,
            batch.episode_starts,
        )
        (
            target1_hidden,
            _,
        ) = self.target1.hidden_sequence(
            batch.observations,
            batch.episode_starts,
        )
        (
            target2_hidden,
            _,
        ) = self.target2.hidden_sequence(
            batch.observations,
            batch.episode_starts,
        )
        (
            next_actions,
            next_log_prob,
            _,
            _,
        ) = self.actor.sample_next_from_hidden(
            batch.next_observations,
            actor_hidden,
        )
        (
            target1_q,
            _,
        ) = self.target1.next_q_from_hidden(
            batch.next_observations,
            next_actions,
            target1_hidden,
        )
        (
            target2_q,
            _,
        ) = self.target2.next_q_from_hidden(
            batch.next_observations,
            next_actions,
            target2_hidden,
        )
        target_q = torch.minimum(
            target1_q,
            target2_q,
        ) - (self.alpha.detach() * next_log_prob)
        full_target = batch.rewards + self.config.gamma * (~batch.terminated).float() * target_q
        return full_target[
            :,
            batch.unroll_slice,
        ]

    def _critic_predictions(
        self,
        batch: SequenceReplayBatch,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        q1 = self.critic1(
            batch.observations,
            batch.actions,
            batch.episode_starts,
        )[
            :,
            batch.unroll_slice,
        ]
        q2 = self.critic2(
            batch.observations,
            batch.actions,
            batch.episode_starts,
        )[
            :,
            batch.unroll_slice,
        ]
        return (
            q1,
            q2,
        )

    def _actor_terms(
        self,
        batch: SequenceReplayBatch,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        (
            sampled_actions,
            log_prob,
            _,
            _,
            _,
        ) = self.actor.sample_sequence(
            batch.observations,
            batch.episode_starts,
        )
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
        return (
            (
                self.alpha.detach()
                * log_prob[
                    :,
                    selector,
                ]
                - torch.minimum(
                    q1_pi[
                        :,
                        selector,
                    ],
                    q2_pi[
                        :,
                        selector,
                    ],
                )
            ),
            log_prob[
                :,
                selector,
            ],
            sampled_actions[
                :,
                selector,
            ],
        )

    def _set_critics_trainable(
        self,
        trainable: bool,
    ) -> None:
        for critic in (
            self.critic1,
            self.critic2,
        ):
            for parameter in critic.parameters():
                parameter.requires_grad_(trainable)

    def update(
        self,
        batch: SequenceReplayBatch,
    ) -> dict[str, float]:
        target = self._target_values(batch)
        q1, q2 = self._critic_predictions(batch)
        critic_loss = torch.nn.functional.mse_loss(
            q1,
            target,
        ) + torch.nn.functional.mse_loss(
            q2,
            target,
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self._clip_gradients(critic_parameters)
        self.critic_optimizer.step()

        self._set_critics_trainable(False)
        actor_terms, log_prob, _ = self._actor_terms(batch)
        actor_loss = actor_terms.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip_gradients(list(self.actor.parameters()))
        self.actor_optimizer.step()
        self._set_critics_trainable(True)

        alpha_loss = torch.zeros(
            (),
            device=self.device,
        )
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
            raise FloatingPointError("nonfinite recurrent SAC metric")
        return metrics

    @torch.no_grad()
    def _polyak_update(
        self,
    ) -> None:
        for source, target in zip(
            self.critic1.parameters(),
            self.target1.parameters(),
            strict=True,
        ):
            target.lerp_(
                source,
                self.config.tau,
            )
        for source, target in zip(
            self.critic2.parameters(),
            self.target2.parameters(),
            strict=True,
        ):
            target.lerp_(
                source,
                self.config.tau,
            )

    def state_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "checkpoint_version": 2,
            "sac_config": asdict(self.config),
            "recurrent_config": asdict(self.recurrent_config),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "actor_hidden": self.actor_hidden.detach().clone(),
            "update_count": self.update_count,
            "recurrent_reset_count": self.recurrent_reset_count,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if state.get("checkpoint_version") != 2:
            raise ValueError("unsupported recurrent SAC checkpoint version")
        if state.get("sac_config") != asdict(self.config):
            raise ValueError("recurrent SAC configuration mismatch")
        if state.get("recurrent_config") != asdict(self.recurrent_config):
            raise ValueError("recurrent SAC architecture configuration mismatch")

        def apply(payload: dict[str, Any]) -> None:
            for name in (
                "actor",
                "critic1",
                "critic2",
                "target1",
                "target2",
            ):
                getattr(
                    self,
                    name,
                ).load_state_dict(payload[name])
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
            self.actor_hidden = payload["actor_hidden"].to(
                self.device,
                dtype=torch.float32,
            )
            self.update_count = int(payload["update_count"])
            self.recurrent_reset_count = int(payload["recurrent_reset_count"])

        transactional_state_load(
            state,
            current_state=self.state_dict,
            apply=apply,
        )
