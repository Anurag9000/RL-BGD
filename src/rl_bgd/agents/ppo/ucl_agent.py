"""Oracle-boundary PPO-UCL baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rl_bgd.agents.ppo.agent import PPOConfig
from rl_bgd.agents.ppo.rollout import PPORolloutBatch, RolloutBuffer
from rl_bgd.baselines.ucl import (
    UCLLayerSnapshot,
    UCLRegularizationConfig,
    snapshot_ucl_layers,
    ucl_regularization,
)
from rl_bgd.models.ucl_ppo import UCLPPOActor, UCLValueNetwork
from rl_bgd.utils.checkpoint_progress import (
    checkpoint_integer,
    checkpoint_nonnegative_integer,
)
from rl_bgd.utils.checkpoint_transaction import transactional_state_load


@dataclass(frozen=True)
class UCLPPOConfig:
    """UCL posterior and regularization controls."""

    variance_ratio: float = 1.0 / 32.0
    beta: float = 0.03
    rho_reference: float = -2.783

    def validate(self) -> None:
        if not 0.0 < self.variance_ratio < 1.0:
            raise ValueError("UCL variance_ratio must lie in (0, 1)")
        UCLRegularizationConfig(
            beta=self.beta,
            rho_reference=self.rho_reference,
        ).validate()


class UCLPPOAgent:
    """PPO with UCL Bayesian hidden layers and boundary snapshots."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        action_low: Tensor,
        action_high: Tensor,
        hidden_dims: tuple[int, ...] = (64, 64),
        ppo_config: PPOConfig | None = None,
        ucl_config: UCLPPOConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = ppo_config or PPOConfig()
        self.config.validate()
        self.ucl_config = ucl_config or UCLPPOConfig()
        self.ucl_config.validate()
        self.device = torch.device(device)

        self.actor = UCLPPOActor(
            observation_dim,
            action_dim,
            action_low=action_low.to(self.device),
            action_high=action_high.to(self.device),
            hidden_dims=hidden_dims,
            ratio=self.ucl_config.variance_ratio,
        ).to(self.device)
        self.value = UCLValueNetwork(
            observation_dim,
            hidden_dims=hidden_dims,
            ratio=self.ucl_config.variance_ratio,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(),
            lr=self.config.value_lr,
        )
        self.actor_snapshot = snapshot_ucl_layers(self.actor.bayesian_layers)
        self.value_snapshot = snapshot_ucl_layers(self.value.bayesian_layers)
        self.boundary_count = 0
        self.update_count = 0

    @property
    def regularization_config(self) -> UCLRegularizationConfig:
        return UCLRegularizationConfig(
            beta=self.ucl_config.beta,
            rho_reference=self.ucl_config.rho_reference,
        )

    def consolidate_boundary(self) -> None:
        """Save the current posterior as the previous-task posterior."""

        self.actor_snapshot = snapshot_ucl_layers(self.actor.bayesian_layers)
        self.value_snapshot = snapshot_ucl_layers(self.value.bayesian_layers)
        self.boundary_count += 1

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
            action, _, _ = self.actor.sample(
                obs,
                sample_weights=True,
            )
        return action.squeeze(0) if squeeze else action

    @torch.no_grad()
    def sample_action(
        self,
        observation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs = observation.to(self.device, dtype=torch.float32)
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        action, log_prob, _ = self.actor.sample(
            obs,
            sample_weights=True,
        )
        value = self.value(
            obs,
            sample_weights=True,
        )
        if squeeze:
            return (
                action.squeeze(0),
                log_prob.squeeze(0),
                value.squeeze(0),
            )
        return action, log_prob, value

    @torch.no_grad()
    def value_of(self, observation: Tensor) -> Tensor:
        obs = observation.to(self.device, dtype=torch.float32)
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        value = self.value(
            obs,
            sample_weights=True,
        )
        return value.squeeze(0) if squeeze else value

    def _actor_loss(
        self,
        batch: PPORolloutBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        log_prob, entropy = self.actor.evaluate_actions(
            batch.observations,
            batch.actions,
            sample_weights=True,
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
        actor_loss = policy_loss - self.config.entropy_coef * entropy.mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > self.config.clip_ratio).float().mean()
        return actor_loss, policy_loss, approx_kl, clip_fraction

    def _value_loss(self, batch: PPORolloutBatch) -> Tensor:
        prediction = self.value(
            batch.observations,
            sample_weights=True,
        )
        if self.config.value_clip_ratio is None:
            return 0.5 * torch.nn.functional.mse_loss(
                prediction,
                batch.returns,
            )
        clipped = batch.old_values + (prediction - batch.old_values).clamp(
            -self.config.value_clip_ratio,
            self.config.value_clip_ratio,
        )
        plain_loss = (prediction - batch.returns).square()
        clipped_loss = (clipped - batch.returns).square()
        return (
            0.5
            * torch.maximum(
                plain_loss,
                clipped_loss,
            ).mean()
        )

    def update(self, rollout: RolloutBuffer) -> dict[str, float]:
        rollout.compute_gae(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize_advantages=self.config.normalize_advantages,
        )
        generator = torch.Generator(device=self.device).manual_seed(self.update_count + 246_810)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "actor_ucl_penalty": 0.0,
            "value_ucl_penalty": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        minibatches = 0
        stop_early = False

        for _ in range(self.config.update_epochs):
            for batch in rollout.batches(
                self.config.minibatch_size,
                generator=generator,
            ):
                (
                    actor_base,
                    policy_loss,
                    approx_kl,
                    clip_fraction,
                ) = self._actor_loss(batch)
                actor_reg = ucl_regularization(
                    self.actor.bayesian_layers,
                    self.actor_snapshot,
                    minibatch_size=int(batch.observations.shape[0]),
                    config=self.regularization_config,
                    saved_task=self.boundary_count > 0,
                )
                actor_total = actor_base + actor_reg["total"]
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_total.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.actor_optimizer.step()

                value_loss = self._value_loss(batch)
                value_reg = ucl_regularization(
                    self.value.bayesian_layers,
                    self.value_snapshot,
                    minibatch_size=int(batch.observations.shape[0]),
                    config=self.regularization_config,
                    saved_task=self.boundary_count > 0,
                )
                value_total = self.config.value_coef * value_loss + value_reg["total"]
                self.value_optimizer.zero_grad(set_to_none=True)
                value_total.backward()
                nn.utils.clip_grad_norm_(
                    self.value.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.value_optimizer.step()

                totals["policy_loss"] += float(policy_loss.detach().item())
                totals["value_loss"] += float(value_loss.detach().item())
                totals["actor_ucl_penalty"] += float(actor_reg["total"].detach().item())
                totals["value_ucl_penalty"] += float(value_reg["total"].detach().item())
                totals["approx_kl"] += float(approx_kl.detach().item())
                totals["clip_fraction"] += float(clip_fraction.detach().item())
                minibatches += 1

                if (
                    self.config.target_kl is not None
                    and float(approx_kl.detach().item()) > self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        if minibatches == 0:
            raise RuntimeError("UCL-PPO update produced no minibatches")
        self.update_count += 1
        metrics = {name: value / minibatches for name, value in totals.items()}
        metrics["epochs_early_stopped"] = float(stop_early)
        metrics["boundary_count"] = float(self.boundary_count)
        metrics["actor_sigma_mean"] = float(
            torch.stack(
                [
                    torch.nn.functional.softplus(layer.weight_rho).mean()
                    for layer in self.actor.bayesian_layers
                ]
            )
            .mean()
            .detach()
            .item()
        )
        metrics["value_sigma_mean"] = float(
            torch.stack(
                [
                    torch.nn.functional.softplus(layer.weight_rho).mean()
                    for layer in self.value.bayesian_layers
                ]
            )
            .mean()
            .detach()
            .item()
        )
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("nonfinite UCL-PPO metric")
        return metrics

    @staticmethod
    def _serialize_snapshots(
        snapshots: tuple[UCLLayerSnapshot, ...],
    ) -> list[dict[str, Tensor]]:
        return [
            {
                "weight_mu": item.weight_mu.clone(),
                "weight_sigma": item.weight_sigma.clone(),
                "bias_mu": item.bias_mu.clone(),
                "bias_sigma": item.bias_sigma.clone(),
            }
            for item in snapshots
        ]

    @staticmethod
    def _restore_snapshots(
        saved: object,
        *,
        device: torch.device,
        reference: tuple[UCLLayerSnapshot, ...],
        name: str,
    ) -> tuple[UCLLayerSnapshot, ...]:
        if not isinstance(saved, list):
            raise TypeError(f"{name} must be a list")
        if len(saved) != len(reference):
            raise ValueError(f"{name} layer count mismatch")
        required = {
            "weight_mu",
            "weight_sigma",
            "bias_mu",
            "bias_sigma",
        }
        restored: list[UCLLayerSnapshot] = []
        for index, (item, expected) in enumerate(zip(saved, reference, strict=True)):
            if not isinstance(item, dict):
                raise TypeError(f"{name}[{index}] must be a dictionary")
            if set(item) != required:
                raise ValueError(f"{name}[{index}] fields do not match")
            prepared: dict[str, Tensor] = {}
            for field in required:
                value = item[field]
                expected_value = getattr(expected, field)
                if not isinstance(value, Tensor):
                    raise TypeError(f"{name}[{index}].{field} must be a tensor")
                if value.shape != expected_value.shape:
                    raise ValueError(f"{name}[{index}].{field} shape mismatch")
                if value.dtype != expected_value.dtype:
                    raise ValueError(f"{name}[{index}].{field} dtype mismatch")
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"{name}[{index}].{field} contains non-finite values")
                if field.endswith("_sigma") and torch.any(value <= 0).item():
                    raise ValueError(f"{name}[{index}].{field} must be strictly positive")
                prepared[field] = value.detach().to(device=device).clone()
            restored.append(
                UCLLayerSnapshot(
                    weight_mu=prepared["weight_mu"],
                    weight_sigma=prepared["weight_sigma"],
                    bias_mu=prepared["bias_mu"],
                    bias_sigma=prepared["bias_sigma"],
                )
            )
        return tuple(restored)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "ppo_config": asdict(self.config),
            "ucl_config": asdict(self.ucl_config),
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "actor_snapshot": self._serialize_snapshots(self.actor_snapshot),
            "value_snapshot": self._serialize_snapshots(self.value_snapshot),
            "boundary_count": self.boundary_count,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = checkpoint_integer(
            state.get("version"),
            name="UCL-PPO checkpoint version",
        )
        if version != 1:
            raise ValueError("unsupported UCL-PPO checkpoint version")
        if state.get("ppo_config") != asdict(self.config):
            raise ValueError("UCL-PPO PPO configuration mismatch")
        if state.get("ucl_config") != asdict(self.ucl_config):
            raise ValueError("UCL-PPO UCL configuration mismatch")

        def apply(payload: dict[str, Any]) -> None:
            actor_snapshot = self._restore_snapshots(
                payload.get("actor_snapshot"),
                device=self.device,
                reference=self.actor_snapshot,
                name="UCL-PPO actor snapshot",
            )
            value_snapshot = self._restore_snapshots(
                payload.get("value_snapshot"),
                device=self.device,
                reference=self.value_snapshot,
                name="UCL-PPO value snapshot",
            )
            boundary_count = checkpoint_nonnegative_integer(
                payload.get("boundary_count"),
                name="UCL-PPO checkpoint boundary_count",
            )
            update_count = checkpoint_nonnegative_integer(
                payload.get("update_count"),
                name="UCL-PPO checkpoint update_count",
            )
            self.actor.load_state_dict(payload["actor"])
            self.value.load_state_dict(payload["value"])
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.value_optimizer.load_state_dict(payload["value_optimizer"])
            self.actor_snapshot = actor_snapshot
            self.value_snapshot = value_snapshot
            self.boundary_count = boundary_count
            self.update_count = update_count

        transactional_state_load(
            state,
            current_state=self.state_dict,
            apply=apply,
        )
