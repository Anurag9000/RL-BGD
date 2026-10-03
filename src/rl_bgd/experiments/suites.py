"""Paper-oriented experiment suite registry and launcher."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_bgd.artifacts import load_run_directory
from rl_bgd.artifacts.suite import (
    parse_runner_stdout,
    record_completed_suite_run,
    record_failed_suite_run,
)
from rl_bgd.experiments.invoke import resolve_target


@dataclass(frozen=True)
class ExperimentJob:
    job_id: str
    hypothesis_id: str
    target: str
    kwargs: dict[str, Any]
    seeds: tuple[int, ...]
    algorithm: str
    environment: str
    protocol: str
    config_path: str | None
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    comparison_group: str | None = None
    optional_extra: str | None = None
    runtime_class: str = "medium"
    seed_kwarg: str | None = "seed"
    output_kwarg: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class ExperimentSuite:
    name: str
    description: str
    jobs: tuple[ExperimentJob, ...]
    revision: int = 3


def _job(
    job_id: str,
    hypothesis_id: str,
    target: str,
    *,
    kwargs: dict[str, Any],
    seeds: tuple[int, ...],
    algorithm: str,
    environment: str,
    protocol: str,
    config_path: str | None,
    primary_metric: str,
    secondary_metrics: tuple[str, ...],
    comparison_group: str | None = None,
    optional_extra: str | None = None,
    runtime_class: str = "medium",
    seed_kwarg: str | None = "seed",
    output_kwarg: str | None = None,
    notes: str = "",
) -> ExperimentJob:
    return ExperimentJob(
        job_id=job_id,
        hypothesis_id=hypothesis_id,
        target=target,
        kwargs=kwargs,
        seeds=seeds,
        algorithm=algorithm,
        environment=environment,
        protocol=protocol,
        config_path=config_path,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        comparison_group=comparison_group,
        optional_extra=optional_extra,
        runtime_class=runtime_class,
        seed_kwarg=seed_kwarg,
        output_kwarg=output_kwarg,
        notes=notes,
    )


_CANONICAL_SAC_CONTROL_KWARGS: dict[str, Any] = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "initial_alpha": 0.2,
    "evaluation_seed": 20_000,
}

_CANONICAL_PPO_CONTROL_KWARGS: dict[str, Any] = {
    "actor_lr": 3e-4,
    "value_lr": 1e-3,
    "update_epochs": 10,
    "evaluation_seed": 30_000,
}


def _hidden_context_jobs(
    *,
    prefix: str,
    steps: int,
    seeds: tuple[int, ...],
    device: str,
    runtime_class: str,
) -> tuple[ExperimentJob, ...]:
    specs = (
        ("feedforward_adam", "SAC-Adam"),
        ("feedforward_bgd", "SAC-BGD"),
        ("recurrent_adam", "Recurrent-SAC-Adam"),
        ("recurrent_bgd", "Recurrent-SAC-BGD"),
        (
            "recurrent_adaptive_bgd",
            "Recurrent-SAC-Adaptive-BGD",
        ),
    )
    return tuple(
        _job(
            f"{prefix}_{variant}",
            "H",
            ("rl_bgd.runners.hidden_context_sac_comparison:run_hidden_context_sac_variant"),
            kwargs={
                "variant": variant,
                "steps": steps,
                "device": device,
            },
            seeds=seeds,
            algorithm=algorithm,
            environment="hidden_recurring_lqr",
            protocol="strict_task_agnostic",
            config_path="configs/environments/lqr_recurring.yaml",
            primary_metric="final_10_mean_return",
            comparison_group=f"{prefix}_hidden_context",
            secondary_metrics=(
                ("training.mean_episode_return",)
                + (
                    ("training.last_update_metrics.critic1_sigma_mean",)
                    if variant
                    in {
                        "feedforward_bgd",
                        "recurrent_bgd",
                        "recurrent_adaptive_bgd",
                    }
                    else ()
                )
                + (
                    ("training.last_update_metrics.retention_lambda",)
                    if variant == "recurrent_adaptive_bgd"
                    else ()
                )
            ),
            runtime_class=runtime_class,
        )
        for variant, algorithm in specs
    )


def _evidence_temperature_jobs(
    *,
    steps: int,
    seeds: tuple[int, ...],
    runtime_class: str,
) -> tuple[ExperimentJob, ...]:
    return tuple(
        _job(
            f"evidence_temperature_{temperature:g}",
            "GB-T",
            "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
            kwargs={
                **_CANONICAL_SAC_CONTROL_KWARGS,
                "steps": steps,
                "device": "auto",
                "bayesianization": "critic_only",
                "mc_samples": 4,
                "evidence_temperature": temperature,
            },
            seeds=seeds,
            algorithm=f"SAC-BGD-beta-{temperature:g}",
            environment="synthetic_lqr",
            protocol="stationary_generalized_bayes_temperature",
            config_path="configs/sweeps/evidence_temperature_lqr.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.last_update_metrics.critic1_sigma_mean",
                "training.last_update_metrics.critic1_effective_lr_mean",
            ),
            comparison_group="evidence_temperature",
            runtime_class=runtime_class,
        )
        for temperature in (
            0.25,
            0.5,
            1.0,
            2.0,
        )
    )


def _replay_evidence_jobs(
    *,
    steps: int,
    seeds: tuple[int, ...],
    runtime_class: str,
) -> tuple[ExperimentJob, ...]:
    modes = (
        "all_replay",
        "fresh_only_uncertainty",
        "inverse_reuse_weight",
        "normalized_batch_evidence",
    )
    return tuple(
        _job(
            f"replay_evidence_{mode}",
            "F",
            "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
            kwargs={
                **_CANONICAL_SAC_CONTROL_KWARGS,
                "steps": steps,
                "device": "auto",
                "bayesianization": "critic_only",
                "mc_samples": 4,
                "replay_evidence_mode": mode,
            },
            seeds=seeds,
            algorithm=f"SAC-BGD-replay-{mode}",
            environment="synthetic_lqr",
            protocol="stationary_replay_evidence_ablation",
            config_path="configs/algorithms/sac_bgd.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.last_update_metrics.evidence_weight_mean",
                "training.last_update_metrics.evidence_mean_usage_count",
                "training.last_update_metrics.critic1_sigma_mean",
            ),
            comparison_group="replay_evidence",
            runtime_class=runtime_class,
        )
        for mode in modes
    )


def _late_plasticity_jobs(
    *,
    seeds: tuple[int, ...],
    runtime_class: str,
) -> tuple[ExperimentJob, ...]:
    specs = (
        (
            "vanilla",
            1.0,
            "BGD-vanilla-consolidation",
        ),
        (
            "tempered",
            0.97,
            "BGD-tempered-consolidation",
        ),
    )
    return tuple(
        _job(
            f"late_plasticity_{name}",
            "C",
            ("rl_bgd.runners.late_plasticity:run_late_plasticity_quadratic"),
            kwargs={
                "consolidation_retention": retention,
                "dimension": 16,
                "consolidation_steps": 400,
                "adaptation_steps": 32,
                "eta": 0.15,
                "prior_std": 0.5,
                "shift": 1.0,
                "curvature": 1.0,
                "mc_samples": 16,
                "device": "auto",
            },
            seeds=seeds,
            algorithm=algorithm,
            environment="two_phase_quadratic",
            protocol="controlled_late_shift",
            config_path=None,
            primary_metric="post_shift_normalized_auc",
            secondary_metrics=(
                "pre_shift_sigma_mean",
                "pre_shift_effective_lr_mean",
                "first_step_mean_movement",
                "post_shift_final_loss",
                "recovery_fraction",
            ),
            runtime_class=runtime_class,
            comparison_group="late_plasticity",
            notes=(
                "Tempering is applied only during pre-shift consolidation; "
                "post-shift retention is fixed to 1.0 in both conditions."
            ),
        )
        for name, retention, algorithm in specs
    )


def _mc_sample_jobs(
    *,
    steps: int,
    seeds: tuple[int, ...],
    runtime_class: str,
) -> tuple[ExperimentJob, ...]:
    return tuple(
        _job(
            f"mc_samples_k{mc_samples}",
            "MC-K",
            "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
            kwargs={
                **_CANONICAL_SAC_CONTROL_KWARGS,
                "steps": steps,
                "device": "auto",
                "bayesianization": "critic_only",
                "mc_samples": mc_samples,
            },
            seeds=seeds,
            algorithm=f"SAC-BGD-K{mc_samples}",
            environment="synthetic_lqr",
            protocol="stationary_mc_sample_ablation",
            config_path="configs/algorithms/sac_bgd.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.last_update_metrics.critic1_sigma_mean",
                "training.last_update_metrics.critic1_effective_lr_mean",
            ),
            comparison_group="mc_samples",
            runtime_class=runtime_class,
        )
        for mc_samples in (
            1,
            2,
            4,
            8,
        )
    )


def _fixed_tempering_jobs(
    *,
    steps: int,
    seeds: tuple[int, ...],
    runtime_class: str,
) -> tuple[ExperimentJob, ...]:
    retentions = (
        1.0,
        0.999,
        0.99,
        0.95,
    )
    return tuple(
        _job(
            f"fixed_tempering_{str(retention).replace('.', 'p')}",
            "D",
            "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
            kwargs={
                **_CANONICAL_SAC_CONTROL_KWARGS,
                "steps": steps,
                "device": "auto",
                "bayesianization": "critic_only",
                "mc_samples": 4,
                "temper_retention": retention,
            },
            seeds=seeds,
            algorithm=f"SAC-BGD-retention-{retention}",
            environment="synthetic_lqr",
            protocol="stationary_fixed_tempering_ablation",
            config_path="configs/algorithms/sac_bgd.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.last_update_metrics.critic1_sigma_mean",
                "training.last_update_metrics.critic1_effective_lr_mean",
            ),
            comparison_group="fixed_tempering",
            runtime_class=runtime_class,
        )
        for retention in retentions
    )


def _recurring_retention_policy_jobs(
    *,
    total_steps: int,
    phase_steps: int,
    seeds: tuple[int, ...],
    runtime_class: str,
) -> tuple[ExperimentJob, ...]:
    specs = (
        (
            "vanilla",
            "none",
            1.0,
            "SAC-BGD-Vanilla-Retention",
        ),
        (
            "fixed_0p97",
            "none",
            0.97,
            "SAC-BGD-Fixed-Retention-0.97",
        ),
        (
            "adaptive_td",
            "td",
            1.0,
            "SAC-Adaptive-BGD-TD",
        ),
        (
            "adaptive_ensemble",
            "ensemble",
            1.0,
            "SAC-Adaptive-BGD-Ensemble",
        ),
        (
            "adaptive_predictive",
            "predictive",
            1.0,
            "SAC-Adaptive-BGD-Predictive",
        ),
    )
    return tuple(
        _job(
            f"retention_policy_{name}",
            "D-E",
            "rl_bgd.runners.adaptive_bgd_lqr_stream:run_adaptive_bgd_lqr_stream",
            kwargs={
                "total_steps": total_steps,
                "phase_steps": phase_steps,
                "device": "auto",
                "surprise_source": source,
                "fixed_retention": fixed_retention,
            },
            seeds=seeds,
            algorithm=algorithm,
            environment="recurring_lqr",
            protocol="strict_task_agnostic_retention_policy",
            config_path="configs/environments/lqr_recurring.yaml",
            primary_metric="training.final_10_mean_return",
            secondary_metrics=(
                "training.mean_episode_return",
                "training.last_update_metrics.critic1_sigma_mean",
                "training.last_update_metrics.critic1_effective_lr_mean",
                "change_detection.f1",
            ),
            comparison_group="recurring_retention_policy",
            runtime_class=runtime_class,
            notes=(
                "Matched recurring-LQR retention-policy comparison; "
                "all arms share stream, budget, seed set, model, and optimizer."
            ),
        )
        for (
            name,
            source,
            fixed_retention,
            algorithm,
        ) in specs
    )


def _regularized_baseline_jobs(
    *,
    steps: int,
    phase_steps: int,
    horizon: int,
    seeds: tuple[int, ...],
    runtime_class: str,
) -> tuple[ExperimentJob, ...]:
    specs = (
        ("ewc", "SAC-EWC"),
        ("online_ewc", "SAC-Online-EWC"),
        ("si", "SAC-SI"),
        ("mas", "SAC-MAS"),
    )
    return tuple(
        _job(
            f"baseline_{method}",
            "B",
            ("rl_bgd.runners.regularized_sac_continual_lqr:run_regularized_sac_recurring_lqr"),
            kwargs={
                "method": method,
                "steps": steps,
                "device": "auto",
                "consolidation_interval_updates": 8,
                "phase_steps": phase_steps,
                "horizon": horizon,
            },
            seeds=seeds,
            algorithm=algorithm,
            environment="recurring_lqr_matched_v1",
            protocol="task_agnostic_fixed_update",
            config_path=None,
            primary_metric="final_10_mean_return",
            secondary_metrics=(
                "consolidation_count",
                "training.mean_episode_return",
            ),
            runtime_class=runtime_class,
            comparison_group="baseline_sac_regularization",
            notes=(
                "Consolidation is driven by optimizer-update count; "
                "no true task boundary is exposed."
            ),
        )
        for method, algorithm in specs
    )


SMOKE = ExperimentSuite(
    name="smoke",
    description="Dependency-light stationary and hidden-context execution gates.",
    jobs=(
        _job(
            "sac_adam_lqr",
            "A",
            "rl_bgd.runners.sac_lqr:run_sac_lqr",
            kwargs={
                **_CANONICAL_SAC_CONTROL_KWARGS,
                "steps": 96,
                "device": "cpu",
            },
            seeds=(0,),
            algorithm="SAC-Adam",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/environments/lqr_smoke.yaml",
            primary_metric="post_return",
            secondary_metrics=("improvement",),
            comparison_group="smoke_sac",
            runtime_class="smoke",
        ),
        _job(
            "sac_bgd_lqr",
            "A",
            "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
            kwargs={
                **_CANONICAL_SAC_CONTROL_KWARGS,
                "steps": 96,
                "device": "cpu",
                "bayesianization": "critic_only",
                "mc_samples": 4,
            },
            seeds=(0,),
            algorithm="SAC-BGD",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/algorithms/sac_bgd.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.last_update_metrics.critic1_sigma_mean",
            ),
            comparison_group="smoke_sac",
            runtime_class="smoke",
        ),
        _job(
            "ppo_adam_lqr",
            "A",
            "rl_bgd.runners.ppo_lqr:run_ppo_lqr",
            kwargs={
                **_CANONICAL_PPO_CONTROL_KWARGS,
                "steps": 128,
                "device": "cpu",
            },
            seeds=(0,),
            algorithm="PPO-Adam",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/algorithms/ppo_adam.yaml",
            primary_metric="post_return",
            secondary_metrics=("improvement",),
            comparison_group="smoke_ppo",
            runtime_class="smoke",
        ),
    ),
)


_STATIONARY_SAC_COMMON: dict[str, Any] = {
    "steps": 800,
    "device": "auto",
    "horizon": 30,
    "hidden_dims": (32, 32),
    "gamma": 0.99,
    "tau": 0.005,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "alpha_lr": 1e-3,
    "initial_alpha": 0.2,
    "automatic_entropy_tuning": True,
    "target_entropy": None,
    "gradient_clip_norm": None,
    "random_steps": 64,
    "batch_size": 64,
    "replay_capacity": 2_000,
    "evaluation_episodes": 10,
    "evaluation_seed": 20_000,
}

_STATIONARY_PPO_COMMON: dict[str, Any] = {
    "steps": 800,
    "device": "auto",
    "horizon": 30,
    "hidden_dims": (32, 32),
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_clip_ratio": 0.2,
    "actor_lr": 1e-3,
    "value_lr": 1e-3,
    "entropy_coef": 0.0,
    "value_coef": 0.5,
    "update_epochs": 4,
    "minibatch_size": 64,
    "gradient_clip_norm": 0.5,
    "normalize_advantages": True,
    "target_kl": None,
    "rollout_steps": 128,
    "evaluation_episodes": 5,
    "evaluation_seed": 30_000,
}


STATIONARY_CORE = ExperimentSuite(
    name="stationary_core",
    description=("Five-seed stationary SAC/PPO Adam-versus-BGD learning confirmation."),
    jobs=(
        _job(
            "stationary_sac_adam",
            "A",
            "rl_bgd.runners.sac_lqr:run_sac_lqr",
            kwargs={
                **_CANONICAL_SAC_CONTROL_KWARGS,
                "steps": 800,
                "device": "auto",
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-Adam",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/algorithms/sac_adam.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.mean_episode_return",
            ),
            comparison_group="stationary_sac",
            runtime_class="medium",
        ),
        _job(
            "stationary_sac_bgd",
            "A",
            "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
            kwargs={
                **_CANONICAL_SAC_CONTROL_KWARGS,
                "steps": 800,
                "device": "auto",
                "bayesianization": "critic_only",
                "mc_samples": 4,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-BGD",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/algorithms/sac_bgd.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.last_update_metrics.critic1_sigma_mean",
            ),
            comparison_group="stationary_sac",
            runtime_class="medium",
        ),
        _job(
            "stationary_ppo_adam",
            "A",
            "rl_bgd.runners.ppo_lqr:run_ppo_lqr",
            kwargs={
                **_CANONICAL_PPO_CONTROL_KWARGS,
                "steps": 800,
                "device": "auto",
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="PPO-Adam",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/algorithms/ppo_adam.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.mean_episode_return",
            ),
            comparison_group="stationary_ppo",
            runtime_class="medium",
        ),
        _job(
            "stationary_ppo_bgd",
            "A",
            "rl_bgd.runners.bgd_ppo_lqr:run_bgd_ppo_lqr",
            kwargs={
                **_CANONICAL_PPO_CONTROL_KWARGS,
                "steps": 800,
                "device": "auto",
                "bayesianization": "actor_and_value",
                "mc_samples": 4,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="PPO-BGD",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/algorithms/ppo_bgd.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.last_update_metrics.actor_sigma_mean",
                "training.last_update_metrics.value_sigma_mean",
            ),
            comparison_group="stationary_ppo",
            runtime_class="medium",
        ),
    ),
)


DEV = ExperimentSuite(
    name="dev",
    description="Bounded multi-method development comparison before expensive benchmarks.",
    jobs=(
        *_hidden_context_jobs(
            prefix="hidden_context",
            steps=96,
            seeds=(0, 1),
            device="cpu",
            runtime_class="dev",
        ),
        _job(
            "regularized_ewc",
            "B",
            "rl_bgd.runners.regularized_sac_continual_lqr:run_regularized_sac_recurring_lqr",
            kwargs={
                "method": "ewc",
                "steps": 128,
                "device": "cpu",
                "consolidation_interval_updates": 8,
            },
            seeds=(0, 1),
            algorithm="SAC-EWC",
            environment="recurring_lqr_matched_v1",
            protocol="task_agnostic_fixed_update",
            config_path=None,
            primary_metric="final_10_mean_return",
            secondary_metrics=("consolidation_count",),
            runtime_class="dev",
        ),
        _job(
            "ucl_oracle",
            "B",
            "rl_bgd.runners.ucl_ppo_lqr:run_ucl_ppo_recurring_lqr",
            kwargs={
                "phase_steps": 64,
                "phases": 3,
                "device": "cpu",
            },
            seeds=(0, 1),
            algorithm="PPO-UCL",
            environment="recurring_lqr_matched_v1",
            protocol="oracle_boundary",
            config_path=None,
            primary_metric="final_phase_return",
            secondary_metrics=(),
            runtime_class="dev",
        ),
    ),
)


BASELINE_CORE = ExperimentSuite(
    name="baseline_core",
    description=("Five-seed external continual-learning baseline confirmation on recurring LQR."),
    jobs=(
        _job(
            "baseline_sac_adam_control",
            "B",
            ("rl_bgd.runners.regularized_sac_continual_lqr:run_sac_recurring_lqr_control"),
            kwargs={
                "steps": 600,
                "device": "auto",
                "phase_steps": 120,
                "horizon": 32,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-Adam-Control",
            environment="recurring_lqr_matched_v1",
            protocol="task_agnostic_fixed_update",
            config_path=None,
            primary_metric="final_10_mean_return",
            secondary_metrics=("training.mean_episode_return",),
            comparison_group="baseline_sac_regularization",
            runtime_class="medium",
            notes=(
                "Unregularized SAC uses the same recurring-LQR profile, "
                "network width, optimizer settings, replay budget, and seeds "
                "as the parameter-regularization baselines."
            ),
        ),
        *_regularized_baseline_jobs(
            steps=600,
            phase_steps=120,
            horizon=32,
            seeds=(0, 1, 2, 3, 4),
            runtime_class="medium",
        ),
        _job(
            "baseline_ppo_adam_oracle_control",
            "UCL",
            ("rl_bgd.runners.ucl_ppo_lqr:run_adam_ppo_oracle_recurring_lqr_control"),
            kwargs={
                "phase_steps": 120,
                "phases": 5,
                "device": "auto",
                "horizon": 32,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="PPO-Adam-Oracle-Control",
            environment="recurring_lqr_matched_v1",
            protocol="oracle_boundary",
            config_path=None,
            primary_metric="final_phase_return",
            secondary_metrics=(),
            comparison_group="baseline_ppo_ucl",
            runtime_class="medium",
            notes=(
                "Adam PPO uses the same phase schedule, model widths, PPO "
                "hyperparameters, and seeds as UCL but no Bayesian layers "
                "or UCL regularization."
            ),
        ),
        _job(
            "baseline_ucl_oracle",
            "UCL",
            "rl_bgd.runners.ucl_ppo_lqr:run_ucl_ppo_recurring_lqr",
            kwargs={
                "phase_steps": 120,
                "phases": 5,
                "device": "auto",
                "horizon": 32,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="PPO-UCL",
            environment="recurring_lqr_matched_v1",
            protocol="oracle_boundary",
            config_path=None,
            primary_metric="final_phase_return",
            secondary_metrics=(),
            comparison_group="baseline_ppo_ucl",
            runtime_class="medium",
            notes=("UCL is an oracle-boundary comparator and is not labelled task-agnostic."),
        ),
    ),
)


CARL_CORE = ExperimentSuite(
    name="carl_core",
    description="Strict hidden-context CARL Pendulum nonstationarity study.",
    jobs=tuple(
        _job(
            f"carl_{mode}_{optimizer}",
            "E" if optimizer == "adaptive_bgd" else "B",
            "rl_bgd.runners.carl_pendulum_sac:run_carl_pendulum_sac",
            kwargs={
                "mode": mode,
                "optimizer": optimizer,
                "steps": 150_000,
                "phase_steps": 50_000,
                "device": "auto",
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm=f"SAC-{optimizer}",
            environment="CARLPendulum",
            protocol="strict_task_agnostic",
            config_path=f"configs/environments/carl_pendulum_{mode}.yaml",
            primary_metric="final_10_mean_return",
            secondary_metrics=(
                ("training.last_update_metrics.critic1_sigma_mean",)
                if optimizer == "bgd"
                else (
                    (
                        "training.last_update_metrics.critic1_sigma_mean",
                        "training.last_update_metrics.retention_lambda",
                    )
                    if optimizer == "adaptive_bgd"
                    else ()
                )
            ),
            comparison_group=f"carl_{mode}",
            optional_extra="carl",
            runtime_class="large",
        )
        for mode in ("abrupt", "smooth", "recurring")
        for optimizer in ("adam", "bgd", "adaptive_bgd")
    ),
)

CW10_CORE = ExperimentSuite(
    name="cw10_core",
    description="Canonical and strict task-agnostic CW10 comparison.",
    jobs=(
        _job(
            "cw10_canonical_adam",
            "CW-CAN10",
            "rl_bgd.runners.canonical_continual_world_sac:run_canonical_continual_world_sac",
            kwargs={
                "benchmark": "CW10",
                "steps_per_task": 1_000_000,
                "device": "auto",
                "evaluation_episodes": 5,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-Adam",
            environment="CW10",
            protocol="canonical_task_aware",
            config_path="configs/benchmarks/continual_world_canonical_cw10.yaml",
            primary_metric="final_average",
            secondary_metrics=("forgetting", "bwt", "success_rate"),
            optional_extra="continual-world",
            runtime_class="very_large",
        ),
        _job(
            "cw10_ta_adam",
            "CW-TA10",
            "rl_bgd.runners.continual_world_sac:run_ta_continual_world_sac",
            kwargs={
                "benchmark": "CW10",
                "optimizer": "adam",
                "steps_per_task": 1_000_000,
                "device": "auto",
                "evaluation_episodes": 5,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-Adam",
            environment="CW10",
            protocol="strict_task_agnostic",
            config_path="configs/benchmarks/continual_world_ta_cw10.yaml",
            primary_metric="final_average",
            comparison_group="cw10_task_agnostic_feedforward",
            secondary_metrics=("forgetting", "bwt", "success_rate"),
            optional_extra="continual-world",
            runtime_class="very_large",
        ),
        _job(
            "cw10_ta_bgd",
            "CW-TA10",
            "rl_bgd.runners.continual_world_sac:run_ta_continual_world_sac",
            kwargs={
                "benchmark": "CW10",
                "optimizer": "bgd",
                "steps_per_task": 1_000_000,
                "device": "auto",
                "evaluation_episodes": 5,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-BGD",
            environment="CW10",
            protocol="strict_task_agnostic",
            config_path="configs/benchmarks/continual_world_ta_cw10.yaml",
            primary_metric="final_average",
            comparison_group="cw10_task_agnostic_feedforward",
            secondary_metrics=(
                "forgetting",
                "bwt",
                "success_rate",
                "training.last_update_metrics.critic1_sigma_mean",
            ),
            optional_extra="continual-world",
            runtime_class="very_large",
        ),
        *tuple(
            _job(
                f"cw10_ta_{method}",
                "B",
                "rl_bgd.runners.continual_world_sac:run_ta_continual_world_sac",
                kwargs={
                    "benchmark": "CW10",
                    "optimizer": method,
                    "steps_per_task": 1_000_000,
                    "device": "auto",
                    "evaluation_episodes": 5,
                    "consolidation_interval_updates": 50_000,
                    "regularization_strength": 0.1,
                    "regularization_target": "actor_and_critic",
                    "importance_samples": 4,
                },
                seeds=(0, 1, 2, 3, 4),
                algorithm={
                    "ewc": "SAC-EWC",
                    "online_ewc": "SAC-Online-EWC",
                    "si": "SAC-SI",
                    "mas": "SAC-MAS",
                }[method],
                environment="CW10",
                protocol="strict_task_agnostic",
                config_path="configs/benchmarks/continual_world_ta_cw10.yaml",
                primary_metric="final_average",
                secondary_metrics=(
                    "forgetting",
                    "bwt",
                    "success_rate",
                    "training.last_update_metrics.consolidation_count",
                    "training.last_update_metrics.actor_regularization_penalty",
                    "training.last_update_metrics.critic_regularization_penalty",
                ),
                comparison_group="cw10_task_agnostic_feedforward",
                optional_extra="continual-world",
                runtime_class="very_large",
                notes=(
                    "Consolidation is driven only by optimizer-update count; "
                    "no ground-truth task boundary or task identity is exposed."
                ),
            )
            for method in (
                "ewc",
                "online_ewc",
                "si",
                "mas",
            )
        ),
        *tuple(
            _job(
                f"cw10_recurrent_{optimizer}",
                "H",
                "rl_bgd.runners.recurrent_continual_world:run_recurrent_ta_continual_world_sac",
                kwargs={
                    "benchmark": "CW10",
                    "optimizer": optimizer,
                    "steps_per_task": 1_000_000,
                    "device": "auto",
                    "evaluation_episodes": 10,
                },
                seeds=(0, 1, 2, 3, 4),
                algorithm={
                    "adam": "Recurrent-SAC-Adam",
                    "bgd": "Recurrent-SAC-BGD",
                    "adaptive_bgd": "Recurrent-SAC-Adaptive-BGD",
                }[optimizer],
                environment="CW10",
                protocol="3RL-style_task_agnostic",
                config_path="configs/benchmarks/three_rl_style_cw10.yaml",
                primary_metric="final_average",
                secondary_metrics=(
                    (
                        "forgetting",
                        "bwt",
                        "success_rate",
                    )
                    + (
                        ("training.last_update_metrics.critic1_sigma_mean",)
                        if optimizer
                        in {
                            "bgd",
                            "adaptive_bgd",
                        }
                        else ()
                    )
                    + (
                        ("training.last_update_metrics.retention_lambda",)
                        if optimizer == "adaptive_bgd"
                        else ()
                    )
                ),
                comparison_group="cw10_recurrent",
                optional_extra="continual-world",
                runtime_class="very_large",
            )
            for optimizer in (
                "adam",
                "bgd",
                "adaptive_bgd",
            )
        ),
    ),
)

CW20_FINAL = ExperimentSuite(
    name="cw20_final",
    description="Final strict and recurrent CW20 confirmation suite.",
    jobs=(
        _job(
            "cw20_canonical_adam",
            "CW-CAN20",
            "rl_bgd.runners.canonical_continual_world_sac:run_canonical_continual_world_sac",
            kwargs={
                "benchmark": "CW20",
                "steps_per_task": 1_000_000,
                "device": "auto",
                "evaluation_episodes": 5,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-Adam",
            environment="CW20",
            protocol="canonical_task_aware",
            config_path="configs/benchmarks/continual_world_canonical_cw20.yaml",
            primary_metric="final_average",
            secondary_metrics=(
                "forgetting",
                "bwt",
                "success_rate",
            ),
            optional_extra="continual-world",
            runtime_class="very_large",
        ),
        _job(
            "cw20_ta_adam",
            "CW-TA20",
            "rl_bgd.runners.continual_world_sac:run_ta_continual_world_sac",
            kwargs={
                "benchmark": "CW20",
                "optimizer": "adam",
                "steps_per_task": 1_000_000,
                "device": "auto",
                "evaluation_episodes": 5,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-Adam",
            environment="CW20",
            protocol="strict_task_agnostic",
            config_path="configs/benchmarks/continual_world_ta_cw20.yaml",
            primary_metric="final_average",
            comparison_group="cw20_task_agnostic_feedforward",
            secondary_metrics=(
                "forgetting",
                "bwt",
                "success_rate",
                "recurrence_summary.reference_success",
                "recurrence_summary.zero_shot_success",
                "recurrence_summary.recovered_success",
                "recurrence_summary.pre_revisit_change",
                "recurrence_summary.relearning_gain",
            ),
            optional_extra="continual-world",
            runtime_class="very_large",
        ),
        _job(
            "cw20_ta_bgd",
            "CW-TA20",
            "rl_bgd.runners.continual_world_sac:run_ta_continual_world_sac",
            kwargs={
                "benchmark": "CW20",
                "optimizer": "bgd",
                "steps_per_task": 1_000_000,
                "device": "auto",
                "evaluation_episodes": 5,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-BGD",
            environment="CW20",
            protocol="strict_task_agnostic",
            config_path="configs/benchmarks/continual_world_ta_cw20.yaml",
            primary_metric="final_average",
            comparison_group="cw20_task_agnostic_feedforward",
            secondary_metrics=(
                "forgetting",
                "bwt",
                "success_rate",
                "training.last_update_metrics.critic1_sigma_mean",
                "recurrence_summary.reference_success",
                "recurrence_summary.zero_shot_success",
                "recurrence_summary.recovered_success",
                "recurrence_summary.pre_revisit_change",
                "recurrence_summary.relearning_gain",
            ),
            optional_extra="continual-world",
            runtime_class="very_large",
        ),
        *tuple(
            _job(
                f"cw20_ta_{method}",
                "B",
                "rl_bgd.runners.continual_world_sac:run_ta_continual_world_sac",
                kwargs={
                    "benchmark": "CW20",
                    "optimizer": method,
                    "steps_per_task": 1_000_000,
                    "device": "auto",
                    "evaluation_episodes": 5,
                    "consolidation_interval_updates": 50_000,
                    "regularization_strength": 0.1,
                    "regularization_target": "actor_and_critic",
                    "importance_samples": 4,
                },
                seeds=(0, 1, 2, 3, 4),
                algorithm={
                    "ewc": "SAC-EWC",
                    "online_ewc": "SAC-Online-EWC",
                    "si": "SAC-SI",
                    "mas": "SAC-MAS",
                }[method],
                environment="CW20",
                protocol="strict_task_agnostic",
                config_path="configs/benchmarks/continual_world_ta_cw20.yaml",
                primary_metric="final_average",
                secondary_metrics=(
                    "forgetting",
                    "bwt",
                    "success_rate",
                    "training.last_update_metrics.consolidation_count",
                    "training.last_update_metrics.actor_regularization_penalty",
                    "training.last_update_metrics.critic_regularization_penalty",
                    "recurrence_summary.reference_success",
                    "recurrence_summary.zero_shot_success",
                    "recurrence_summary.recovered_success",
                    "recurrence_summary.pre_revisit_change",
                    "recurrence_summary.relearning_gain",
                ),
                comparison_group="cw20_task_agnostic_feedforward",
                optional_extra="continual-world",
                runtime_class="very_large",
                notes=(
                    "Consolidation is driven only by optimizer-update count; "
                    "no ground-truth task boundary or task identity is exposed."
                ),
            )
            for method in (
                "ewc",
                "online_ewc",
                "si",
                "mas",
            )
        ),
        _job(
            "cw20_recurrent_adam",
            "H",
            "rl_bgd.runners.recurrent_continual_world:run_recurrent_ta_continual_world_sac",
            kwargs={
                "benchmark": "CW20",
                "optimizer": "adam",
                "steps_per_task": 1_000_000,
                "device": "auto",
                "evaluation_episodes": 10,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="Recurrent-SAC-Adam",
            environment="CW20",
            protocol="3RL-style_task_agnostic",
            config_path="configs/benchmarks/three_rl_style_cw20.yaml",
            primary_metric="final_average",
            comparison_group="cw20_recurrent",
            secondary_metrics=(
                "forgetting",
                "bwt",
                "success_rate",
                "recurrence_summary.reference_success",
                "recurrence_summary.zero_shot_success",
                "recurrence_summary.recovered_success",
                "recurrence_summary.pre_revisit_change",
                "recurrence_summary.relearning_gain",
            ),
            optional_extra="continual-world",
            runtime_class="very_large",
        ),
        _job(
            "cw20_recurrent_bgd",
            "H",
            "rl_bgd.runners.recurrent_continual_world:run_recurrent_ta_continual_world_sac",
            kwargs={
                "benchmark": "CW20",
                "optimizer": "bgd",
                "steps_per_task": 1_000_000,
                "device": "auto",
                "evaluation_episodes": 10,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="Recurrent-SAC-BGD",
            environment="CW20",
            protocol="3RL-style_task_agnostic",
            config_path="configs/benchmarks/three_rl_style_cw20.yaml",
            primary_metric="final_average",
            comparison_group="cw20_recurrent",
            secondary_metrics=(
                "forgetting",
                "bwt",
                "success_rate",
                "training.last_update_metrics.critic1_sigma_mean",
                "recurrence_summary.reference_success",
                "recurrence_summary.zero_shot_success",
                "recurrence_summary.recovered_success",
                "recurrence_summary.pre_revisit_change",
                "recurrence_summary.relearning_gain",
            ),
            optional_extra="continual-world",
            runtime_class="very_large",
        ),
        _job(
            "cw20_recurrent_adaptive",
            "H",
            "rl_bgd.runners.recurrent_continual_world:run_recurrent_ta_continual_world_sac",
            kwargs={
                "benchmark": "CW20",
                "optimizer": "adaptive_bgd",
                "steps_per_task": 1_000_000,
                "device": "auto",
                "evaluation_episodes": 10,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="Recurrent-SAC-Adaptive-BGD",
            environment="CW20",
            protocol="3RL-style_task_agnostic",
            config_path="configs/benchmarks/three_rl_style_cw20.yaml",
            primary_metric="final_average",
            comparison_group="cw20_recurrent",
            secondary_metrics=(
                "forgetting",
                "bwt",
                "success_rate",
                "training.last_update_metrics.critic1_sigma_mean",
                "training.last_update_metrics.retention_lambda",
                "recurrence_summary.reference_success",
                "recurrence_summary.zero_shot_success",
                "recurrence_summary.recovered_success",
                "recurrence_summary.pre_revisit_change",
                "recurrence_summary.relearning_gain",
            ),
            optional_extra="continual-world",
            runtime_class="very_large",
        ),
    ),
)

TASK_AGNOSTIC_FINAL = ExperimentSuite(
    name="task_agnostic_final",
    description="Cross-environment strict task-agnostic confirmation suite.",
    jobs=(
        *_hidden_context_jobs(
            prefix="ta_hidden",
            steps=512,
            seeds=(0, 1, 2, 3, 4),
            device="auto",
            runtime_class="medium",
        ),
        *CARL_CORE.jobs,
        *tuple(job for job in CW10_CORE.jobs if "canonical" not in job.job_id),
        *tuple(job for job in CW20_FINAL.jobs if "canonical" not in job.job_id),
    ),
)


ABLATION_CORE = ExperimentSuite(
    name="ablation_core",
    description="Bayesianization and generalized-Bayes control ablations.",
    jobs=tuple(
        _job(
            f"bgd_sac_{mode}",
            "G",
            "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
            kwargs={
                **_CANONICAL_SAC_CONTROL_KWARGS,
                "steps": 600,
                "device": "auto",
                "bayesianization": mode,
                "evidence_temperature": 1.0,
                "mc_samples": 4,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm=f"SAC-BGD-{mode}",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/algorithms/sac_bgd.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                ("improvement",)
                + (
                    (
                        "training.last_update_metrics.critic1_sigma_mean",
                        "training.last_update_metrics.critic1_effective_lr_mean",
                    )
                    if mode
                    in {
                        "critic_only",
                        "actor_and_critic",
                    }
                    else ()
                )
                + (
                    (
                        "training.last_update_metrics.actor_sigma_mean",
                        "training.last_update_metrics.actor_effective_lr_mean",
                    )
                    if mode
                    in {
                        "actor_only",
                        "actor_and_critic",
                    }
                    else ()
                )
            ),
            comparison_group="bayesianization",
            runtime_class="medium",
        )
        for mode in (
            "critic_only",
            "actor_only",
            "actor_and_critic",
        )
    )
    + _replay_evidence_jobs(
        steps=600,
        seeds=(0, 1, 2, 3, 4),
        runtime_class="medium",
    )
    + _late_plasticity_jobs(
        seeds=(0, 1, 2, 3, 4),
        runtime_class="medium",
    )
    + _fixed_tempering_jobs(
        steps=1200,
        seeds=(0, 1, 2, 3, 4),
        runtime_class="medium",
    )
    + _evidence_temperature_jobs(
        steps=600,
        seeds=(0, 1, 2, 3, 4),
        runtime_class="medium",
    )
    + _mc_sample_jobs(
        steps=600,
        seeds=(0, 1, 2, 3, 4),
        runtime_class="medium",
    ),
)


UNCERTAINTY_ANALYSIS = ExperimentSuite(
    name="uncertainty_analysis",
    description=(
        "Posterior uncertainty, matched fixed/adaptive retention and "
        "surprise-source ablations, plus evidence-temperature diagnostics."
    ),
    jobs=(
        _job(
            "adaptive_timeline",
            "E",
            "rl_bgd.runners.adaptive_bgd_lqr_stream:run_adaptive_bgd_lqr_stream",
            kwargs={
                "total_steps": 900,
                "phase_steps": 300,
                "device": "auto",
                "surprise_source": "td",
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-Adaptive-BGD-TD",
            environment="recurring_lqr",
            protocol="strict_task_agnostic",
            config_path="configs/environments/lqr_recurring.yaml",
            primary_metric="change_detection.f1",
            secondary_metrics=(
                "change_detection.precision",
                "change_detection.recall",
                "change_detection.false_alarms_per_million_steps",
                "surprise_auroc",
                "training.final_10_mean_return",
                "training.last_update_metrics.retention_lambda",
                "training.last_update_metrics.critic1_sigma_mean",
            ),
            comparison_group="surprise_source_detection",
            runtime_class="medium",
        ),
        *tuple(
            _job(
                f"surprise_source_{source}",
                "E",
                "rl_bgd.runners.adaptive_bgd_lqr_stream:run_adaptive_bgd_lqr_stream",
                kwargs={
                    "total_steps": 900,
                    "phase_steps": 300,
                    "device": "auto",
                    "surprise_source": source,
                },
                seeds=(0, 1, 2, 3, 4),
                algorithm=(
                    "SAC-BGD-No-Adaptive"
                    if source == "none"
                    else f"SAC-Adaptive-BGD-{source.title()}"
                ),
                environment="recurring_lqr",
                protocol="strict_task_agnostic",
                config_path="configs/environments/lqr_recurring.yaml",
                primary_metric="change_detection.f1",
                secondary_metrics=(
                    (
                        "change_detection.precision",
                        "change_detection.recall",
                        "change_detection.false_alarms_per_million_steps",
                        "training.final_10_mean_return",
                        "training.last_update_metrics.critic1_sigma_mean",
                    )
                    + (
                        (
                            "surprise_auroc",
                            "training.last_update_metrics.retention_lambda",
                        )
                        if source != "none"
                        else ()
                    )
                    + (
                        ("training.last_update_metrics.predictive_model_loss",)
                        if source == "predictive"
                        else ()
                    )
                ),
                comparison_group="surprise_source_detection",
                runtime_class="medium",
            )
            for source in (
                "none",
                "ensemble",
                "predictive",
            )
        ),
        *_recurring_retention_policy_jobs(
            total_steps=900,
            phase_steps=300,
            seeds=(0, 1, 2, 3, 4),
            runtime_class="medium",
        ),
        *_evidence_temperature_jobs(
            steps=600,
            seeds=(0, 1, 2, 3, 4),
            runtime_class="medium",
        ),
    ),
)


MECHANISM_ANALYSIS = ExperimentSuite(
    name="mechanism_analysis",
    description="Deterministic causal parameter-level mechanism suite.",
    jobs=(
        _job(
            "mechanistic_quadratic",
            "I-J",
            ("rl_bgd.analysis.mechanistic:run_seeded_mechanistic_analysis"),
            kwargs={
                "device": "auto",
            },
            seeds=(150, 151, 152, 153, 154),
            algorithm="BGD",
            environment="anisotropic_quadratic",
            protocol="mechanistic",
            config_path=None,
            primary_metric="movement_sigma_spearman",
            secondary_metrics=(
                "perturbation_precision_spearman",
                "curvature_signal_mean_relative_error",
                "freezing_target_loss.none",
                "freezing_target_loss.freeze_low_sigma",
                "freezing_target_loss.freeze_high_sigma",
            ),
            runtime_class="analysis",
            seed_kwarg="seed",
            output_kwarg="output_dir",
        ),
    ),
)

COMPUTE_ANALYSIS = ExperimentSuite(
    name="compute_analysis",
    description=(
        "Matched 600-step stationary SAC compute/performance comparison "
        "with launcher-recorded wall-clock metadata."
    ),
    jobs=(
        _job(
            "compute_sac_adam",
            "COMPUTE",
            "rl_bgd.runners.sac_lqr:run_sac_lqr",
            kwargs={
                **_CANONICAL_SAC_CONTROL_KWARGS,
                "steps": 600,
                "device": "auto",
            },
            seeds=(0, 1, 2),
            algorithm="SAC-Adam",
            environment="synthetic_lqr",
            protocol="stationary_compute",
            config_path="configs/algorithms/sac_adam.yaml",
            primary_metric="duration_seconds",
            secondary_metrics=(
                "post_return",
                "improvement",
            ),
            comparison_group="compute_bayesianization",
            runtime_class="compute",
        ),
        *tuple(
            _job(
                f"compute_bgd_{mode}",
                "COMPUTE",
                "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
                kwargs={
                    **_CANONICAL_SAC_CONTROL_KWARGS,
                    "steps": 600,
                    "device": "auto",
                    "bayesianization": mode,
                    "mc_samples": 4,
                },
                seeds=(0, 1, 2),
                algorithm=f"SAC-BGD-{mode}",
                environment="synthetic_lqr",
                protocol="stationary_compute",
                config_path="configs/algorithms/sac_bgd.yaml",
                primary_metric="duration_seconds",
                secondary_metrics=(
                    "post_return",
                    "improvement",
                ),
                comparison_group="compute_bayesianization",
                runtime_class="compute",
            )
            for mode in (
                "critic_only",
                "actor_only",
                "actor_and_critic",
            )
        ),
        *tuple(
            _job(
                f"compute_bgd_k{mc_samples}",
                "MC-K",
                "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
                kwargs={
                    **_CANONICAL_SAC_CONTROL_KWARGS,
                    "steps": 600,
                    "device": "auto",
                    "bayesianization": "critic_only",
                    "mc_samples": mc_samples,
                },
                seeds=(0, 1, 2),
                algorithm=f"SAC-BGD-K{mc_samples}",
                environment="synthetic_lqr",
                protocol="stationary_compute",
                config_path="configs/algorithms/sac_bgd.yaml",
                primary_metric="duration_seconds",
                secondary_metrics=(
                    "post_return",
                    "improvement",
                ),
                comparison_group="compute_mc_samples",
                runtime_class="compute",
            )
            for mc_samples in (
                1,
                2,
                4,
                8,
            )
        ),
    ),
)


SUITES: dict[str, ExperimentSuite] = {
    suite.name: suite
    for suite in (
        SMOKE,
        STATIONARY_CORE,
        DEV,
        BASELINE_CORE,
        CARL_CORE,
        CW10_CORE,
        CW20_FINAL,
        TASK_AGNOSTIC_FINAL,
        ABLATION_CORE,
        UNCERTAINTY_ANALYSIS,
        MECHANISM_ANALYSIS,
        COMPUTE_ANALYSIS,
    )
}


def _git_head() -> str | None:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA")
    return result.stdout.strip() or None


def _resolved_call_kwargs(
    target: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a runner call including signature defaults for provenance."""

    function = resolve_target(target)
    signature = inspect.signature(function)
    bound = signature.bind(**kwargs)
    bound.apply_defaults()
    resolved = dict(bound.arguments)
    try:
        return json.loads(
            json.dumps(
                resolved,
                sort_keys=True,
            )
        )
    except TypeError as exc:
        raise TypeError(
            f"runner {target} has non-JSON-serializable resolved arguments"
        ) from exc


def _expanded_jobs(
    suite: ExperimentSuite,
    output_root: Path,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for job in suite.jobs:
        for seed in job.seeds:
            kwargs = dict(job.kwargs)
            if job.seed_kwarg is not None:
                kwargs[job.seed_kwarg] = seed
            run_id = f"{suite.name}__{job.job_id}__seed_{seed}"
            run_dir = output_root / suite.name / run_id
            if job.output_kwarg is not None:
                kwargs[job.output_kwarg] = str(run_dir / "artifacts")
            command = [
                sys.executable,
                "-m",
                "rl_bgd.experiments.invoke",
                "--target",
                job.target,
                "--kwargs-json",
                json.dumps(kwargs, sort_keys=True),
            ]
            resolved_call_kwargs = _resolved_call_kwargs(
                job.target,
                kwargs,
            )
            expanded.append(
                {
                    **asdict(job),
                    "suite_revision": suite.revision,
                    "seed": seed,
                    "kwargs": kwargs,
                    "resolved_call_kwargs": resolved_call_kwargs,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "command": command,
                }
            )
    return expanded


def validate_suite_registry() -> None:
    required = {
        "smoke",
        "stationary_core",
        "dev",
        "baseline_core",
        "carl_core",
        "cw10_core",
        "cw20_final",
        "task_agnostic_final",
        "ablation_core",
        "uncertainty_analysis",
        "mechanism_analysis",
        "compute_analysis",
    }
    if set(SUITES) != required:
        raise RuntimeError("paper suite registry is incomplete")

    for suite in SUITES.values():
        if not suite.jobs:
            raise RuntimeError(f"suite {suite.name} has no jobs")

        context_contracts: dict[
            str,
            tuple[
                tuple[int, ...],
                str,
                str,
                str,
            ],
        ] = {}
        for job in suite.jobs:
            if not job.seeds:
                raise RuntimeError(f"job {job.job_id} has no seeds")
            if not job.primary_metric:
                raise RuntimeError(f"job {job.job_id} lacks a primary metric")

            if job.comparison_group is not None and not job.comparison_group.strip():
                raise RuntimeError(f"job {job.job_id} has an empty comparison group")
            context = (
                job.comparison_group
                if job.comparison_group is not None
                else f"__unpaired__:{job.job_id}"
            )
            contract = (
                job.seeds,
                job.primary_metric,
                job.protocol,
                job.environment,
            )
            previous = context_contracts.setdefault(
                context,
                contract,
            )
            if previous != contract:
                raise RuntimeError(
                    "paper-comparison context has inconsistent seed/outcome/"
                    "protocol/benchmark contracts in suite "
                    f"{suite.name}: {context!r} / "
                    f"expected {previous!r}, found {contract!r} at {job.job_id}"
                )

            function = resolve_target(job.target)
            signature = inspect.signature(function)
            kwargs = dict(job.kwargs)
            if job.seed_kwarg is not None:
                kwargs[job.seed_kwarg] = job.seeds[0]
            if job.output_kwarg is not None:
                kwargs[job.output_kwarg] = "artifacts/test"
            signature.bind(**kwargs)


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically publish launcher JSON for crash/concurrent-shard safety."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(
            temporary,
            path,
        )
    finally:
        temporary.unlink(missing_ok=True)


def materialize_suite(
    suite_name: str,
    output_root: str | Path,
) -> dict[str, Any]:
    validate_suite_registry()
    if suite_name not in SUITES:
        raise KeyError(f"unknown experiment suite: {suite_name}")
    suite = SUITES[suite_name]
    root = Path(output_root)
    suite_dir = root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "suite": suite.name,
        "suite_revision": suite.revision,
        "description": suite.description,
        "git_commit": _git_head(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "jobs": _expanded_jobs(suite, root),
    }
    path = suite_dir / "suite_manifest.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(
            temporary,
            path,
        )
    finally:
        temporary.unlink(missing_ok=True)
    manifest["manifest_path"] = str(path)
    return manifest


def _select_execution_jobs(
    jobs: list[dict[str, Any]],
    *,
    job_ids: tuple[str, ...] | None,
    seeds: tuple[int, ...] | None,
    run_ids: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    requested_job_ids = set(job_ids or ())
    requested_seeds = set(seeds or ())
    requested_run_ids = set(run_ids or ())

    known_job_ids = {str(job["job_id"]) for job in jobs}
    known_seeds = {int(job["seed"]) for job in jobs}
    known_run_ids = {str(job["run_id"]) for job in jobs}

    unknown_job_ids = requested_job_ids - known_job_ids
    if unknown_job_ids:
        raise KeyError("unknown suite job IDs: " + ", ".join(sorted(unknown_job_ids)))
    unknown_seeds = requested_seeds - known_seeds
    if unknown_seeds:
        raise KeyError(
            "unknown suite seeds: " + ", ".join(str(value) for value in sorted(unknown_seeds))
        )
    unknown_run_ids = requested_run_ids - known_run_ids
    if unknown_run_ids:
        raise KeyError("unknown suite run IDs: " + ", ".join(sorted(unknown_run_ids)))

    selected = [
        job
        for job in jobs
        if (not requested_job_ids or str(job["job_id"]) in requested_job_ids)
        and (not requested_seeds or int(job["seed"]) in requested_seeds)
        and (not requested_run_ids or str(job["run_id"]) in requested_run_ids)
    ]
    if (requested_job_ids or requested_seeds or requested_run_ids) and not selected:
        raise ValueError("suite execution filters select no runs")
    return selected


def _archive_existing_run(
    run_dir: Path,
    *,
    output_root: str | Path,
    suite_name: str,
) -> Path | None:
    """Move a superseded run outside the active results tree before rerunning."""

    if not run_dir.is_dir():
        return None
    try:
        has_contents = (
            next(
                run_dir.iterdir(),
                None,
            )
            is not None
        )
    except OSError:
        has_contents = True
    if not has_contents:
        return None

    prior_commit = "unknown"
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        try:
            payload = json.loads(
                metadata_path.read_text(
                    encoding="utf-8",
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            payload = None
        if isinstance(
            payload,
            dict,
        ):
            raw_commit = payload.get("git_commit")
            if (
                isinstance(
                    raw_commit,
                    str,
                )
                and raw_commit.strip()
            ):
                prior_commit = raw_commit.strip()[:12]

    root = Path(output_root)
    archive_base = root.parent / ((root.name or "results") + "_archives")
    archive_parent = archive_base / suite_name / run_dir.name
    archive_parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = archive_parent / f"{prior_commit}__{timestamp}"
    counter = 1
    while destination.exists():
        destination = archive_parent / (f"{prior_commit}__{timestamp}__{counter}")
        counter += 1
    shutil.move(
        str(run_dir),
        str(destination),
    )
    return destination


def _completed_run_matches_job(
    run_dir: Path,
    *,
    suite_name: str,
    git_commit: str,
    job: dict[str, Any],
) -> bool:
    """Return whether a strict completed run exactly matches this job contract."""

    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(raw, dict):
        return False
    expected = {
        "schema_version": 2,
        "run_id": job["run_id"],
        "job_id": job["job_id"],
        "comparison_group": job["comparison_group"],
        "suite": suite_name,
        "suite_revision": job["suite_revision"],
        "git_commit": git_commit,
        "target": job["target"],
        "kwargs": json.loads(
            json.dumps(
                job["kwargs"],
                sort_keys=True,
            )
        ),
        "resolved_call_kwargs": job["resolved_call_kwargs"],
        "seed": job["seed"],
        "algorithm": job["algorithm"],
        "environment": job["environment"],
        "protocol": job["protocol"],
        "hypothesis_id": job["hypothesis_id"],
        "config_path": job["config_path"],
        "primary_metric": job["primary_metric"],
        "secondary_metrics": list(job["secondary_metrics"]),
        "optional_extra": job["optional_extra"],
        "runtime_class": job["runtime_class"],
        "status": "success",
        "returncode": 0,
        "strict_artifacts": True,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            return False
    try:
        loaded = load_run_directory(
            run_dir,
            require_completed=True,
        )
    except (OSError, TypeError, ValueError):
        return False
    return (
        loaded.manifest.run_id == job["run_id"]
        and loaded.manifest.git_commit == git_commit
        and loaded.manifest.metadata.get("suite_revision") == job["suite_revision"]
    )


def execute_suite(
    suite_name: str,
    output_root: str | Path,
    *,
    continue_on_error: bool = False,
    resume: bool = True,
    job_ids: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    run_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    manifest = materialize_suite(
        suite_name,
        output_root,
    )
    git_commit = manifest.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit:
        raise RuntimeError("suite execution requires a concrete git commit for provenance")

    manifest_jobs = manifest["jobs"]
    if not isinstance(
        manifest_jobs,
        list,
    ):
        raise TypeError("suite manifest jobs must be a list")
    selected_jobs = _select_execution_jobs(
        manifest_jobs,
        job_ids=job_ids,
        seeds=seeds,
        run_ids=run_ids,
    )

    failures: list[str] = []
    skipped: list[str] = []
    executed: list[str] = []
    archived: list[dict[str, str]] = []
    for job in selected_jobs:
        run_dir = Path(job["run_dir"])
        if resume and _completed_run_matches_job(
            run_dir,
            suite_name=suite_name,
            git_commit=git_commit,
            job=job,
        ):
            skipped.append(str(job["run_id"]))
            continue
        archived_path = _archive_existing_run(
            run_dir,
            output_root=output_root,
            suite_name=suite_name,
        )
        if archived_path is not None:
            archived.append(
                {
                    "run_id": str(job["run_id"]),
                    "path": str(archived_path),
                }
            )
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        executed.append(str(job["run_id"]))
        metadata_path = run_dir / "run_metadata.json"
        stdout_path = run_dir / "stdout.json"
        stderr_path = run_dir / "stderr.log"
        started = datetime.now(UTC)
        start_clock = time.perf_counter()
        completed = subprocess.run(
            job["command"],
            capture_output=True,
            text=True,
            check=False,
        )
        duration = time.perf_counter() - start_clock
        stdout_path.write_text(
            completed.stdout,
            encoding="utf-8",
        )
        stderr_path.write_text(
            completed.stderr,
            encoding="utf-8",
        )

        status = "success" if completed.returncode == 0 else "failed"
        artifact_error: str | None = None
        if status == "success":
            try:
                result = parse_runner_stdout(completed.stdout)
                record_completed_suite_run(
                    run_dir,
                    suite_name=suite_name,
                    git_commit=git_commit,
                    job=job,
                    result=result,
                    duration_seconds=duration,
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                status = "failed"
                artifact_error = f"{type(exc).__name__}: {exc}"
                with stderr_path.open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write("\nSTRICT_ARTIFACT_ERROR: " + artifact_error + "\n")

        if status != "success":
            failure_reason = (
                artifact_error
                if artifact_error is not None
                else (f"runner subprocess exited with code {completed.returncode}")
            )
            record_failed_suite_run(
                run_dir,
                suite_name=suite_name,
                git_commit=git_commit,
                job=job,
                failure_reason=failure_reason,
            )

        metadata = {
            "schema_version": 2,
            "run_id": job["run_id"],
            "job_id": job["job_id"],
            "comparison_group": job["comparison_group"],
            "suite": suite_name,
            "suite_revision": job["suite_revision"],
            "git_commit": git_commit,
            "target": job["target"],
            "kwargs": job["kwargs"],
            "resolved_call_kwargs": job["resolved_call_kwargs"],
            "seed": job["seed"],
            "algorithm": job["algorithm"],
            "environment": job["environment"],
            "protocol": job["protocol"],
            "hypothesis_id": job["hypothesis_id"],
            "config_path": job["config_path"],
            "primary_metric": job["primary_metric"],
            "secondary_metrics": job["secondary_metrics"],
            "optional_extra": job["optional_extra"],
            "runtime_class": job["runtime_class"],
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "duration_seconds": duration,
            "returncode": completed.returncode,
            "status": status,
            "artifact_error": artifact_error,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "strict_artifacts": status == "success",
            "supersedes_archive": (str(archived_path) if archived_path is not None else None),
        }
        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if status != "success":
            failures.append(str(job["run_id"]))
            if not continue_on_error:
                break

    selected_run_ids = [str(job["run_id"]) for job in selected_jobs]
    selection = {
        "job_ids": sorted(set(job_ids or ())),
        "seeds": sorted(set(seeds or ())),
        "run_ids": sorted(set(run_ids or ())),
    }
    summary = {
        "suite": suite_name,
        "manifest_path": manifest["manifest_path"],
        "jobs_declared": len(manifest_jobs),
        "jobs_selected": len(selected_jobs),
        "selected_run_ids": selected_run_ids,
        "jobs_executed": len(executed),
        "jobs_skipped": len(skipped),
        "executed_run_ids": executed,
        "skipped_run_ids": skipped,
        "archived_runs": archived,
        "failures": failures,
        "selection": selection,
        "status": "success" if not failures else "failed",
    }

    filtered = any(
        (
            job_ids,
            seeds,
            run_ids,
        )
    )
    suite_dir = Path(output_root) / suite_name
    if filtered:
        selection_key = hashlib.sha256(
            "\n".join(sorted(selected_run_ids)).encode("utf-8")
        ).hexdigest()[:16]
        summary_dir = suite_dir / "execution_summaries"
        summary_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        summary_path = summary_dir / ("selection_" + selection_key + ".json")
    else:
        summary_path = suite_dir / "suite_execution_summary.json"
    summary["summary_path"] = str(summary_path)
    _write_json_atomic(
        summary_path,
        summary,
    )
    return summary
