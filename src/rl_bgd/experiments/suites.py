"""Paper-oriented experiment suite registry and launcher."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_bgd.artifacts import (
    RunManifest,
    RunSummary,
    metrics_rows_from_result,
    summarize_runner_result,
    write_run_artifacts,
)
from rl_bgd.artifacts.suite import information_access_for_protocol
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
    revision: int = 1


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
        optional_extra=optional_extra,
        runtime_class=runtime_class,
        seed_kwarg=seed_kwarg,
        output_kwarg=output_kwarg,
        notes=notes,
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
            "F",
            "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
            kwargs={
                "steps": steps,
                "device": "auto",
                "bayesianization": "critic_only",
                "evidence_temperature": temperature,
            },
            seeds=seeds,
            algorithm=f"SAC-BGD-beta-{temperature:g}",
            environment="synthetic_lqr",
            protocol="stationary_temperature_control",
            config_path="configs/sweeps/evidence_temperature_lqr.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.last_update_metrics.critic1_sigma_mean",
                "training.last_update_metrics.critic1_effective_lr_mean",
            ),
            runtime_class=runtime_class,
        )
        for temperature in (
            0.25,
            0.5,
            1.0,
            2.0,
        )
    )


SMOKE = ExperimentSuite(
    name="smoke",
    description="Dependency-light stationary and hidden-context execution gates.",
    jobs=(
        _job(
            "sac_adam_lqr",
            "A",
            "rl_bgd.runners.sac_lqr:run_sac_lqr",
            kwargs={"steps": 96, "device": "cpu"},
            seeds=(0,),
            algorithm="SAC-Adam",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/environments/lqr_smoke.yaml",
            primary_metric="post_return",
            secondary_metrics=("improvement",),
            runtime_class="smoke",
        ),
        _job(
            "sac_bgd_lqr",
            "A",
            "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
            kwargs={
                "steps": 96,
                "device": "cpu",
                "bayesianization": "critic_only",
            },
            seeds=(0,),
            algorithm="SAC-BGD",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/algorithms/sac_bgd.yaml",
            primary_metric="post_return",
            secondary_metrics=("improvement", "critic_sigma_mean"),
            runtime_class="smoke",
        ),
        _job(
            "ppo_adam_lqr",
            "A",
            "rl_bgd.runners.ppo_lqr:run_ppo_lqr",
            kwargs={"steps": 128, "device": "cpu"},
            seeds=(0,),
            algorithm="PPO-Adam",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/algorithms/ppo_adam.yaml",
            primary_metric="post_return",
            secondary_metrics=("improvement",),
            runtime_class="smoke",
        ),
    ),
)

DEV = ExperimentSuite(
    name="dev",
    description="Bounded multi-method development comparison before expensive benchmarks.",
    jobs=(
        *_hidden_context_jobs(
            prefix="dev_hidden",
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
            environment="recurring_lqr",
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
            environment="recurring_lqr",
            protocol="oracle_boundary",
            config_path=None,
            primary_metric="final_phase_return",
            secondary_metrics=(
                "phase_summaries",
                "boundaries",
            ),
            runtime_class="dev",
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
            secondary_metrics=("sigma_mean", "retention_lambda"),
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
            secondary_metrics=("forgetting", "bwt", "success_rate", "sigma_mean"),
            optional_extra="continual-world",
            runtime_class="very_large",
        ),
        _job(
            "cw10_recurrent_adaptive",
            "H",
            "rl_bgd.runners.recurrent_continual_world:run_recurrent_ta_continual_world_sac",
            kwargs={
                "benchmark": "CW10",
                "optimizer": "adaptive_bgd",
                "steps_per_task": 1_000_000,
                "device": "auto",
                "evaluation_episodes": 10,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="Recurrent-SAC-Adaptive-BGD",
            environment="CW10",
            protocol="3RL-style_task_agnostic",
            config_path="configs/benchmarks/three_rl_style_cw10.yaml",
            primary_metric="final_average",
            secondary_metrics=("forgetting", "bwt", "recurrence", "success_rate"),
            optional_extra="continual-world",
            runtime_class="very_large",
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
            secondary_metrics=("forgetting", "bwt", "recurrence", "success_rate"),
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
            secondary_metrics=("forgetting", "bwt", "recurrence", "success_rate"),
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
            secondary_metrics=("forgetting", "bwt", "recurrence", "success_rate"),
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
        *tuple(
            job
            for job in CW10_CORE.jobs
            if "canonical" not in job.job_id
        ),
        *tuple(
            job
            for job in CW20_FINAL.jobs
            if "canonical" not in job.job_id
        ),
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
                "steps": 600,
                "device": "auto",
                "bayesianization": mode,
                "evidence_temperature": 1.0,
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm=f"SAC-BGD-{mode}",
            environment="synthetic_lqr",
            protocol="stationary",
            config_path="configs/algorithms/sac_bgd.yaml",
            primary_metric="post_return",
            secondary_metrics=(
                "improvement",
                "training.last_update_metrics.critic1_sigma_mean",
                "training.last_update_metrics.critic1_effective_lr_mean",
            ),
            runtime_class="medium",
        )
        for mode in (
            "critic_only",
            "actor_only",
            "actor_and_critic",
        )
    )
    + _evidence_temperature_jobs(
        steps=600,
        seeds=(0, 1, 2, 3, 4),
        runtime_class="medium",
    ),
)


UNCERTAINTY_ANALYSIS = ExperimentSuite(
    name="uncertainty_analysis",
    description="Posterior uncertainty, surprise, and evidence-temperature diagnostics.",
    jobs=(
        _job(
            "adaptive_timeline",
            "E",
            "rl_bgd.runners.adaptive_bgd_lqr_stream:run_adaptive_bgd_lqr_stream",
            kwargs={
                "total_steps": 900,
                "phase_steps": 300,
                "device": "auto",
            },
            seeds=(0, 1, 2, 3, 4),
            algorithm="SAC-Adaptive-BGD",
            environment="recurring_lqr",
            protocol="strict_task_agnostic",
            config_path="configs/environments/lqr_recurring.yaml",
            primary_metric="change_detection.f1",
            secondary_metrics=(
                "change_detection.mean_detection_delay",
                "change_detection.false_alarms_per_million_steps",
                "training.last_update_metrics.retention_lambda",
                "training.last_update_metrics.critic1_sigma_mean",
            ),
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
            "rl_bgd.analysis.mechanistic:run_mechanistic_analysis",
            kwargs={},
            seeds=(150,),
            algorithm="BGD",
            environment="anisotropic_quadratic",
            protocol="mechanistic",
            config_path=None,
            primary_metric="movement_sigma_spearman",
            secondary_metrics=(
                "perturbation_precision_spearman",
                "curvature_signal_mean_relative_error",
                "freezing_target_loss",
            ),
            runtime_class="analysis",
            seed_kwarg=None,
            output_kwarg="output_dir",
        ),
    ),
)

COMPUTE_ANALYSIS = ExperimentSuite(
    name="compute_analysis",
    description="Matched stationary methods with launcher-recorded wall-clock metadata.",
    jobs=SMOKE.jobs
    + tuple(
        _job(
            f"compute_bgd_{mode}",
            "COMPUTE",
            "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr",
            kwargs={
                "steps": 600,
                "device": "auto",
                "bayesianization": mode,
            },
            seeds=(0, 1, 2),
            algorithm=f"SAC-BGD-{mode}",
            environment="synthetic_lqr",
            protocol="stationary_compute",
            config_path="configs/algorithms/sac_bgd.yaml",
            primary_metric="duration_seconds",
            secondary_metrics=("post_return", "improvement"),
            runtime_class="compute",
        )
        for mode in ("critic_only", "actor_only", "actor_and_critic")
    ),
)

SUITES: dict[str, ExperimentSuite] = {
    suite.name: suite
    for suite in (
        SMOKE,
        DEV,
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
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA")
    return result.stdout.strip() or None


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
            run_id = f"{job.job_id}__seed_{seed}"
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
            expanded.append(
                {
                    **asdict(job),
                    "seed": seed,
                    "kwargs": kwargs,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "command": command,
                }
            )
    return expanded


def validate_suite_registry() -> None:
    required = {
        "smoke",
        "dev",
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
        for job in suite.jobs:
            if not job.seeds:
                raise RuntimeError(f"job {job.job_id} has no seeds")
            if not job.primary_metric:
                raise RuntimeError(f"job {job.job_id} lacks a primary metric")
            function = resolve_target(job.target)
            signature = inspect.signature(function)
            kwargs = dict(job.kwargs)
            if job.seed_kwarg is not None:
                kwargs[job.seed_kwarg] = job.seeds[0]
            if job.output_kwarg is not None:
                kwargs[job.output_kwarg] = "artifacts/test"
            signature.bind(**kwargs)


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
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(path)
    return manifest


def _result_information_access(
    result: Mapping[str, object],
    *,
    protocol: str,
) -> dict[str, bool]:
    derived = information_access_for_protocol(
        protocol
    )
    raw = result.get(
        "information_access",
        {},
    )
    if not isinstance(
        raw,
        Mapping,
    ):
        raise TypeError(
            "runner information_access must be a mapping when present"
        )
    explicit = {
        str(key): value
        for key, value in raw.items()
        if isinstance(
            value,
            bool,
        )
    }
    for key, expected in derived.items():
        if (
            key in explicit
            and explicit[key] != expected
        ):
            raise ValueError(
                "runner information-access metadata contradicts "
                f"registered protocol {protocol!r}: {key}"
            )
    return {
        **derived,
        **explicit,
    }

def _result_task_order(
    result: Mapping[str, object],
) -> tuple[str, ...]:
    raw = result.get(
        "task_names",
        (),
    )
    if (
        isinstance(
            raw,
            (str, bytes),
        )
        or not isinstance(
            raw,
            Sequence,
        )
    ):
        return ()
    return tuple(
        str(value)
        for value in raw
    )


def _resolve_primary_metric(
    summary: RunSummary,
    metric: str,
) -> float:
    if metric in summary.metrics:
        return float(
            summary.metrics[
                metric
            ]
        )
    if metric in summary.resources:
        return float(
            summary.resources[
                metric
            ]
        )
    matches = [
        float(value)
        for key, value in summary.metrics.items()
        if key.endswith(
            f".{metric}"
        )
    ]
    if len(
        matches
    ) == 1:
        return matches[
            0
        ]
    if not matches:
        raise ValueError(
            f"declared primary metric {metric!r} is absent from run result"
        )
    raise ValueError(
        f"declared primary metric {metric!r} is ambiguous across "
        f"{len(matches)} numeric result paths"
    )


def _write_strict_suite_artifacts(
    *,
    suite_name: str,
    suite_manifest: Mapping[str, object],
    job: Mapping[str, object],
    run_dir: Path,
    result: Mapping[str, object],
    duration_seconds: float,
) -> None:
    git_commit = suite_manifest.get(
        "git_commit"
    )
    if (
        not isinstance(
            git_commit,
            str,
        )
        or not git_commit
    ):
        raise ValueError(
            "strict run provenance requires a concrete git commit"
        )
    run_id = str(
        job[
            "run_id"
        ]
    )
    summary = summarize_runner_result(
        run_id,
        result,
        duration_seconds=duration_seconds,
    )
    primary_metric = str(
        job[
            "primary_metric"
        ]
    )
    primary_value = _resolve_primary_metric(
        summary,
        primary_metric,
    )
    if (
        primary_metric
        not in summary.metrics
        and primary_metric
        not in summary.resources
    ):
        summary = RunSummary(
            run_id=summary.run_id,
            metrics={
                **summary.metrics,
                primary_metric: primary_value,
            },
            task_metrics=summary.task_metrics,
            resources=summary.resources,
            metadata=summary.metadata,
        )

    manifest = RunManifest(
        run_id=run_id,
        method=str(
            job[
                "algorithm"
            ]
        ),
        setting=str(
            job[
                "protocol"
            ]
        ),
        benchmark=str(
            job[
                "environment"
            ]
        ),
        seed=int(
            job[
                "seed"
            ]
        ),
        git_commit=git_commit,
        status="completed",
        task_order=_result_task_order(
            result
        ),
        information_access=_result_information_access(
            result,
            protocol=str(
                job[
                    "protocol"
                ]
            ),
        ),
        metadata={
            "suite": suite_name,
            "job_id": str(
                job[
                    "job_id"
                ]
            ),
            "hypothesis_id": str(
                job[
                    "hypothesis_id"
                ]
            ),
            "target": str(
                job[
                    "target"
                ]
            ),
            "primary_metric": primary_metric,
            "secondary_metrics": list(
                job[
                    "secondary_metrics"
                ]
            ),
            "runtime_class": str(
                job[
                    "runtime_class"
                ]
            ),
            "config_path": job[
                "config_path"
            ],
            "optional_extra": job[
                "optional_extra"
            ],
        },
    )
    write_run_artifacts(
        run_dir,
        manifest=manifest,
        summary=summary,
        resolved_config={
            "suite": suite_name,
            "job_id": job[
                "job_id"
            ],
            "target": job[
                "target"
            ],
            "kwargs": job[
                "kwargs"
            ],
            "seed": job[
                "seed"
            ],
            "algorithm": job[
                "algorithm"
            ],
            "environment": job[
                "environment"
            ],
            "protocol": job[
                "protocol"
            ],
            "config_path": job[
                "config_path"
            ],
            "primary_metric": primary_metric,
            "secondary_metrics": list(
                job[
                    "secondary_metrics"
                ]
            ),
        },
        metrics_rows=metrics_rows_from_result(
            result
        ),
    )


def _write_failed_suite_manifest(
    *,
    suite_name: str,
    suite_manifest: Mapping[str, object],
    job: Mapping[str, object],
    run_dir: Path,
    reason: str,
) -> None:
    git_commit = suite_manifest.get(
        "git_commit"
    )
    if (
        not isinstance(
            git_commit,
            str,
        )
        or not git_commit
    ):
        raise ValueError(
            "strict failed-run provenance requires a concrete git commit"
        )
    manifest = RunManifest(
        run_id=str(
            job[
                "run_id"
            ]
        ),
        method=str(
            job[
                "algorithm"
            ]
        ),
        setting=str(
            job[
                "protocol"
            ]
        ),
        benchmark=str(
            job[
                "environment"
            ]
        ),
        seed=int(
            job[
                "seed"
            ]
        ),
        git_commit=git_commit,
        status="failed",
        information_access=information_access_for_protocol(
            str(
                job[
                    "protocol"
                ]
            )
        ),
        metadata={
            "suite": suite_name,
            "job_id": str(
                job[
                    "job_id"
                ]
            ),
            "hypothesis_id": str(
                job[
                    "hypothesis_id"
                ]
            ),
            "target": str(
                job[
                    "target"
                ]
            ),
            "primary_metric": str(
                job[
                    "primary_metric"
                ]
            ),
            "secondary_metrics": list(
                job[
                    "secondary_metrics"
                ]
            ),
            "runtime_class": str(
                job[
                    "runtime_class"
                ]
            ),
            "config_path": job[
                "config_path"
            ],
            "optional_extra": job[
                "optional_extra"
            ],
            "failure_reason": reason,
        },
    )
    (
        run_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def execute_suite(
    suite_name: str,
    output_root: str | Path,
    *,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    manifest = materialize_suite(
        suite_name,
        output_root,
    )
    git_commit = manifest.get(
        "git_commit"
    )
    if (
        not isinstance(
            git_commit,
            str,
        )
        or not git_commit
    ):
        raise RuntimeError(
            "suite execution requires a concrete git commit for provenance"
        )
    failures: list[str] = []
    for job in manifest["jobs"]:
        run_dir = Path(
            job[
                "run_dir"
            ]
        )
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        metadata_path = (
            run_dir
            / "run_metadata.json"
        )
        stdout_path = (
            run_dir
            / "stdout.json"
        )
        stderr_path = (
            run_dir
            / "stderr.log"
        )
        started = datetime.now(
            UTC
        )
        start_clock = time.perf_counter()
        completed = subprocess.run(
            job[
                "command"
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        duration = (
            time.perf_counter()
            - start_clock
        )
        stdout_path.write_text(
            completed.stdout,
            encoding="utf-8",
        )
        stderr_path.write_text(
            completed.stderr,
            encoding="utf-8",
        )

        status = (
            "success"
            if completed.returncode == 0
            else "failed"
        )
        artifact_error: str | None = None
        if status == "success":
            try:
                parsed = json.loads(
                    completed.stdout
                )
                if not isinstance(
                    parsed,
                    Mapping,
                ):
                    raise TypeError(
                        "runner stdout JSON must contain an object"
                    )
                _write_strict_suite_artifacts(
                    suite_name=suite_name,
                    suite_manifest=manifest,
                    job=job,
                    run_dir=run_dir,
                    result={
                        str(key): value
                        for key, value in parsed.items()
                    },
                    duration_seconds=duration,
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                status = "failed"
                artifact_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                with stderr_path.open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(
                        "\nSTRICT_ARTIFACT_ERROR: "
                        + artifact_error
                        + "\n"
                    )

        if status != "success":
            failure_reason = (
                artifact_error
                if artifact_error is not None
                else (
                    "runner subprocess exited with "
                    f"code {completed.returncode}"
                )
            )
            _write_failed_suite_manifest(
                suite_name=suite_name,
                suite_manifest=manifest,
                job=job,
                run_dir=run_dir,
                reason=failure_reason,
            )

        metadata = {
            "schema_version": 2,
            "run_id": job[
                "run_id"
            ],
            "job_id": job[
                "job_id"
            ],
            "suite": suite_name,
            "git_commit": manifest[
                "git_commit"
            ],
            "target": job[
                "target"
            ],
            "kwargs": job[
                "kwargs"
            ],
            "seed": job[
                "seed"
            ],
            "algorithm": job[
                "algorithm"
            ],
            "environment": job[
                "environment"
            ],
            "protocol": job[
                "protocol"
            ],
            "hypothesis_id": job[
                "hypothesis_id"
            ],
            "config_path": job[
                "config_path"
            ],
            "primary_metric": job[
                "primary_metric"
            ],
            "secondary_metrics": job[
                "secondary_metrics"
            ],
            "optional_extra": job[
                "optional_extra"
            ],
            "runtime_class": job[
                "runtime_class"
            ],
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(
                UTC
            ).isoformat(),
            "duration_seconds": duration,
            "returncode": completed.returncode,
            "status": status,
            "artifact_error": artifact_error,
            "stdout_path": str(
                stdout_path
            ),
            "stderr_path": str(
                stderr_path
            ),
            "strict_artifacts": (
                status == "success"
            ),
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
            failures.append(
                str(
                    job[
                        "run_id"
                    ]
                )
            )
            if not continue_on_error:
                break

    summary = {
        "suite": suite_name,
        "manifest_path": manifest[
            "manifest_path"
        ],
        "jobs_declared": len(
            manifest[
                "jobs"
            ]
        ),
        "failures": failures,
        "status": (
            "success"
            if not failures
            else "failed"
        ),
    }
    summary_path = (
        Path(
            output_root
        )
        / suite_name
        / "suite_execution_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary

