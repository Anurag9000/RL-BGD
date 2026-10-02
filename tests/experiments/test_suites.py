import json
from pathlib import Path

import pytest

from rl_bgd.artifacts import load_run_directory
from rl_bgd.experiments.suites import (
    SUITES,
    ExperimentJob,
    ExperimentSuite,
    execute_suite,
    materialize_suite,
    validate_suite_registry,
)


def test_required_paper_suites_validate_against_real_targets() -> None:
    validate_suite_registry()
    assert set(SUITES) == {
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


def test_suite_manifest_contains_complete_job_metadata(
    tmp_path: Path,
) -> None:
    manifest = materialize_suite(
        "smoke",
        tmp_path,
    )
    path = Path(manifest["manifest_path"])
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["suite"] == "smoke"
    assert saved["suite_revision"] == SUITES["smoke"].revision
    assert saved["jobs"]
    for job in saved["jobs"]:
        assert job["suite_revision"] == saved["suite_revision"]
        assert job["run_id"]
        assert job["hypothesis_id"]
        assert job["target"]
        assert job["algorithm"]
        assert job["environment"]
        assert job["protocol"]
        assert job["primary_metric"]
        assert job["command"]
        assert job["run_dir"]


def test_comparison_and_ablation_jobs_are_atomic() -> None:
    hidden_jobs = [job for job in SUITES["dev"].jobs if job.job_id.startswith("hidden_context_")]
    assert len(hidden_jobs) == 5
    assert all(job.target.endswith(":run_hidden_context_sac_variant") for job in hidden_jobs)
    assert len({job.kwargs["variant"] for job in hidden_jobs}) == 5

    ablations = SUITES["ablation_core"].jobs
    replay_jobs = [job for job in ablations if job.hypothesis_id == "F"]
    assert {job.kwargs["replay_evidence_mode"] for job in replay_jobs} == {
        "all_replay",
        "fresh_only_uncertainty",
        "inverse_reuse_weight",
        "normalized_batch_evidence",
    }
    assert {job.hypothesis_id for job in ablations} >= {
        "C",
        "D",
        "F",
        "G",
        "GB-T",
    }

    plasticity_jobs = [job for job in ablations if job.hypothesis_id == "C"]
    assert {job.kwargs["consolidation_retention"] for job in plasticity_jobs} == {
        1.0,
        0.97,
    }
    assert all(job.primary_metric == "post_shift_normalized_auc" for job in plasticity_jobs)


def test_paper_suites_cover_every_execution_hypothesis() -> None:
    hypothesis_ids = {job.hypothesis_id for suite in SUITES.values() for job in suite.jobs}
    assert {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "GB-T",
        "H",
        "CW-CAN10",
        "CW-CAN20",
        "CW-TA10",
        "CW-TA20",
        "UCL",
    } <= hypothesis_ids
    assert "I-J" in hypothesis_ids


def test_stationary_and_mechanism_suites_have_replicate_coverage() -> None:
    stationary = SUITES["stationary_core"].jobs
    assert {job.job_id for job in stationary} == {
        "stationary_sac_adam",
        "stationary_sac_bgd",
        "stationary_ppo_adam",
        "stationary_ppo_bgd",
    }
    assert all(job.hypothesis_id == "A" for job in stationary)
    assert all(
        job.seeds
        == (
            0,
            1,
            2,
            3,
            4,
        )
        for job in stationary
    )

    mechanisms = SUITES["mechanism_analysis"].jobs
    assert len(mechanisms) == 1
    assert mechanisms[0].seeds == (
        150,
        151,
        152,
        153,
        154,
    )
    assert mechanisms[0].seed_kwarg == "seed"


def test_external_baseline_suite_has_full_method_coverage() -> None:
    jobs = SUITES["baseline_core"].jobs
    regularized = [job for job in jobs if job.target.endswith(":run_regularized_sac_recurring_lqr")]
    assert {job.kwargs["method"] for job in regularized} == {
        "ewc",
        "online_ewc",
        "si",
        "mas",
    }
    assert all(
        job.seeds
        == (
            0,
            1,
            2,
            3,
            4,
        )
        for job in jobs
    )

    sac_controls = [job for job in jobs if job.target.endswith(":run_sac_recurring_lqr_control")]
    assert len(sac_controls) == 1
    sac_family = [
        *sac_controls,
        *regularized,
    ]
    assert {job.environment for job in sac_family} == {"recurring_lqr_matched_v1"}
    assert {job.protocol for job in sac_family} == {"task_agnostic_fixed_update"}
    assert {int(job.kwargs["steps"]) for job in sac_family} == {600}
    assert {int(job.kwargs["phase_steps"]) for job in sac_family} == {120}
    assert {int(job.kwargs["horizon"]) for job in sac_family} == {32}

    ucl = [job for job in jobs if job.target.endswith(":run_ucl_ppo_recurring_lqr")]
    ucl_controls = [
        job for job in jobs if job.target.endswith(":run_adam_ppo_oracle_recurring_lqr_control")
    ]
    assert len(ucl) == 1
    assert len(ucl_controls) == 1
    ppo_family = [
        *ucl_controls,
        *ucl,
    ]
    assert {job.environment for job in ppo_family} == {"recurring_lqr_matched_v1"}
    assert {job.protocol for job in ppo_family} == {"oracle_boundary"}
    assert {int(job.kwargs["phase_steps"]) for job in ppo_family} == {120}
    assert {int(job.kwargs["phases"]) for job in ppo_family} == {5}
    assert {int(job.kwargs["horizon"]) for job in ppo_family} == {32}


def test_carl_and_cw_metric_declarations_match_runner_contracts() -> None:
    carl_jobs = SUITES[
        "carl_core"
    ].jobs
    for job in carl_jobs:
        optimizer = job.kwargs[
            "optimizer"
        ]
        if optimizer == "adam":
            assert job.secondary_metrics == ()
        elif optimizer == "bgd":
            assert job.secondary_metrics == (
                "training.last_update_metrics.critic1_sigma_mean",
            )
        else:
            assert optimizer == "adaptive_bgd"
            assert job.secondary_metrics == (
                "training.last_update_metrics.critic1_sigma_mean",
                "training.last_update_metrics.retention_lambda",
            )

    cw10_jobs = {
        job.job_id: job
        for job in SUITES[
            "cw10_core"
        ].jobs
    }
    assert "recurrence" not in cw10_jobs[
        "cw10_recurrent_adaptive"
    ].secondary_metrics
    assert (
        "training.last_update_metrics.critic1_sigma_mean"
        in cw10_jobs[
            "cw10_ta_bgd"
        ].secondary_metrics
    )

    cw20_jobs = {
        job.job_id: job
        for job in SUITES[
            "cw20_final"
        ].jobs
    }
    assert not any(
        metric.startswith(
            "recurrence_summary."
        )
        for metric in cw20_jobs[
            "cw20_canonical_adam"
        ].secondary_metrics
    )
    recurrence_metrics = {
        "recurrence_summary.reference_success",
        "recurrence_summary.zero_shot_success",
        "recurrence_summary.recovered_success",
        "recurrence_summary.pre_revisit_change",
        "recurrence_summary.relearning_gain",
    }
    for job_id in (
        "cw20_ta_adam",
        "cw20_ta_bgd",
        "cw20_recurrent_adam",
        "cw20_recurrent_bgd",
        "cw20_recurrent_adaptive",
    ):
        assert recurrence_metrics <= set(
            cw20_jobs[
                job_id
            ].secondary_metrics
        )


def test_cw20_final_has_matched_task_agnostic_and_recurrent_controls() -> None:
    jobs = SUITES[
        "cw20_final"
    ].jobs
    task_agnostic = [
        job
        for job in jobs
        if job.target.endswith(
            ":run_ta_continual_world_sac"
        )
    ]
    assert {
        job.kwargs[
            "optimizer"
        ]
        for job in task_agnostic
    } == {
        "adam",
        "bgd",
    }
    assert {
        int(
            job.kwargs[
                "steps_per_task"
            ]
        )
        for job in task_agnostic
    } == {
        1_000_000
    }
    assert {
        int(
            job.kwargs[
                "evaluation_episodes"
            ]
        )
        for job in task_agnostic
    } == {
        5
    }
    assert {
        job.seeds
        for job in task_agnostic
    } == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }
    assert {
        job.protocol
        for job in task_agnostic
    } == {
        "strict_task_agnostic"
    }

    recurrent = [
        job
        for job in jobs
        if job.target.endswith(
            ":run_recurrent_ta_continual_world_sac"
        )
    ]
    assert {
        job.kwargs[
            "optimizer"
        ]
        for job in recurrent
    } == {
        "adam",
        "bgd",
        "adaptive_bgd",
    }
    assert {
        int(
            job.kwargs[
                "steps_per_task"
            ]
        )
        for job in recurrent
    } == {
        1_000_000
    }
    assert {
        int(
            job.kwargs[
                "evaluation_episodes"
            ]
        )
        for job in recurrent
    } == {
        10
    }
    assert {
        job.seeds
        for job in recurrent
    } == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }
    assert {
        job.protocol
        for job in recurrent
    } == {
        "3RL-style_task_agnostic"
    }


def test_hidden_and_bayesianization_metric_declarations_are_capability_matched() -> None:
    hidden = {
        str(
            job.kwargs[
                "variant"
            ]
        ): job
        for job in SUITES[
            "dev"
        ].jobs
        if job.job_id.startswith(
            "hidden_context_"
        )
    }
    assert hidden[
        "feedforward_adam"
    ].secondary_metrics == (
        "training.mean_episode_return",
    )
    assert hidden[
        "recurrent_adam"
    ].secondary_metrics == (
        "training.mean_episode_return",
    )
    for variant in (
        "feedforward_bgd",
        "recurrent_bgd",
    ):
        assert hidden[
            variant
        ].secondary_metrics == (
            "training.mean_episode_return",
            "training.last_update_metrics.critic1_sigma_mean",
        )
    assert hidden[
        "recurrent_adaptive_bgd"
    ].secondary_metrics == (
        "training.mean_episode_return",
        "training.last_update_metrics.critic1_sigma_mean",
        "training.last_update_metrics.retention_lambda",
    )

    ablations = {
        str(
            job.kwargs[
                "bayesianization"
            ]
        ): job
        for job in SUITES[
            "ablation_core"
        ].jobs
        if job.hypothesis_id == "G"
    }
    assert ablations[
        "critic_only"
    ].secondary_metrics == (
        "improvement",
        "training.last_update_metrics.critic1_sigma_mean",
        "training.last_update_metrics.critic1_effective_lr_mean",
    )
    assert ablations[
        "actor_only"
    ].secondary_metrics == (
        "improvement",
        "training.last_update_metrics.actor_sigma_mean",
        "training.last_update_metrics.actor_effective_lr_mean",
    )
    assert set(
        ablations[
            "actor_and_critic"
        ].secondary_metrics
    ) == {
        "improvement",
        "training.last_update_metrics.critic1_sigma_mean",
        "training.last_update_metrics.critic1_effective_lr_mean",
        "training.last_update_metrics.actor_sigma_mean",
        "training.last_update_metrics.actor_effective_lr_mean",
    }


def test_compute_suite_has_matched_sac_budget_and_resources() -> None:
    jobs = SUITES[
        "compute_analysis"
    ].jobs
    assert {
        job.job_id
        for job in jobs
    } == {
        "compute_sac_adam",
        "compute_bgd_critic_only",
        "compute_bgd_actor_only",
        "compute_bgd_actor_and_critic",
    }
    assert {
        int(
            job.kwargs[
                "steps"
            ]
        )
        for job in jobs
    } == {
        600
    }
    assert {
        job.seeds
        for job in jobs
    } == {
        (
            0,
            1,
            2,
        )
    }
    assert {
        job.environment
        for job in jobs
    } == {
        "synthetic_lqr"
    }
    assert {
        job.protocol
        for job in jobs
    } == {
        "stationary_compute"
    }
    assert {
        job.primary_metric
        for job in jobs
    } == {
        "duration_seconds"
    }
    assert all(
        job.secondary_metrics
        == (
            "post_return",
            "improvement",
        )
        for job in jobs
    )


def test_ucl_structured_outputs_are_not_declared_as_scalar_metrics() -> None:
    ucl_jobs = [
        job
        for suite_name in (
            "dev",
            "baseline_core",
        )
        for job in SUITES[
            suite_name
        ].jobs
        if (
            "ucl_ppo_lqr"
            in job.target
        )
    ]
    assert ucl_jobs
    assert all(
        job.secondary_metrics
        == ()
        for job in ucl_jobs
    )


def test_uncertainty_suite_has_matched_surprise_source_controls() -> None:
    jobs = [
        job
        for job in SUITES[
            "uncertainty_analysis"
        ].jobs
        if job.target.endswith(
            ":run_adaptive_bgd_lqr_stream"
        )
    ]
    by_source = {
        str(
            job.kwargs[
                "surprise_source"
            ]
        ): job
        for job in jobs
    }
    assert set(
        by_source
    ) == {
        "none",
        "td",
        "ensemble",
        "predictive",
    }
    assert {
        int(
            job.kwargs[
                "total_steps"
            ]
        )
        for job in jobs
    } == {
        900
    }
    assert {
        int(
            job.kwargs[
                "phase_steps"
            ]
        )
        for job in jobs
    } == {
        300
    }
    assert {
        job.seeds
        for job in jobs
    } == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }
    assert {
        job.environment
        for job in jobs
    } == {
        "recurring_lqr"
    }
    assert {
        job.protocol
        for job in jobs
    } == {
        "strict_task_agnostic"
    }
    for source in (
        "td",
        "ensemble",
        "predictive",
    ):
        assert (
            "training.last_update_metrics.retention_lambda"
            in by_source[
                source
            ].secondary_metrics
        )
    assert (
        "training.last_update_metrics.retention_lambda"
        not in by_source[
            "none"
        ].secondary_metrics
    )
    assert (
        "training.last_update_metrics.predictive_model_loss"
        in by_source[
            "predictive"
        ].secondary_metrics
    )


def test_mechanistic_and_detection_metrics_are_scalar_safe() -> None:
    mechanism = SUITES[
        "mechanism_analysis"
    ].jobs[
        0
    ]
    assert mechanism.secondary_metrics == (
        "perturbation_precision_spearman",
        "curvature_signal_mean_relative_error",
        "freezing_target_loss.none",
        "freezing_target_loss.freeze_low_sigma",
        "freezing_target_loss.freeze_high_sigma",
    )

    adaptive = next(
        job
        for job in SUITES[
            "uncertainty_analysis"
        ].jobs
        if job.job_id
        == "adaptive_timeline"
    )
    assert (
        "change_detection.mean_detection_delay"
        not in adaptive.secondary_metrics
    )
    assert {
        "change_detection.precision",
        "change_detection.recall",
        "change_detection.false_alarms_per_million_steps",
    } <= set(
        adaptive.secondary_metrics
    )


def _tiny_suite(
    primary_metric: str,
) -> ExperimentSuite:
    return ExperimentSuite(
        name="smoke",
        description=("strict-artifact execution test"),
        jobs=(
            ExperimentJob(
                job_id="tiny_smoke",
                hypothesis_id="TEST",
                target=("rl_bgd.runners.smoke:run_smoke"),
                kwargs={
                    "steps": 4,
                    "device": "cpu",
                },
                seeds=(0,),
                algorithm="BGD-smoke",
                environment="quadratic",
                protocol="stationary",
                config_path=None,
                primary_metric=primary_metric,
                secondary_metrics=("initial_abs_mean",),
                runtime_class="smoke",
            ),
        ),
    )


def test_execute_suite_writes_strict_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = _tiny_suite("final_abs_mean")
    monkeypatch.setitem(
        SUITES,
        "smoke",
        suite,
    )
    result = execute_suite(
        "smoke",
        tmp_path,
    )
    assert result["status"] == "success"
    run_dir = tmp_path / "smoke" / "smoke__tiny_smoke__seed_0"
    loaded = load_run_directory(run_dir)
    assert loaded.manifest.method == ("BGD-smoke")
    assert loaded.summary.metrics["final_abs_mean"] >= 0.0
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["strict_artifacts"] is True
    assert metadata["job_id"] == "tiny_smoke"


def test_execute_suite_fails_closed_on_missing_primary_metric(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = _tiny_suite("does_not_exist")
    monkeypatch.setitem(
        SUITES,
        suite.name,
        suite,
    )
    result = execute_suite(
        suite.name,
        tmp_path,
    )
    assert result["status"] == "failed"
    run_dir = tmp_path / suite.name / "smoke__tiny_smoke__seed_0"
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["strict_artifacts"] is False
    assert "declared primary metric" in metadata["artifact_error"]
    failed_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert failed_manifest["status"] == "failed"
    with pytest.raises(
        ValueError,
        match="refuses incomplete",
    ):
        load_run_directory(run_dir)
