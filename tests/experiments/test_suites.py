import json
from pathlib import Path

import pytest
import yaml

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
    assert saved["suite_revision"] == SUITES["smoke"].revision == 3
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
        assert "comparison_group" in job
        assert job["resolved_call_kwargs"]
        assert job["resolved_call_kwargs"]["seed"] == job["seed"]
        assert job["command"]
        assert job["run_dir"]


def test_explicit_comparison_groups_prevent_cross_family_pairing() -> None:
    stationary = SUITES["stationary_core"].jobs
    by_id = {job.job_id: job for job in stationary}
    assert by_id["stationary_sac_adam"].comparison_group == "stationary_sac"
    assert by_id["stationary_sac_bgd"].comparison_group == "stationary_sac"
    assert by_id["stationary_ppo_adam"].comparison_group == "stationary_ppo"
    assert by_id["stationary_ppo_bgd"].comparison_group == "stationary_ppo"

    carl = SUITES["carl_core"].jobs
    for mode in (
        "abrupt",
        "smooth",
        "recurring",
    ):
        jobs = [job for job in carl if job.job_id.startswith(f"carl_{mode}_")]
        assert len(jobs) == 3
        assert {job.comparison_group for job in jobs} == {f"carl_{mode}"}

    baseline = SUITES["baseline_core"].jobs
    sac_jobs = [job for job in baseline if job.algorithm.startswith("SAC")]
    ppo_jobs = [job for job in baseline if job.algorithm.startswith("PPO")]
    assert {job.comparison_group for job in sac_jobs} == {"baseline_sac_regularization"}
    assert {job.comparison_group for job in ppo_jobs} == {"baseline_ppo_ucl"}


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
        "MC-K",
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
        "MC-K",
        "CW-CAN10",
        "CW-CAN20",
        "CW-TA10",
        "CW-TA20",
        "UCL",
    } <= hypothesis_ids
    assert "I-J" in hypothesis_ids


def test_stationary_suite_has_exact_matched_optimizer_controls() -> None:
    jobs = {
        job.job_id: job
        for job in SUITES["stationary_core"].jobs
    }

    sac_adam = jobs["stationary_sac_adam"].kwargs
    sac_bgd = jobs["stationary_sac_bgd"].kwargs
    for key, value in sac_adam.items():
        assert sac_bgd[key] == value
    assert sac_bgd["bayesianization"] == "critic_only"
    assert sac_bgd["mc_samples"] == 2
    assert sac_adam["evaluation_seed"] == sac_bgd["evaluation_seed"] == 20_000

    ppo_adam = jobs["stationary_ppo_adam"].kwargs
    ppo_bgd = jobs["stationary_ppo_bgd"].kwargs
    for key, value in ppo_adam.items():
        assert ppo_bgd[key] == value
    assert ppo_bgd["bayesianization"] == "actor_and_value"
    assert ppo_bgd["mc_samples"] == 2
    assert ppo_adam["update_epochs"] == ppo_bgd["update_epochs"] == 4
    assert ppo_adam["evaluation_seed"] == ppo_bgd["evaluation_seed"] == 30_000


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
    carl_jobs = SUITES["carl_core"].jobs
    for job in carl_jobs:
        optimizer = job.kwargs["optimizer"]
        if optimizer == "adam":
            assert job.secondary_metrics == ()
        elif optimizer == "bgd":
            assert job.secondary_metrics == ("training.last_update_metrics.critic1_sigma_mean",)
        else:
            assert optimizer == "adaptive_bgd"
            assert job.secondary_metrics == (
                "training.last_update_metrics.critic1_sigma_mean",
                "training.last_update_metrics.retention_lambda",
            )

    cw10_jobs = {job.job_id: job for job in SUITES["cw10_core"].jobs}
    for job_id in (
        "cw10_recurrent_adam",
        "cw10_recurrent_bgd",
        "cw10_recurrent_adaptive_bgd",
    ):
        assert not any(
            metric.startswith("recurrence_summary.")
            for metric in cw10_jobs[job_id].secondary_metrics
        )
    assert (
        "training.last_update_metrics.critic1_sigma_mean"
        in cw10_jobs["cw10_ta_bgd"].secondary_metrics
    )
    assert (
        "training.last_update_metrics.critic1_sigma_mean"
        in cw10_jobs["cw10_recurrent_bgd"].secondary_metrics
    )

    cw20_jobs = {job.job_id: job for job in SUITES["cw20_final"].jobs}
    assert not any(
        metric.startswith("recurrence_summary.")
        for metric in cw20_jobs["cw20_canonical_adam"].secondary_metrics
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
        "cw20_ta_ewc",
        "cw20_ta_online_ewc",
        "cw20_ta_si",
        "cw20_ta_mas",
        "cw20_recurrent_adam",
        "cw20_recurrent_bgd",
        "cw20_recurrent_adaptive",
    ):
        assert recurrence_metrics <= set(cw20_jobs[job_id].secondary_metrics)


def test_cw10_regularization_baselines_are_matched_and_boundary_free() -> None:
    jobs = [
        job
        for job in SUITES["cw10_core"].jobs
        if job.kwargs.get("optimizer")
        in {
            "ewc",
            "online_ewc",
            "si",
            "mas",
        }
    ]
    assert {job.kwargs["optimizer"] for job in jobs} == {
        "ewc",
        "online_ewc",
        "si",
        "mas",
    }
    assert {job.seeds for job in jobs} == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }
    assert {int(job.kwargs["steps_per_task"]) for job in jobs} == {1_000_000}
    assert {int(job.kwargs["evaluation_episodes"]) for job in jobs} == {5}
    assert {int(job.kwargs["consolidation_interval_updates"]) for job in jobs} == {50_000}
    assert {job.kwargs["regularization_target"] for job in jobs} == {"actor_and_critic"}
    assert {job.protocol for job in jobs} == {"strict_task_agnostic"}
    assert {job.comparison_group for job in jobs} == {"cw10_task_agnostic_feedforward"}
    assert all("no ground-truth task boundary" in job.notes for job in jobs)


def test_cw10_has_matched_feedforward_and_recurrent_families() -> None:
    jobs = SUITES["cw10_core"].jobs
    feedforward = [job for job in jobs if job.target.endswith(":run_ta_continual_world_sac")]
    assert {job.kwargs["optimizer"] for job in feedforward} == {
        "adam",
        "bgd",
        "ewc",
        "online_ewc",
        "si",
        "mas",
    }
    assert {job.protocol for job in feedforward} == {"strict_task_agnostic"}
    assert {job.comparison_group for job in feedforward} == {"cw10_task_agnostic_feedforward"}
    assert {job.seeds for job in feedforward} == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }
    assert {int(job.kwargs["steps_per_task"]) for job in feedforward} == {1_000_000}

    recurrent = [
        job for job in jobs if job.target.endswith(":run_recurrent_ta_continual_world_sac")
    ]
    assert {job.kwargs["optimizer"] for job in recurrent} == {
        "adam",
        "bgd",
        "adaptive_bgd",
    }
    assert {job.protocol for job in recurrent} == {"3RL-style_task_agnostic"}
    assert {job.comparison_group for job in recurrent} == {"cw10_recurrent"}
    assert {job.seeds for job in recurrent} == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }


def test_cw20_final_has_matched_task_agnostic_and_recurrent_controls() -> None:
    jobs = SUITES["cw20_final"].jobs
    task_agnostic = [job for job in jobs if job.target.endswith(":run_ta_continual_world_sac")]
    assert {job.kwargs["optimizer"] for job in task_agnostic} == {
        "adam",
        "bgd",
        "ewc",
        "online_ewc",
        "si",
        "mas",
    }
    assert {int(job.kwargs["steps_per_task"]) for job in task_agnostic} == {1_000_000}
    assert {int(job.kwargs["evaluation_episodes"]) for job in task_agnostic} == {5}
    assert {job.seeds for job in task_agnostic} == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }
    assert {job.protocol for job in task_agnostic} == {"strict_task_agnostic"}
    assert {job.comparison_group for job in task_agnostic} == {"cw20_task_agnostic_feedforward"}

    recurrent = [
        job for job in jobs if job.target.endswith(":run_recurrent_ta_continual_world_sac")
    ]
    assert {job.kwargs["optimizer"] for job in recurrent} == {
        "adam",
        "bgd",
        "adaptive_bgd",
    }
    assert {int(job.kwargs["steps_per_task"]) for job in recurrent} == {1_000_000}
    assert {int(job.kwargs["evaluation_episodes"]) for job in recurrent} == {10}
    assert {job.seeds for job in recurrent} == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }
    assert {job.protocol for job in recurrent} == {"3RL-style_task_agnostic"}
    assert {job.comparison_group for job in recurrent} == {"cw20_recurrent"}


def test_hidden_and_bayesianization_metric_declarations_are_capability_matched() -> None:
    hidden = {
        str(job.kwargs["variant"]): job
        for job in SUITES["dev"].jobs
        if job.job_id.startswith("hidden_context_")
    }
    assert hidden["feedforward_adam"].secondary_metrics == ("training.mean_episode_return",)
    assert hidden["recurrent_adam"].secondary_metrics == ("training.mean_episode_return",)
    for variant in (
        "feedforward_bgd",
        "recurrent_bgd",
    ):
        assert hidden[variant].secondary_metrics == (
            "training.mean_episode_return",
            "training.last_update_metrics.critic1_sigma_mean",
        )
    assert hidden["recurrent_adaptive_bgd"].secondary_metrics == (
        "training.mean_episode_return",
        "training.last_update_metrics.critic1_sigma_mean",
        "training.last_update_metrics.retention_lambda",
    )

    ablations = {
        str(job.kwargs["bayesianization"]): job
        for job in SUITES["ablation_core"].jobs
        if job.hypothesis_id == "G"
    }
    assert ablations["critic_only"].secondary_metrics == (
        "improvement",
        "training.last_update_metrics.critic1_sigma_mean",
        "training.last_update_metrics.critic1_effective_lr_mean",
    )
    assert ablations["actor_only"].secondary_metrics == (
        "improvement",
        "training.last_update_metrics.actor_sigma_mean",
        "training.last_update_metrics.actor_effective_lr_mean",
    )
    assert set(ablations["actor_and_critic"].secondary_metrics) == {
        "improvement",
        "training.last_update_metrics.critic1_sigma_mean",
        "training.last_update_metrics.critic1_effective_lr_mean",
        "training.last_update_metrics.actor_sigma_mean",
        "training.last_update_metrics.actor_effective_lr_mean",
    }


def test_mc_sample_ablation_has_matched_k_support() -> None:
    jobs = [job for job in SUITES["ablation_core"].jobs if job.hypothesis_id == "MC-K"]
    assert {int(job.kwargs["mc_samples"]) for job in jobs} == {
        1,
        2,
        4,
        8,
    }
    assert {int(job.kwargs["steps"]) for job in jobs} == {600}
    assert {job.seeds for job in jobs} == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }
    assert {job.kwargs["bayesianization"] for job in jobs} == {"critic_only"}
    assert {job.comparison_group for job in jobs} == {"mc_samples"}


def test_compute_suite_has_matched_factorized_controls() -> None:
    jobs = SUITES["compute_analysis"].jobs
    assert {job.job_id for job in jobs} == {
        "compute_sac_adam",
        "compute_bgd_critic_only",
        "compute_bgd_actor_only",
        "compute_bgd_actor_and_critic",
        "compute_bgd_k1",
        "compute_bgd_k2",
        "compute_bgd_k4",
        "compute_bgd_k8",
    }
    assert {int(job.kwargs["steps"]) for job in jobs} == {600}
    assert {job.seeds for job in jobs} == {
        (
            0,
            1,
            2,
        )
    }
    assert {job.environment for job in jobs} == {"synthetic_lqr"}
    assert {job.protocol for job in jobs} == {"stationary_compute"}
    assert {job.primary_metric for job in jobs} == {"duration_seconds"}
    assert all(
        job.secondary_metrics
        == (
            "post_return",
            "improvement",
        )
        for job in jobs
    )

    bayesianization_jobs = [
        job for job in jobs if job.comparison_group == "compute_bayesianization"
    ]
    assert {job.job_id for job in bayesianization_jobs} == {
        "compute_sac_adam",
        "compute_bgd_critic_only",
        "compute_bgd_actor_only",
        "compute_bgd_actor_and_critic",
    }
    bgd_modes = {
        str(job.kwargs["bayesianization"]): job
        for job in bayesianization_jobs
        if job.job_id != "compute_sac_adam"
    }
    assert set(bgd_modes) == {
        "critic_only",
        "actor_only",
        "actor_and_critic",
    }
    assert {int(job.kwargs["mc_samples"]) for job in bgd_modes.values()} == {2}

    compute_mc_jobs = {
        int(job.kwargs["mc_samples"]): job
        for job in jobs
        if job.comparison_group == "compute_mc_samples"
    }
    assert set(compute_mc_jobs) == {
        1,
        2,
        4,
        8,
    }
    assert all(job.kwargs["bayesianization"] == "critic_only" for job in compute_mc_jobs.values())

    stationary_sac = {
        job.job_id: job
        for job in SUITES["stationary_core"].jobs
    }["stationary_sac_adam"].kwargs
    shared_keys = set(stationary_sac)
    for job in jobs:
        for key in shared_keys:
            expected = 600 if key == "steps" else stationary_sac[key]
            assert job.kwargs[key] == expected
    assert {int(job.kwargs["evaluation_seed"]) for job in jobs} == {20_000}


def test_ucl_structured_outputs_are_not_declared_as_scalar_metrics() -> None:
    ucl_jobs = [
        job
        for suite_name in (
            "dev",
            "baseline_core",
        )
        for job in SUITES[suite_name].jobs
        if ("ucl_ppo_lqr" in job.target)
    ]
    assert ucl_jobs
    assert all(job.secondary_metrics == () for job in ucl_jobs)


def test_uncertainty_suite_separates_detection_and_retention_comparisons() -> None:
    all_jobs = SUITES["uncertainty_analysis"].jobs

    detection_jobs = [
        job for job in all_jobs if job.comparison_group == "surprise_source_detection"
    ]
    by_source = {str(job.kwargs["surprise_source"]): job for job in detection_jobs}
    assert set(by_source) == {
        "none",
        "td",
        "ensemble",
        "predictive",
    }
    assert all(job.primary_metric == "change_detection.f1" for job in detection_jobs)
    assert {int(job.kwargs["total_steps"]) for job in detection_jobs} == {900}
    assert {int(job.kwargs["phase_steps"]) for job in detection_jobs} == {300}
    assert {job.seeds for job in detection_jobs} == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }
    assert {job.environment for job in detection_jobs} == {"recurring_lqr"}
    assert {job.protocol for job in detection_jobs} == {"strict_task_agnostic"}
    for source in (
        "td",
        "ensemble",
        "predictive",
    ):
        assert (
            "training.last_update_metrics.retention_lambda" in by_source[source].secondary_metrics
        )
        assert "surprise_auroc" in by_source[source].secondary_metrics
    assert (
        "training.last_update_metrics.retention_lambda" not in by_source["none"].secondary_metrics
    )
    assert (
        "training.last_update_metrics.predictive_model_loss"
        in by_source["predictive"].secondary_metrics
    )

    retention_jobs = [
        job for job in all_jobs if job.comparison_group == "recurring_retention_policy"
    ]
    assert {str(job.kwargs["surprise_source"]) for job in retention_jobs} == {
        "none",
        "td",
        "ensemble",
        "predictive",
    }
    assert len(retention_jobs) == 5
    assert {
        float(job.kwargs["fixed_retention"])
        for job in retention_jobs
        if job.kwargs["surprise_source"] == "none"
    } == {
        1.0,
        0.97,
    }
    assert all(job.primary_metric == "training.final_10_mean_return" for job in retention_jobs)
    assert {job.protocol for job in retention_jobs} == {"strict_task_agnostic_retention_policy"}
    assert {job.seeds for job in retention_jobs} == {
        (
            0,
            1,
            2,
            3,
            4,
        )
    }
    assert {int(job.kwargs["total_steps"]) for job in retention_jobs} == {900}
    assert {int(job.kwargs["phase_steps"]) for job in retention_jobs} == {300}


def test_mechanistic_and_detection_metrics_are_scalar_safe() -> None:
    mechanism = SUITES["mechanism_analysis"].jobs[0]
    assert mechanism.secondary_metrics == (
        "perturbation_precision_spearman",
        "curvature_signal_mean_relative_error",
        "freezing_target_loss.none",
        "freezing_target_loss.freeze_low_sigma",
        "freezing_target_loss.freeze_high_sigma",
    )

    adaptive = next(
        job for job in SUITES["uncertainty_analysis"].jobs if job.job_id == "adaptive_timeline"
    )
    assert "change_detection.mean_detection_delay" not in adaptive.secondary_metrics
    assert {
        "change_detection.precision",
        "change_detection.recall",
        "change_detection.false_alarms_per_million_steps",
    } <= set(adaptive.secondary_metrics)


def _tiny_suite(
    primary_metric: str,
    *,
    seeds: tuple[int, ...] = (0,),
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
                seeds=seeds,
                algorithm="BGD-smoke",
                environment="quadratic",
                protocol="stationary",
                config_path=None,
                primary_metric=primary_metric,
                secondary_metrics=("initial_abs_mean",),
                comparison_group="tiny_smoke_group",
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
    assert metadata["comparison_group"] == "tiny_smoke_group"
    assert metadata["suite_revision"] == 3
    assert loaded.manifest.metadata["comparison_group"] == "tiny_smoke_group"
    assert loaded.manifest.metadata["suite_revision"] == 3
    config = yaml.safe_load(
        (run_dir / "config.yaml").read_text(
            encoding="utf-8",
        )
    )
    resolved = config["invocation"]["resolved_call_kwargs"]
    assert resolved == {
        "steps": 4,
        "seed": 0,
        "device": "cpu",
    }


def test_execute_suite_filters_one_seed_without_touching_other_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = _tiny_suite(
        "final_abs_mean",
        seeds=(
            0,
            1,
        ),
    )
    monkeypatch.setitem(
        SUITES,
        suite.name,
        suite,
    )

    result = execute_suite(
        suite.name,
        tmp_path,
        seeds=(1,),
    )

    assert result["status"] == "success"
    assert result["jobs_declared"] == 2
    assert result["jobs_selected"] == 1
    assert result["jobs_executed"] == 1
    assert result["selected_run_ids"] == ["smoke__tiny_smoke__seed_1"]
    assert result["selection"] == {
        "job_ids": [],
        "seeds": [1],
        "run_ids": [],
    }
    summary_path = Path(result["summary_path"])
    assert summary_path.parent.name == "execution_summaries"
    assert summary_path.is_file()

    seed_one = tmp_path / "smoke" / "smoke__tiny_smoke__seed_1"
    seed_zero = tmp_path / "smoke" / "smoke__tiny_smoke__seed_0"
    assert (seed_one / "run_metadata.json").is_file()
    assert not (seed_zero / "run_metadata.json").exists()


def test_execute_suite_filter_validation_is_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = _tiny_suite(
        "final_abs_mean",
        seeds=(
            0,
            1,
        ),
    )
    monkeypatch.setitem(
        SUITES,
        suite.name,
        suite,
    )

    with pytest.raises(
        KeyError,
        match="unknown suite seeds",
    ):
        execute_suite(
            suite.name,
            tmp_path,
            seeds=(2,),
        )

    with pytest.raises(
        KeyError,
        match="unknown suite job IDs",
    ):
        execute_suite(
            suite.name,
            tmp_path,
            job_ids=("missing_job",),
        )

    with pytest.raises(
        KeyError,
        match="unknown suite run IDs",
    ):
        execute_suite(
            suite.name,
            tmp_path,
            run_ids=("missing_run",),
        )


def test_execute_suite_resumes_matching_strict_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = _tiny_suite("final_abs_mean")
    monkeypatch.setitem(
        SUITES,
        suite.name,
        suite,
    )
    first = execute_suite(
        suite.name,
        tmp_path,
    )
    assert first["jobs_executed"] == 1
    assert first["jobs_skipped"] == 0

    run_dir = tmp_path / suite.name / "smoke__tiny_smoke__seed_0"
    stdout_path = run_dir / "stdout.json"
    stdout_path.write_text(
        "resume-sentinel",
        encoding="utf-8",
    )

    second = execute_suite(
        suite.name,
        tmp_path,
    )
    assert second["status"] == "success"
    assert second["jobs_executed"] == 0
    assert second["jobs_skipped"] == 1
    assert second["skipped_run_ids"] == ["smoke__tiny_smoke__seed_0"]
    assert stdout_path.read_text(encoding="utf-8") == "resume-sentinel"


def test_execute_suite_reruns_when_saved_contract_is_tampered(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = _tiny_suite("final_abs_mean")
    monkeypatch.setitem(
        SUITES,
        suite.name,
        suite,
    )
    execute_suite(
        suite.name,
        tmp_path,
    )
    run_dir = tmp_path / suite.name / "smoke__tiny_smoke__seed_0"
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["primary_metric"] = "tampered_metric"
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    stdout_path = run_dir / "stdout.json"
    stdout_path.write_text(
        "tampered-contract-sentinel",
        encoding="utf-8",
    )

    result = execute_suite(
        suite.name,
        tmp_path,
    )

    assert result["jobs_executed"] == 1
    assert result["jobs_skipped"] == 0
    assert stdout_path.read_text(encoding="utf-8") != "tampered-contract-sentinel"


def test_execute_suite_reruns_when_canonical_artifact_is_corrupted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = _tiny_suite("final_abs_mean")
    monkeypatch.setitem(
        SUITES,
        suite.name,
        suite,
    )
    execute_suite(
        suite.name,
        tmp_path,
    )
    run_dir = tmp_path / suite.name / "smoke__tiny_smoke__seed_0"
    (run_dir / "summary.json").write_text(
        "{not-valid-json",
        encoding="utf-8",
    )
    stdout_path = run_dir / "stdout.json"
    stdout_path.write_text(
        "corrupt-artifact-sentinel",
        encoding="utf-8",
    )

    result = execute_suite(
        suite.name,
        tmp_path,
    )

    assert result["jobs_executed"] == 1
    assert result["jobs_skipped"] == 0
    assert stdout_path.read_text(encoding="utf-8") != "corrupt-artifact-sentinel"


def test_execute_suite_can_force_rerun_of_matching_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = _tiny_suite("final_abs_mean")
    monkeypatch.setitem(
        SUITES,
        suite.name,
        suite,
    )
    execute_suite(
        suite.name,
        tmp_path,
    )
    run_dir = tmp_path / suite.name / "smoke__tiny_smoke__seed_0"
    stdout_path = run_dir / "stdout.json"
    stdout_path.write_text(
        "force-rerun-sentinel",
        encoding="utf-8",
    )

    result = execute_suite(
        suite.name,
        tmp_path,
        resume=False,
    )
    assert result["status"] == "success"
    assert result["jobs_executed"] == 1
    assert result["jobs_skipped"] == 0
    assert stdout_path.read_text(encoding="utf-8") != "force-rerun-sentinel"
    assert len(result["archived_runs"]) == 1
    archived = Path(result["archived_runs"][0]["path"])
    assert archived.parent.name == "smoke__tiny_smoke__seed_0"
    archived_run = load_run_directory(archived)
    assert archived_run.manifest.run_id == "smoke__tiny_smoke__seed_0"
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["supersedes_archive"] == str(archived)


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
    assert failed_manifest["metadata"]["comparison_group"] == "tiny_smoke_group"
    with pytest.raises(
        ValueError,
        match="refuses incomplete",
    ):
        load_run_directory(run_dir)
