import csv
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from simple_adp.runner import launcher
from simple_adp.runner.bundles import restore_results_bundle as real_restore_results_bundle


def test_launcher_applies_cuda_controls_before_first_torch_import(tmp_path):
    repository = Path(__file__).parents[1]
    observer = r'''
import builtins
import os
import sys
from simple_adp.runner import launcher
assert "torch" not in sys.modules
original_import = builtins.__import__
def observed_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:64"
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    return original_import(name, *args, **kwargs)
builtins.__import__ = observed_import
try:
    launcher.main(["--report", "--run-root", os.environ["TEST_RUN_ROOT"]])
except SystemExit as exc:
    assert exc.code == 2  # expected: the test run root has no persisted plan
else:
    raise AssertionError("report mode should reject a missing plan")
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(repository / "src"), str(repository)))
    env["SIMPLE_ADP_CUDA_VISIBLE_DEVICES"] = "7"
    env["SIMPLE_ADP_PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    env["SIMPLE_ADP_CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TEST_RUN_ROOT"] = str(tmp_path / "run")
    completed = subprocess.run(
        [sys.executable, "-c", observer],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "y", " TRUE ", "YeS"])
def test_launcher_truth_parser_recognizes_all_true_tokens(value):
    assert launcher._truthy(value, default=False) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "n", "", " FALSE ", "No"])
def test_launcher_truth_parser_recognizes_all_false_tokens(value):
    assert launcher._truthy(value, default=True) is False


@pytest.mark.parametrize("default", [False, True])
def test_launcher_truth_parser_uses_default_for_missing_or_unknown_values(default):
    assert launcher._truthy(None, default=default) is default
    assert launcher._truthy("unrecognized", default=default) is default


def test_launcher_preserves_nested_run_root_restore_guard_and_finally_export(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    results = repository / "results"
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(launcher, "configure_console_logging", lambda: None)
    bootstrap_environment = []
    runtime_keys = ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG")
    inherited_runtime_environment = {key: __import__("os").environ.get(key) for key in runtime_keys}
    for key in runtime_keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        launcher, "bootstrap_runtime",
        lambda *_args, **_kwargs: bootstrap_environment.append({
            key: __import__("os").environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG")
        }),
    )
    registry = SimpleNamespace(adapters={})
    monkeypatch.setattr(launcher, "load_registries", lambda: (registry, registry))
    monkeypatch.setattr(
        launcher.SuitePlan, "build",
        lambda *_args, **_kwargs: SimpleNamespace(run_signature="sig", manifest={"run_signature": "sig"}),
    )
    observed = []

    def restore(target, root):
        observed.append(("restore", list(target.iterdir())))
        return real_restore_results_bundle(target, root)

    monkeypatch.setattr(launcher, "restore_results_bundle", restore)
    monkeypatch.setattr(
        launcher, "rehydrate_sharded_telemetry",
        lambda target, *_args: observed.append(("rehydrate", Path(target).resolve())) or 0,
    )
    child_environments = []
    monkeypatch.setattr(launcher, "launcher_child_env", lambda env, **_kwargs: dict(env))

    def run_controller(*_args, **kwargs):
        child_environments.append(kwargs["env"])
        observed.append(("controller", None))
        return 7

    monkeypatch.setattr(launcher, "_run_with_watchdog", run_controller)
    monkeypatch.setattr(
        launcher, "export_results_bundle",
        lambda target, root: observed.append(("export", Path(target).resolve())),
    )

    assert launcher.main([
        "--output", str(results), "--dry-run",
        "--set", 'cuda_visible_devices="2"',
        "--set", 'pytorch_cuda_alloc_conf="custom_alloc"',
        "--set", 'cublas_workspace_config=":16:8"',
    ]) == 7
    expected_runtime_defaults = {
        "CUDA_VISIBLE_DEVICES": "2",
        "PYTORCH_CUDA_ALLOC_CONF": "custom_alloc",
        "CUBLAS_WORKSPACE_CONFIG": ":16:8",
    }
    assert bootstrap_environment == [expected_runtime_defaults]
    assert child_environments[0]["CUDA_VISIBLE_DEVICES"] == "2"
    assert child_environments[0]["PYTORCH_CUDA_ALLOC_CONF"] == "custom_alloc"
    assert child_environments[0]["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert [item[0] for item in observed] == ["restore", "rehydrate", "controller", "export"]
    assert observed[1][1] == results.resolve()
    # Creating the default nested run_root first makes results nonempty, so the
    # ledger's empty-directory restore guard rejects automatic restoration.
    assert len(observed[0][1]) == 1
    assert observed[0][1][0].name == "run_root"
    assert (results / "run_root" / "suite_launch_manifest.json").is_file()
    # launcher.main intentionally applies runtime defaults to its process
    # environment. This test calls it in-process, so restore the caller's
    # environment instead of leaking its sentinel values into later workers.
    environment = __import__("os").environ
    for key, value in inherited_runtime_environment.items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value


def test_launcher_disables_bundle_restore_and_export_but_still_rehydrates_telemetry(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    results = repository / "results"
    events = []
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(launcher, "configure_console_logging", lambda: None)
    monkeypatch.setattr(launcher, "bootstrap_runtime", lambda *_args, **_kwargs: None)
    registry = SimpleNamespace(adapters={})
    monkeypatch.setattr(launcher, "load_registries", lambda: (registry, registry))
    monkeypatch.setattr(
        launcher.SuitePlan, "build",
        lambda *_args, **_kwargs: SimpleNamespace(
            run_signature="sig", manifest={"run_signature": "sig"},
        ),
    )
    monkeypatch.setattr(
        launcher, "restore_results_bundle",
        lambda *_args: events.append("restore"),
    )
    monkeypatch.setattr(
        launcher, "rehydrate_sharded_telemetry",
        lambda target, *_args: events.append(("rehydrate", Path(target).resolve())) or 0,
    )
    monkeypatch.setattr(launcher, "launcher_child_env", lambda env, **_kwargs: dict(env))
    monkeypatch.setattr(
        launcher, "_run_with_watchdog",
        lambda *_args, **_kwargs: events.append("controller") or 7,
    )
    monkeypatch.setattr(
        launcher, "export_results_bundle",
        lambda *_args: events.append("export"),
    )

    assert launcher.main([
        "--output", str(results), "--dry-run", "--set", "auto_resume_bundle=false",
    ]) == 7
    assert events == [("rehydrate", results.resolve()), "controller"]


def test_launcher_rehydration_error_propagates_before_controller_even_when_bundles_disabled(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    results = repository / "results"
    events = []
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(launcher, "configure_console_logging", lambda: None)
    monkeypatch.setattr(launcher, "bootstrap_runtime", lambda *_args, **_kwargs: None)
    registry = SimpleNamespace(adapters={})
    monkeypatch.setattr(launcher, "load_registries", lambda: (registry, registry))
    monkeypatch.setattr(
        launcher, "restore_results_bundle",
        lambda *_args: events.append("restore"),
    )

    def fail_rehydration(target, *_args):
        events.append(("rehydrate", Path(target).resolve()))
        raise FileNotFoundError("telemetry shard missing")

    monkeypatch.setattr(launcher, "rehydrate_sharded_telemetry", fail_rehydration)
    monkeypatch.setattr(
        launcher, "_run_with_watchdog",
        lambda *_args, **_kwargs: pytest.fail("controller must not start after rehydration failure"),
    )
    monkeypatch.setattr(
        launcher, "export_results_bundle",
        lambda *_args: pytest.fail("setup failure occurs before the export finally"),
    )

    with pytest.raises(FileNotFoundError, match="telemetry shard missing"):
        launcher.main([
            "--output", str(results), "--dry-run", "--set", "auto_resume_bundle=false",
        ])
    assert events == [("rehydrate", results.resolve())]


@pytest.mark.parametrize(
    ("field", "environment_key", "configured", "inherited", "expected"),
    [
        ("cuda_visible_devices", "CUDA_VISIBLE_DEVICES", "2,3", None, "2,3"),
        ("cuda_visible_devices", "CUDA_VISIBLE_DEVICES", "2,3", "", "2,3"),
        ("cuda_visible_devices", "CUDA_VISIBLE_DEVICES", "2,3", "5", "5"),
        ("cuda_visible_devices", "CUDA_VISIBLE_DEVICES", "", None, "0"),
        ("pytorch_cuda_alloc_conf", "PYTORCH_CUDA_ALLOC_CONF", "custom_alloc", None, "custom_alloc"),
        ("pytorch_cuda_alloc_conf", "PYTORCH_CUDA_ALLOC_CONF", "custom_alloc", "inherited_alloc", "inherited_alloc"),
        ("pytorch_cuda_alloc_conf", "PYTORCH_CUDA_ALLOC_CONF", "", None, "expandable_segments:True,max_split_size_mb:128"),
        ("cublas_workspace_config", "CUBLAS_WORKSPACE_CONFIG", ":16:8", None, ":16:8"),
        ("cublas_workspace_config", "CUBLAS_WORKSPACE_CONFIG", ":16:8", "inherited", "inherited"),
        ("cublas_workspace_config", "CUBLAS_WORKSPACE_CONFIG", "", None, ":4096:8"),
    ],
)
def test_launcher_runtime_environment_defaults_preserve_inherited_values(
    monkeypatch, field, environment_key, configured, inherited, expected
):
    if inherited is None:
        monkeypatch.delenv(environment_key, raising=False)
    else:
        monkeypatch.setenv(environment_key, inherited)

    launcher._apply_runtime_environment_defaults(launcher.RunnerSettings(**{field: configured}))

    assert __import__("os").environ[environment_key] == expected


def test_launcher_settings_precedence_preserves_explicit_zero(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"concurrency": 2, "batch_size": 16, "cpu_only": False}))
    monkeypatch.setenv("SIMPLE_ADP_concurrency", "1")
    monkeypatch.setenv("SIMPLE_ADP_batch_size", "8")
    parser = launcher._argument_parser()
    args = parser.parse_args([
        "--settings-json", str(settings_file),
        "--set", "concurrency=3",
        "--set", "batch_size=32",
        "--set", "cpu_only=true",
        "--set", "ex_k_width=3",
        "--set", "learning_rate=0.002",
        "--concurrency", "0",
        "--batch-size", "64",
    ])
    settings = launcher._settings(args, parser)
    assert settings.concurrency == 0  # direct CLI zero wins over all earlier sources
    assert settings.batch_size == 64
    assert settings.cpu_only is True
    assert settings.ex_k_width == 3
    assert settings.learning_rate == 0.002


def test_launcher_resolves_custom_provider_suites_for_listing_and_plan(tmp_path, monkeypatch, capsys):
    from simple_adp.runner.datasets import DatasetRegistry, DatasetTaskRef

    class TinyProvider:
        name = "tiny"

        def __init__(self):
            self.calls = []

        def list_tasks(self, suite):
            self.calls.append(suite)
            return [DatasetTaskRef(self.name, suite, "task-7")]

    provider = TinyProvider()
    datasets = DatasetRegistry()
    datasets.register(provider)
    models = SimpleNamespace(adapters={"test-model": object()})
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(launcher, "configure_console_logging", lambda: None)
    monkeypatch.setattr(launcher, "bootstrap_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "load_registries", lambda: (datasets, models))

    assert launcher.main(["--list-suites", "--dataset-suites", "tiny:demo"]) == 0
    assert json.loads(capsys.readouterr().out) == {"tiny:demo": ["task-7"]}
    assert provider.calls == ["demo"]

    observed = {}

    def build_plan(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(run_signature="sig", manifest={"run_signature": "sig"})

    monkeypatch.setattr(launcher.SuitePlan, "build", build_plan)
    monkeypatch.setattr(launcher, "restore_results_bundle", lambda *_args: False)
    monkeypatch.setattr(launcher, "rehydrate_sharded_telemetry", lambda *_args: 0)
    monkeypatch.setattr(launcher, "launcher_child_env", lambda env, **_kwargs: dict(env))
    monkeypatch.setattr(launcher, "_run_with_watchdog", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(launcher, "export_results_bundle", lambda *_args: None)

    results = repository / "results"
    assert launcher.main([
        "--dataset-suites", "tiny:demo", "--dry-run", "--output", str(results),
    ]) == 0
    assert observed["dataset_suites"] == (("tiny", "demo"),)
    assert observed["dataset_providers"] == (provider,)


def test_dataset_suite_selector_keeps_openml_bare_names_and_accepts_mixed_providers():
    assert launcher._dataset_suites("cc18, tiny:demo") == (
        ("openml", "cc18"), ("tiny", "demo"),
    )
    for value in ("", ",,,", "tiny:", ":demo"):
        try:
            launcher._dataset_suites(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid dataset suite selector: {value!r}")


def test_explicit_resume_reuses_verified_persisted_plan(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    results = repository / "results"
    run_root = results / "run_root"
    run_root.mkdir(parents=True)
    settings = launcher.RunnerSettings(results_dir=str(results), run_root=str(run_root))
    plan_path = run_root / "suite_plan.json"
    plan_path.write_text(json.dumps({
        "run_signature": "persisted-signature",
        "runner_settings": settings.as_dict(),
        "providers_and_suites": [["openml", "cc18"]],
        "model_adapters": ["simple_adp"],
        "results_dir": str(results.resolve()),
        "run_root": str(run_root.resolve()),
        "jobs": [],
    }))
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(launcher, "configure_console_logging", lambda: None)
    monkeypatch.setattr(launcher, "bootstrap_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "load_registries", lambda: (
        SimpleNamespace(adapters={"openml": object()}),
        SimpleNamespace(adapters={"simple_adp": object()}),
    ))
    monkeypatch.setattr(launcher, "restore_results_bundle", lambda *_args: False)
    monkeypatch.setattr(launcher, "rehydrate_sharded_telemetry", lambda *_args: 0)
    monkeypatch.setattr(launcher, "launcher_child_env", lambda env, **_kwargs: dict(env))
    monkeypatch.setattr(launcher.SuitePlan, "build", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume must not rebuild plan")))
    calls = []
    monkeypatch.setattr(launcher, "_run_with_watchdog", lambda command, **_kwargs: calls.append(command) or 0)
    monkeypatch.setattr(launcher, "export_results_bundle", lambda *_args: None)

    assert launcher.main([
        "--output", str(results), "--run-root", str(run_root),
        "--dataset-suites", "cc18", "--resume",
    ]) == 0
    assert len(calls) == 1
    assert (run_root / "suite_plan.json").is_file()


def test_custom_provider_suite_selector_is_validated_on_resume_with_provider_registry(tmp_path, monkeypatch):
    from simple_adp.runner.datasets import DatasetRegistry

    class TinyProvider:
        name = "tiny"

        def list_tasks(self, suite):
            raise AssertionError("resume must not rebuild or reload dataset tasks")

    datasets = DatasetRegistry()
    datasets.register(TinyProvider())
    models = SimpleNamespace(adapters={"test-model": object()})
    repository = tmp_path / "repo"
    repository.mkdir()
    results = repository / "results"
    run_root = results / "run_root"
    run_root.mkdir(parents=True)
    settings = launcher.RunnerSettings(results_dir=str(results), run_root=str(run_root))
    plan_path = run_root / "suite_plan.json"
    plan_path.write_text(json.dumps({
        "run_signature": "persisted-custom-signature",
        "runner_settings": settings.as_dict(),
        "providers_and_suites": [["tiny", "demo"]],
        "model_adapters": ["test-model"],
        "results_dir": str(results.resolve()),
        "run_root": str(run_root.resolve()),
        "jobs": [],
    }))
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(launcher, "configure_console_logging", lambda: None)
    monkeypatch.setattr(launcher, "bootstrap_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "load_registries", lambda: (datasets, models))
    monkeypatch.setattr(launcher, "restore_results_bundle", lambda *_args: False)
    monkeypatch.setattr(launcher, "rehydrate_sharded_telemetry", lambda *_args: 0)
    monkeypatch.setattr(launcher, "launcher_child_env", lambda env, **_kwargs: dict(env))
    monkeypatch.setattr(
        launcher.SuitePlan, "build",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume must use the saved plan")),
    )
    commands = []
    monkeypatch.setattr(launcher, "_run_with_watchdog", lambda command, **_kwargs: commands.append(command) or 0)
    monkeypatch.setattr(launcher, "export_results_bundle", lambda *_args: None)

    assert launcher.main([
        "--output", str(results), "--run-root", str(run_root),
        "--dataset-suites", "tiny:demo", "--resume",
    ]) == 0
    assert len(commands) == 1


def test_report_mode_uses_saved_job_state_without_running_children(tmp_path):
    from simple_adp.runner.scheduler import SuiteCoordinator

    result_dir = tmp_path / "results" / "route"
    run_id = "provider/suite/task-1/route"
    spec = SimpleNamespace(
        identity=SimpleNamespace(run_id=run_id),
        result_dir=str(result_dir),
    )
    plan = SimpleNamespace(run_signature="sig", runs=(spec,))
    coordinator = SuiteCoordinator(
        plan, launcher.RunnerSettings(run_hardness_screen=False),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    state_path = tmp_path / "run" / "jobs" / run_id / "job_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({
        "job_id": run_id, "status": "completed", "exit_code": 0,
        "attempt": 1, "retry_count": 2, "result_dir": str(result_dir),
    }))

    summary = coordinator.write_persisted_state_report()
    assert summary["report_mode"] == "persisted_state_snapshot"
    assert summary["completed"] == 1
    assert summary["terminal_failures"] == 0
    assert summary["retries"] == 2
    assert summary["failed_attempts"] is None
    assert summary["status_counts"]["completed"] == 1
    assert (tmp_path / "run" / "final_report.json").is_file()
    with (tmp_path / "run" / "task_summary.csv").open() as stream:
        assert run_id in stream.read()


def test_persisted_report_covers_all_fields_status_normalization_and_telemetry(tmp_path):
    from simple_adp.runner.scheduler import SuiteCoordinator

    run_root = tmp_path / "run"
    result_root = tmp_path / "results"
    route_states = {
        "success": {"status": "completed", "exit_code": 0, "retry_count": "2",
                    "attempt": 3, "last_exit_code": 1, "history_rows": 4, "final_epoch": "3"},
        "nonzero-completed": {"status": "completed", "exit_code": 1, "retry_count": -5},
        "launching": {"status": "launching", "retry_count": "invalid"},
        "unknown": {"status": "surprise", "retry_count": 1},
        "paused": {"status": "paused", "retry_count": 0},
    }
    specs = []
    for name, state in route_states.items():
        run_id = f"provider/suite/{name}"
        result_dir = result_root / name
        specs.append(SimpleNamespace(
            identity=SimpleNamespace(run_id=run_id), result_dir=str(result_dir),
        ))
        state_path = run_root / "jobs" / run_id / "job_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")
    missing_id = "provider/suite/no-state"
    specs.append(SimpleNamespace(
        identity=SimpleNamespace(run_id=missing_id),
        result_dir=str(result_root / "no-state"),
    ))

    paused_progress = result_root / "paused" / "telemetry" / "stage" / "progress_state.json"
    paused_progress.parent.mkdir(parents=True)
    paused_progress.write_text(json.dumps({"history_rows": 2, "final_epoch": 1}), encoding="utf-8")
    run_root.mkdir(parents=True, exist_ok=True)
    saved_resources = {"host_available_mib": 777, "gpu_free_mib": None}
    (run_root / "status.json").write_text(json.dumps({"resource_snapshot": saved_resources}), encoding="utf-8")
    plan = SimpleNamespace(run_signature="signature", runs=tuple(specs))
    coordinator = SuiteCoordinator(
        plan, launcher.RunnerSettings(run_hardness_screen=False), run_root=run_root,
        results_dir=result_root, repository_root=tmp_path,
    )

    summary = coordinator.write_persisted_state_report()

    expected_statuses = {
        "queued": 2, "launching": 0, "running": 1, "paused": 1,
        "retrying": 0, "completed": 2, "failed": 0,
    }
    assert set(summary) == {
        "report_mode", "run_root", "results_dir", "run_signature", "manifest_jobs",
        "jobs_with_state", "status_counts", "completed", "terminal_failures",
        "failed_attempts", "retries", "recovered_unknown_exit", "timestamp",
        "resource_snapshot", "batches_done", "epochs_done", "live_batches_done",
        "live_epochs_done", "telemetry_jobs", "telemetry_dirs",
    }
    assert summary["report_mode"] == "persisted_state_snapshot"
    assert summary["manifest_jobs"] == 6 and summary["jobs_with_state"] == 5
    assert summary["status_counts"] == expected_statuses
    assert summary["completed"] == 1 and summary["terminal_failures"] == 0
    assert summary["failed_attempts"] is None and summary["recovered_unknown_exit"] is None
    assert summary["retries"] == 3
    assert summary["resource_snapshot"] == saved_resources
    assert (summary["batches_done"], summary["epochs_done"]) == (6, 4)
    assert (summary["live_batches_done"], summary["live_epochs_done"]) == (2, 1)
    # The state-field fallback increments telemetry_jobs but leaves
    # telemetry_dirs unchanged; the sidecar path increments both.
    assert (summary["telemetry_jobs"], summary["telemetry_dirs"]) == (2, 1)

    for name in ("final_report.json", "suite_summary.json"):
        assert json.loads((run_root / name).read_text(encoding="utf-8")) == summary
    final_markdown = (run_root / "final_report.md").read_text(encoding="utf-8")
    suite_markdown = (run_root / "suite_summary.md").read_text(encoding="utf-8")
    for text in (
        "- Report mode: persisted state snapshot", "- Total jobs: 6",
        "- Completed: 1", "- Terminal failed: 0", "- Failed attempts: None",
        "- Retries recorded in job state: 3", "- Batches done: 6", "- Epochs done: 4",
        "- Live-job batches done: 2", "- Live-job epochs done: 1",
    ):
        assert text in final_markdown and text in suite_markdown
    assert "- Timestamp:" in final_markdown and "- Timestamp:" not in suite_markdown

    with (run_root / "task_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["job_id", "status", "attempt", "retry_count", "exit_code", "last_exit_code", "result_dir"]
    assert rows[1] == ["provider/suite/success", "completed", "3", "2", "0", "1", str(result_root / "success")]
    assert rows[2] == ["provider/suite/nonzero-completed", "completed", "", "-5", "1", "", str(result_root / "nonzero-completed")]
    assert rows[3] == ["provider/suite/launching", "launching", "", "invalid", "", "", str(result_root / "launching")]
    assert rows[4] == ["provider/suite/unknown", "surprise", "", "1", "", "", str(result_root / "unknown")]
    assert rows[5] == ["provider/suite/paused", "paused", "", "0", "", "", str(result_root / "paused")]
    assert rows[6] == ["provider/suite/no-state", "queued", "", "", "", "", str(result_root / "no-state")]


def test_report_cli_rebuilds_reports_from_saved_plan_without_controller(tmp_path, monkeypatch, capsys):
    repository = tmp_path / "repo"
    repository.mkdir()
    results = repository / "results"
    run_root = results / "run_root"
    run_root.mkdir(parents=True)
    settings = launcher.RunnerSettings(
        results_dir=str(results), run_root=str(run_root), run_hardness_screen=False,
    )
    (run_root / "suite_plan.json").write_text(json.dumps({
        "run_signature": "report-signature",
        "runner_settings": settings.as_dict(),
        "results_dir": str(results.resolve()),
        "run_root": str(run_root.resolve()),
        "jobs": [],
    }))
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(launcher, "configure_console_logging", lambda: None)
    monkeypatch.setattr(launcher, "bootstrap_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "load_registries", lambda: (SimpleNamespace(adapters={}), SimpleNamespace(adapters={})))
    monkeypatch.setattr(launcher, "_run_with_watchdog", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("report mode must not launch controller")))

    assert launcher.main([
        "--output", str(results), "--run-root", str(run_root),
        "--dataset-suites", "cc18", "--report",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["report_mode"] == "persisted_state_snapshot"
    assert (run_root / "final_report.json").is_file()


def test_detach_uses_results_run_root_and_leaves_started_child_on_pid_write_error(tmp_path, monkeypatch, capsys):
    results = tmp_path / "results"
    custom_run_root = tmp_path / "custom-run-root"
    process_calls = []

    class Spawned:
        pid = 4321
        terminated = False

        def terminate(self):
            self.terminated = True

    child = Spawned()

    def popen(command, **kwargs):
        process_calls.append((command, kwargs))
        return child

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path / "repo")
    monkeypatch.setattr(
        launcher.sys, "argv",
        ["simple-adp-suite", "--output", str(results), "--run-root", str(custom_run_root), "--detach"],
    )
    original_open = Path.open

    def fail_pid_open(path, *args, **kwargs):
        if path.name == "launcher.pid":
            raise OSError("synthetic PID file failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_pid_open)
    settings = launcher.RunnerSettings(results_dir=str(results), run_root=str(custom_run_root))

    assert launcher._detach(SimpleNamespace(), settings) == 0

    command, kwargs = process_calls[0]
    expected_root = results / "run_root"
    assert command == [launcher.sys.executable, "-u", "-m", "simple_adp.runner.launcher", "--output", str(results), "--run-root", str(custom_run_root)]
    assert kwargs["cwd"] == tmp_path / "repo"
    assert kwargs["stdin"] is launcher.subprocess.DEVNULL
    assert kwargs["start_new_session"] is True and kwargs["close_fds"] is True
    assert expected_root.joinpath("launcher.log").is_file()
    assert child.terminated is False  # PID-file failure does not kill detached Popen.
    assert "PID file failed" in capsys.readouterr().err


def test_watchdog_launcher_waits_for_child_and_kills_watcher_after_timeout(tmp_path, monkeypatch):
    events = []

    class Child:
        pid = 1234

        def wait(self):
            events.append("child_wait")
            return 7

    class Watchdog:
        def terminate(self):
            events.append("watchdog_terminate")

        def wait(self, *, timeout):
            events.append(("watchdog_wait", timeout))
            raise launcher.subprocess.TimeoutExpired("watchdog", timeout)

        def kill(self):
            events.append("watchdog_kill")

    spawned = []

    def popen(command, **kwargs):
        spawned.append((command, kwargs))
        return Child() if len(spawned) == 1 else Watchdog()

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.os, "getpid", lambda: 5678)

    code = launcher._run_with_watchdog(
        ["controller", "--plan", "plan.json"], cwd=tmp_path,
        env={"TASK": "toy"}, log_path=tmp_path / "watchdog.log",
    )

    assert code == 7
    assert spawned[0] == (
        ["controller", "--plan", "plan.json"],
        {"cwd": tmp_path, "env": {"TASK": "toy"}},
    )
    watchdog_command, watchdog_kwargs = spawned[1]
    assert watchdog_command == [
        launcher.sys.executable, "-u", "-m", "simple_adp.runner.watchdog",
        "--master", "5678", "--child", "1234", "--log", str(tmp_path / "watchdog.log"),
    ]
    assert watchdog_kwargs["start_new_session"] is True
    assert watchdog_kwargs["stdin"] is launcher.subprocess.DEVNULL
    assert watchdog_kwargs["stdout"] is launcher.subprocess.DEVNULL
    assert watchdog_kwargs["stderr"] is launcher.subprocess.DEVNULL
    assert events == [
        "child_wait", "watchdog_terminate", ("watchdog_wait", 2.0), "watchdog_kill",
    ]


@pytest.mark.parametrize(("exit_code", "remove_log"), [(0, True), (9, False)])
def test_watchdog_launcher_removes_log_only_after_success(tmp_path, monkeypatch, exit_code, remove_log):
    log_path = tmp_path / "watchdog.log"
    log_path.write_text("watchdog output\n")

    class Child:
        pid = 1234

        def wait(self):
            return exit_code

    class Watchdog:
        def terminate(self):
            pass

        def wait(self, *, timeout):
            return 0

    children = iter((Child(), Watchdog()))
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: next(children))
    assert launcher._run_with_watchdog(
        ["controller"], cwd=tmp_path, env={}, log_path=log_path,
    ) == exit_code
    assert log_path.exists() is not remove_log


def test_watchdog_launcher_terminates_live_controller_when_wait_is_interrupted(tmp_path, monkeypatch):
    events = []

    class Child:
        pid = 4321

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait():
            events.append("child_wait")
            raise KeyboardInterrupt

    class Watchdog:
        @staticmethod
        def terminate():
            events.append("watchdog_terminate")

        @staticmethod
        def wait(*, timeout):
            events.append(("watchdog_wait", timeout))

    spawned = 0

    def popen(*_args, **_kwargs):
        nonlocal spawned
        spawned += 1
        return Child() if spawned == 1 else Watchdog()

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.os, "getpid", lambda: 8765)
    monkeypatch.setattr(
        launcher, "terminate_process_tree",
        lambda child: events.append(("terminate_child_tree", child.pid)),
    )

    with pytest.raises(KeyboardInterrupt):
        launcher._run_with_watchdog(
            ["controller"], cwd=tmp_path, env={}, log_path=tmp_path / "watchdog.log",
        )

    assert events == [
        "child_wait", ("terminate_child_tree", 4321), "watchdog_terminate",
        ("watchdog_wait", 2.0),
    ]


def test_watchdog_start_failure_terminates_controller_tree(tmp_path, monkeypatch):
    events = []

    class Child:
        pid = 4322

        @staticmethod
        def poll():
            return None

    calls = 0

    def popen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Child()
        raise OSError("watchdog unavailable")

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.os, "getpid", lambda: 8765)
    monkeypatch.setattr(
        launcher, "terminate_process_tree",
        lambda child: events.append(("terminate_child_tree", child.pid)),
    )

    with pytest.raises(OSError, match="watchdog unavailable"):
        launcher._run_with_watchdog(
            ["controller"], cwd=tmp_path, env={}, log_path=tmp_path / "watchdog.log",
        )
    assert events == [("terminate_child_tree", 4322)]


def test_watchdog_launcher_installs_and_runs_shell_equivalent_signal_cleanup(tmp_path, monkeypatch):
    import signal

    events = []
    handlers = {}

    def set_handler(signum, handler):
        events.append(("signal_handler", signum, handler))
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    class Child:
        pid = 4323

        def poll(self):
            return None

        def wait(self):
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    class Watchdog:
        def kill(self):
            events.append("watchdog_kill")

    spawned = iter((Child(), Watchdog()))
    monkeypatch.setattr(launcher.signal, "signal", set_handler)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: next(spawned))
    monkeypatch.setattr(
        launcher, "terminate_process_tree_after_grace",
        lambda pid, *, grace_seconds: events.append(("tree_cleanup", pid, grace_seconds)),
    )

    with pytest.raises(SystemExit) as exc_info:
        launcher._run_with_watchdog(
            ["controller"], cwd=tmp_path, env={}, log_path=tmp_path / "watchdog.log",
        )

    assert exc_info.value.code == 128 + signal.SIGTERM
    assert ("tree_cleanup", 4323, 2.0) in events
    assert events.index(("tree_cleanup", 4323, 2.0)) < events.index("watchdog_kill")
    registrations = [entry[1] for entry in events if isinstance(entry, tuple) and entry[0] == "signal_handler"]
    assert registrations[:4] == [
        signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT,
    ]
    assert registrations[4:] == registrations[:4]


def test_linux_watchdog_signal_cleans_real_controller_process_tree(tmp_path):
    import signal
    import time

    import psutil

    if os.name != "posix":
        pytest.skip("this signal integration contract targets Linux/POSIX")

    repository = Path(__file__).parents[1]
    grandchild_pid_path = tmp_path / "grandchild.pid"
    child_code = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        f"pathlib.Path({str(grandchild_pid_path)!r}).write_text(str(p.pid)); "
        "time.sleep(120)"
    )
    parent_code = (
        "import os,sys; from pathlib import Path; "
        "from simple_adp.runner.launcher import _run_with_watchdog; "
        f"_run_with_watchdog([sys.executable,'-c',{child_code!r}], "
        f"cwd=Path({str(repository)!r}), env=os.environ.copy(), "
        f"log_path=Path({str(tmp_path / 'watchdog.log')!r}))"
    )
    environment = os.environ.copy()
    src_path = str(repository / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (src_path, environment.get("PYTHONPATH", "")) if item
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], cwd=repository, env=environment,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    grandchild_pid = None
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not grandchild_pid_path.exists():
            if parent.poll() is not None:
                pytest.fail("watchdog parent exited before starting its controller")
            time.sleep(0.02)
        assert grandchild_pid_path.exists(), "controller did not start its grandchild"
        grandchild_pid = int(grandchild_pid_path.read_text())
        os.kill(parent.pid, signal.SIGTERM)
        assert parent.wait(timeout=8) == 128 + signal.SIGTERM
        try:
            grandchild = psutil.Process(grandchild_pid)
            assert not grandchild.is_running() or grandchild.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
        if grandchild_pid is not None:
            try:
                grandchild = psutil.Process(grandchild_pid)
                if grandchild.is_running() and grandchild.status() != psutil.STATUS_ZOMBIE:
                    grandchild.kill()
            except psutil.Error:
                pass
        if parent.stderr is not None:
            parent.stderr.close()
