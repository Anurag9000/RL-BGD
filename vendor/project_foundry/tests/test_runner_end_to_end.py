import json
import os
import time
import csv
import hashlib
from pathlib import Path

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.resources import HostMemorySample
from simple_adp.runner.scheduler import SuiteCoordinator


def test_coordinator_worker_process_runs_plugin_dataset_through_builtin_adp_routes(
    tmp_path, monkeypatch,
):
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    (plugin_root / "suite_fixture.py").write_text(
        """
import numpy as np
import hashlib
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.plan import SplitSpec

class Data:
    def __init__(self):
        self.X = np.asarray([[i / 5, (i % 3) / 3] for i in range(12)], dtype=np.float32)
        self.y = np.asarray([0, 1] * 6)
        self.task_kind = 'classification'
        self.dataset_id = 'plugin-toy'
        self.content_sha256 = 'plugin-toy-content'
        self.feature_schema = None
        self.name = 'plugin-toy'
        self.target_name = 'label'

class Provider:
    name = 'plugin_fixture'
    def list_tasks(self, suite):
        return [DatasetTaskRef(self.name, suite, 'task-1')]
    def load_task(self, ref):
        return Data()
    def list_splits(self, ref, data, settings):
        train, validation, test = tuple(range(8)), (8, 9), (10, 11)
        digest = hashlib.sha256()
        for indices in (train, validation, test):
            digest.update(np.asarray(indices, dtype=np.int64).tobytes())
        return [SplitSpec(0, 0, 0, train, validation, test, digest.hexdigest())]

class PluginModelAdapter:
    name = 'plugin_model'
    def enumerate_routes(self, generic_profiles, settings=None):
        from simple_adp.runner.models import ModelRoute
        return [ModelRoute('PLUGIN-MLP', suite_group='width_only', init_width=1, init_depth=1)]
    def fit_predict(self, route, **kwargs):
        from simple_adp.runner.models import ModelRoute, SimpleADPModelAdapter
        delegated = ModelRoute('ADP-W-All', init_width=route.init_width, init_depth=route.init_depth)
        return SimpleADPModelAdapter().fit_predict(delegated, **kwargs)
""",
        encoding="utf-8",
    )
    dist_info = plugin_root / "suite_fixture-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "entry_points.txt").write_text(
        "[simple_adp.dataset_providers]\nfixture = suite_fixture:Provider\n"
        "[simple_adp.model_adapters]\nfixture = suite_fixture:PluginModelAdapter\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(plugin_root))
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(part for part in (str(plugin_root), inherited_pythonpath) if part),
    )
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")

    from simple_adp.runner.plugins import load_registries

    dataset_registry, model_registry = load_registries()
    settings = RunnerSettings(
        concurrency=1,
        cpu_only=True,
        max_retries=1,
        max_width=1,
        max_depth=1,
        max_neurons=32,
        max_epochs_adp=9,
        smoke_test=True,
        model_seeds=(5,),
        generic_profiles=("faithful",),
        tasks=("width_only",),
        include_vanilla=False,
        include_width_to_depth=False,
        include_depth_to_width=False,
        include_depth_only=False,
        width_only_depths=(1,),
        width_only_width=1,
        width_to_depth_width=1,
        width_to_depth_widths=(1,),
        depth_only_widths=(1,),
        vanilla_depths=(1,),
        vanilla_widths=(1,),
        screen_widths=(1,),
        screen_depths=(1,),
        deeper_screen_widths=(1,),
        run_hardness_screen=False,
        repeat_count=1,
        missing_present_task_repeats=1,
        post_launch_sample_delay_seconds=0,
        pressure_poll_seconds=0.01,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=tmp_path / "results",
        run_root=tmp_path / "run",
        dataset_providers=(dataset_registry.get("plugin_fixture"),),
        dataset_suites=(("plugin_fixture", "toy"),),
        model_adapters=(
            model_registry.get("simple_adp"),
            model_registry.get("plugin_model"),
        ),
    )
    assert {run.route.model_id for run in plan.runs} == {
        "LGND-W-Greedy", "ADP-W-RR", "ADP-W-All", "PLUGIN-MLP",
    }

    coordinator = SuiteCoordinator(
        plan,
        settings,
        run_root=tmp_path / "run",
        results_dir=tmp_path / "results",
        repository_root=Path(__file__).resolve().parents[1],
        host_sampler=lambda: HostMemorySample(8192, 8192, 0, 0, 0, 0),
        gpu_sampler=lambda _device: None,
        poll_sleep=lambda duration: time.sleep(min(duration, 0.02)),
    )
    summary = coordinator.run()

    assert summary["completed"] == 4
    records = [json.loads(Path(run.result_dir, "result.json").read_text()) for run in plan.runs]
    assert {record["model_id"] for record in records} == {
        "LGND-W-Greedy", "ADP-W-RR", "ADP-W-All", "PLUGIN-MLP",
    }
    assert all(record["status"] == "completed" for record in records)
    assert all(record["smoke_test"] is True for record in records)
    assert all(record["history_entries"] >= 1 for record in records)
    assert all(record["split_sha256"] == plan.runs[0].identity.split_sha256 for record in records)


def test_pressure_kill_relaunches_real_worker_from_estimator_checkpoint(tmp_path, monkeypatch):
    plugin_root = tmp_path / "resume-plugin"
    plugin_root.mkdir()
    (plugin_root / "resume_fixture.py").write_text(
        """
import hashlib
import numpy as np
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.plan import SplitSpec

class Data:
    def __init__(self):
        self.X = np.asarray([[i / 10, (i % 3) / 3] for i in range(24)], dtype=np.float32)
        self.y = np.asarray([0, 1] * 12)
        self.task_kind = 'classification'
        self.dataset_id = 'pressure-resume-toy'
        self.content_sha256 = 'pressure-resume-toy-content'
        self.feature_schema = None
        self.name = 'pressure-resume-toy'
        self.target_name = 'label'

class Provider:
    name = 'resume_fixture'
    def list_tasks(self, suite):
        return [DatasetTaskRef(self.name, suite, 'task-1')]
    def load_task(self, ref):
        return Data()
    def list_splits(self, ref, data, settings):
        train, validation, test = tuple(range(12)), (12, 13, 14, 15, 16, 17), (18, 19, 20, 21, 22, 23)
        digest = hashlib.sha256()
        for indices in (train, validation, test):
            digest.update(np.asarray(indices, dtype=np.int64).tobytes())
        return [SplitSpec(0, 0, 0, train, validation, test, digest.hexdigest())]

class ResumableModelAdapter:
    name = 'resumable_fixture'
    def enumerate_routes(self, generic_profiles, settings=None):
        from simple_adp.runner.models import ModelRoute
        return [ModelRoute('PLUGIN-RESUMABLE', suite_group='width_only', init_width=1, init_depth=1)]
    def fit_predict(self, route, **kwargs):
        import torch
        import time
        from pathlib import Path
        from simple_adp.runner.models import ModelRoute, SimpleADPModelAdapter
        checkpoint = Path(kwargs['checkpoint_path'])
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            cursor = int(payload['payload']['stage_state']['batch_cursor'])
            checkpoint.with_name('resumed_batch_cursor.txt').write_text(str(cursor))
        progress_callback = kwargs.get('progress_callback')
        if progress_callback is not None:
            def slow_progress(state):
                progress_callback(state)
                if int(state.get('batch_cursor', 0) or 0) > 0:
                    time.sleep(0.1)
            kwargs['progress_callback'] = slow_progress
        delegated = ModelRoute('ADP-W-All', init_width=1, init_depth=1)
        return SimpleADPModelAdapter().fit_predict(delegated, **kwargs)
""",
        encoding="utf-8",
    )
    dist_info = plugin_root / "resume_fixture-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "entry_points.txt").write_text(
        "[simple_adp.dataset_providers]\nresume = resume_fixture:Provider\n"
        "[simple_adp.model_adapters]\nresume = resume_fixture:ResumableModelAdapter\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(plugin_root))
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(part for part in (str(plugin_root), inherited_pythonpath) if part),
    )
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")

    from simple_adp.runner.plugins import load_registries
    from simple_adp.runner.artifacts import estimator_checkpoint_path

    dataset_registry, model_registry = load_registries()
    settings = RunnerSettings(
        concurrency=1,
        cpu_only=True,
        max_retries=1,
        max_width=1,
        max_depth=1,
        max_neurons=8,
        max_epochs_adp=4,
        batch_size=1,
        patience_es=100,
        patience_width_exp=1,
        smoke_test=False,
        model_seeds=(17,),
        generic_profiles=("faithful",),
        tasks=("width_only",),
        include_vanilla=False,
        include_width_to_depth=False,
        include_depth_to_width=False,
        include_depth_only=False,
        width_only_depths=(1,),
        width_only_width=1,
        width_to_depth_width=1,
        width_to_depth_widths=(1,),
        depth_only_widths=(1,),
        vanilla_depths=(1,),
        vanilla_widths=(1,),
        screen_widths=(1,),
        screen_depths=(1,),
        deeper_screen_widths=(1,),
        run_hardness_screen=False,
        repeat_count=1,
        missing_present_task_repeats=1,
        post_launch_sample_delay_seconds=0,
        pressure_poll_seconds=0.01,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=tmp_path / "resume-results",
        run_root=tmp_path / "resume-run",
        dataset_providers=(dataset_registry.get("resume_fixture"),),
        dataset_suites=(("resume_fixture", "toy"),),
        model_adapters=(model_registry.get("resumable_fixture"),),
    )
    assert len(plan.runs) == 1
    run = plan.runs[0]
    checkpoint_path = estimator_checkpoint_path(run.result_dir)
    progress_path = Path(run.result_dir) / "telemetry" / "simple_adp" / run.identity.run_id / "progress_state.json"
    pressure = {"triggered": False}

    def host_sample():
        if checkpoint_path.exists() and progress_path.exists() and not pressure["triggered"]:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            cursor = int(checkpoint["payload"]["stage_state"]["batch_cursor"])
            if cursor > 0:
                pressure["triggered"] = True
                pressure["batch_cursor"] = cursor
                return HostMemorySample(8192, 512, 93.75, 0, 0, 0)
        return HostMemorySample(8192, 8192, 0, 0, 0, 0)

    coordinator = SuiteCoordinator(
        plan,
        settings,
        run_root=tmp_path / "resume-run",
        results_dir=tmp_path / "resume-results",
        repository_root=Path(__file__).resolve().parents[1],
        host_sampler=host_sample,
        gpu_sampler=lambda _device: None,
        poll_sleep=lambda duration: time.sleep(min(duration, 0.01)),
    )
    summary = coordinator.run()

    state = json.loads((tmp_path / "resume-run" / "jobs" / run.identity.run_id / "job_state.json").read_text())
    events = list(csv.DictReader((tmp_path / "resume-run" / "events.csv").open(newline="", encoding="utf-8")))
    result = json.loads((Path(run.result_dir) / "result.json").read_text())
    resume_cursor_path = Path(run.result_dir) / "resumed_batch_cursor.txt"
    assert pressure["triggered"] is True
    assert pressure["batch_cursor"] > 0
    assert summary["completed"] == 1
    assert state["status"] == "completed"
    assert state["pause_count"] == 1
    assert state["host_pause_count"] == 1
    assert state["attempt"] == 2
    assert state["retry_count"] == 0
    assert state["device_mode"] == "cpu"
    assert state["launch_context"]["resumed_from_pause"] is True
    assert [event["event"] for event in events].count("pause") == 1
    assert [event["event"] for event in events].count("resume") == 1
    assert result["status"] == "completed"
    assert result["model_id"] == "PLUGIN-RESUMABLE"
    assert checkpoint_path.is_file()
    assert int(resume_cursor_path.read_text(encoding="utf-8")) == pressure["batch_cursor"]
