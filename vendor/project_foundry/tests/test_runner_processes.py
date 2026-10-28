from types import SimpleNamespace
import json
import os
import select
import subprocess
import sys

from simple_adp.runner import processes
from simple_adp.runner import process_cleanup
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec
from simple_adp.runner.state import JobRuntime, JobStateStore


def test_launch_persists_source_contract_metadata_and_inherits_stdin(tmp_path, monkeypatch):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="route", seed=7,
        split_sha256="split", run_signature="signature",
    )
    ref = DatasetTaskRef("toy", "suite", "1")
    spec = RunSpec(
        identity, ref, "probe", ModelRoute("ADP-W-All"), "split.npz",
        str(tmp_path / "results"), {}, ("python", "-c", "pass"),
    )
    runtime = JobRuntime(spec)
    store = JobStateStore(tmp_path / "run")
    checkpoint_path = tmp_path / "results" / "estimator.checkpoint.pt"
    context = {"resumed_from_pause": True, "checkpoint_path": str(checkpoint_path)}
    observed = {}

    def child_env(*_args, **_kwargs):
        observed["env_built"] = True
        observed["device_mode"] = _kwargs.get("device_mode")
        observed["launch_budget"] = _kwargs.get("launch_budget")
        return {"PATH": "/usr/bin"}

    class Child:
        pid = 456

    def popen(command, **kwargs):
        observed["command"] = command
        observed["popen_kwargs"] = kwargs
        assert observed["env_built"]
        return Child()

    monkeypatch.setattr(processes, "launcher_child_env", child_env)
    monkeypatch.setattr(processes.subprocess, "Popen", popen)

    processes.launch_job(
        runtime, settings=RunnerSettings(cpu_only=True), state_store=store,
        repository_root=tmp_path, launch_budget=10**9, device="cpu", launch_context=context,
    )

    job_dir = store.job_dir(runtime)
    metadata = json.loads((job_dir / "job_metadata.json").read_text())
    state = store.load(runtime.job_id)
    log = (job_dir / "job.log").read_text()
    assert metadata["attempt"] == 1 and metadata["device_mode"] == "cpu"
    assert metadata["launch_context"] == context
    assert state["status"] == "running" and state["launch_context"] == context
    assert state["state_path"] == str(store.state_path(runtime))
    assert f"=== LAUNCH {runtime.job_id} attempt=1 device=cpu ===" in log
    assert "COMMAND: python -c pass" in log and "LAUNCH_CONTEXT:" in log
    assert observed["command"] == list(spec.command)
    assert observed["device_mode"] == "cpu"
    assert observed["launch_budget"] == 10**9
    assert observed["popen_kwargs"]["env"]["SIMPLE_ADP_CHECKPOINT_PATH"] == str(checkpoint_path)
    for name in ("LAST", "BEST", "BEST_BATCH"):
        assert observed["popen_kwargs"]["env"][f"SIMPLE_ADP_CHECKPOINT_{name}_PATH"] == str(
            job_dir / f"checkpoint_{name.lower()}.pt"
        )
    assert observed["popen_kwargs"]["env"]["TABULAR_CPU_JOB_CONCURRENCY"] == "1"
    assert "stdin" not in observed["popen_kwargs"]
    assert observed["popen_kwargs"]["text"] is True
    assert observed["popen_kwargs"]["stdout"].encoding.lower() == "utf-8"
    assert observed["popen_kwargs"]["start_new_session"] is (os.name == "posix")
    runtime.log_handle.close()


def test_launch_persists_launching_before_environment_and_log_creation(tmp_path, monkeypatch):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="launch-order", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="route", seed=7,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "launch-order"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "results"), {},
        ("python", "worker.py"),
    )
    runtime = JobRuntime(spec)
    store = JobStateStore(tmp_path / "run")
    observed = {}

    def child_env(*_args, **_kwargs):
        job_dir = store.job_dir(runtime)
        observed["state_at_env_build"] = store.load(runtime.job_id)["status"]
        observed["metadata_exists_at_env_build"] = (job_dir / "job_metadata.json").is_file()
        observed["log_exists_at_env_build"] = (job_dir / "job.log").exists()
        raise RuntimeError("environment build failed")

    monkeypatch.setattr(processes, "launcher_child_env", child_env)

    import pytest
    with pytest.raises(RuntimeError, match="environment build failed"):
        processes.launch_job(
            runtime, settings=RunnerSettings(), state_store=store,
            repository_root=tmp_path, launch_budget=1, device="gpu",
        )

    assert observed == {
        "state_at_env_build": "launching",
        "metadata_exists_at_env_build": True,
        "log_exists_at_env_build": False,
    }
    assert runtime.attempt == 1
    assert runtime.status.value == "launching"
    assert store.load(runtime.job_id)["status"] == "launching"


def test_launch_popen_failure_closes_log_and_keeps_launching_state(tmp_path, monkeypatch):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="popen-failure", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="route", seed=7,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "popen-failure"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "results"), {},
        ("python", "worker.py"),
    )
    runtime = JobRuntime(spec)
    store = JobStateStore(tmp_path / "run")

    def fail_to_spawn(*_args, **_kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(processes.subprocess, "Popen", fail_to_spawn)
    import pytest
    with pytest.raises(OSError, match="spawn failed"):
        processes.launch_job(
            runtime, settings=RunnerSettings(), state_store=store,
            repository_root=tmp_path, launch_budget=1, device="gpu",
        )

    job_dir = store.job_dir(runtime)
    assert runtime.attempt == 1
    assert runtime.status.value == "launching"
    assert runtime.pid is None
    assert runtime.log_handle is None
    assert store.load(runtime.job_id)["status"] == "launching"
    assert "=== LAUNCH " in (job_dir / "job.log").read_text()


def test_launch_rewrites_shared_cpu_flag_after_inherited_shared_budgeting(tmp_path, monkeypatch):
    import simple_adp.runner.runtime_tuning as tuning

    monkeypatch.setattr(tuning, "detected_cpus", lambda: (0, 1, 2, 3))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TABULAR_CHILD_SHARED_CPU", "1")
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="shared-mode", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="shared-mode", seed=7,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "shared-mode"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "results"), {},
        ("python", "-c", "pass"),
    )
    runtime = JobRuntime(spec)
    store = JobStateStore(tmp_path / "run")

    class Child:
        pid = 789

    observed = {}

    def popen(_command, **kwargs):
        observed["env"] = kwargs["env"]
        return Child()

    monkeypatch.setattr(processes.subprocess, "Popen", popen)
    processes.launch_job(
        runtime, settings=RunnerSettings(cpu_only=True), state_store=store,
        repository_root=tmp_path, launch_budget=2, device="cpu",
    )

    env = observed["env"]
    # launcher_child_env applies the source's inherited shared-mode OR before
    # launch_job resets the flag and CPU concurrency metadata for the child.
    assert env["TABULAR_CHILD_SHARED_CPU"] == "0"
    assert env["TABULAR_CPU_WORKERS"] == "0"
    assert env["TABULAR_CPU_JOB_CONCURRENCY"] == "2"
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["TABULAR_CPU_AFFINITY_CPUS"] == "0,1,2,3"
    assert env["GOMP_CPU_AFFINITY"] == "0,1,2,3"
    runtime.log_handle.close()


def test_single_inherited_gpu_mask_can_differ_from_coordinator_sensor_index(
    tmp_path, monkeypatch,
):
    import simple_adp.runner.runtime_tuning as tuning
    from simple_adp.runner.resources import (
        CallbackResourceProvider, GPUMemorySample, HostMemorySample,
        ProcessMemorySample,
    )
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.runner.plan import SuitePlan

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(tuning, "detected_cpus", lambda: (0, 1))
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="gpu-mask", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="gpu-mask", seed=3,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "gpu-mask"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "results"), {},
        ("python", "-c", "pass"),
    )
    settings = RunnerSettings(gpu_device_index=0, concurrency=1)
    runtime = JobRuntime(spec)
    store = JobStateStore(tmp_path / "run")
    observed = {"sampled_gpu_indices": [], "child_gpu_indices": []}

    class Child:
        pid = 812

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        processes.subprocess, "Popen",
        lambda _command, **kwargs: observed.update(env=kwargs["env"]) or Child(),
    )
    processes.launch_job(
        runtime, settings=settings, state_store=store,
        repository_root=tmp_path, launch_budget=1, device="gpu",
    )
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert observed["env"]["TABULAR_LOCAL_GPU_INDEX"] == "0"

    provider = CallbackResourceProvider(
        host_sampler=lambda: HostMemorySample(8192, 8192, 0, 0, 0, 0),
        gpu_sampler=lambda index: (
            observed["sampled_gpu_indices"].append(index)
            or GPUMemorySample(1000, 0, 1000, 0)
        ),
        tree_sampler=lambda pid: frozenset({pid}),
        child_sampler=lambda pid, **kwargs: (
            observed["child_gpu_indices"].append(
                (kwargs["gpu_device_index"], kwargs["gpu_mode"])
            )
            or ProcessMemorySample(host_rss_mib=10, gpu_vram_mib=50, pid_set=frozenset({pid}))
        ),
        external_host_sampler=lambda _excluded: ProcessMemorySample(),
        external_gpu_sampler=lambda index, _excluded: (
            observed.setdefault("external_gpu_indices", []).append(index)
            or ProcessMemorySample()
        ),
    )
    plan = SuitePlan(
        "signature", {"run_signature": "signature", "jobs": [spec.as_dict()]},
        (spec,),
    )
    coordinator = SuiteCoordinator(
        plan, settings, run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=provider,
    )
    coordinator._sample_snapshot([runtime], [], concurrency=1)

    assert observed["sampled_gpu_indices"] == [0]
    assert observed["child_gpu_indices"] == [(0, True)]
    assert observed["external_gpu_indices"] == [0]
    runtime.log_handle.close()


def test_log_close_is_attempted_even_when_flush_fails():
    class Handle:
        closed = False

        def flush(self):
            raise RuntimeError("flush failure")

        def close(self):
            self.closed = True

    handle = Handle()
    processes.close_log_handle(handle)
    assert handle.closed


def test_log_close_failure_is_swallowed_after_successful_flush():
    class Handle:
        flushed = False

        def flush(self):
            self.flushed = True

        def close(self):
            raise RuntimeError("close failure")

    handle = Handle()
    processes.close_log_handle(handle)
    assert handle.flushed


def test_exact_argv_match_uses_string_tokens_without_path_normalizing_and_earliest_pid(monkeypatch):
    zombie = processes.psutil.STATUS_ZOMBIE

    class Process:
        def __init__(self, pid, argv, created, *, cwd="/repo", status="running", running=True):
            self.info = {
                "pid": pid, "cmdline": argv, "create_time": created, "cwd": cwd,
            }
            self._status = status
            self._running = running

        def is_running(self):
            return self._running

        def status(self):
            return self._status

    expected = Process(20, ["python", "", "worker.py", "--path", "a/../b"], 4.0)
    earliest_pid = Process(
        8, ["python", "worker.py", "--path", "a/../b"], 2.0,
        cwd="/unrelated-working-directory",
    )
    tied_later_pid = Process(12, ["python", "worker.py", "--path", "a/../b"], 2.0)
    wrong_path_value = Process(1, ["python", "worker.py", "--path", "b"], 0.1)
    zombie_process = Process(2, ["python", "worker.py", "--path", "a/../b"], 0.0, status=zombie)
    stopped_process = Process(3, ["python", "worker.py", "--path", "a/../b"], 0.0, running=False)
    malformed = SimpleNamespace(info={"cmdline": ["python", "worker.py"]})
    observed = {}

    def process_iter(*, attrs):
        observed["attrs"] = attrs
        return iter((expected, wrong_path_value, tied_later_pid, zombie_process, malformed, stopped_process, earliest_pid))

    monkeypatch.setattr(processes.psutil, "process_iter", process_iter)
    result = processes.find_exact_argv(["python", "", "worker.py", "--path", "a/../b"])

    assert result is earliest_pid
    # Startup recovery matches normalized argv but deliberately does not ask
    # process_iter for or compare cwd, as in the source runner.
    assert observed["attrs"] == ["pid", "cmdline", "create_time"]


def test_exact_argv_matching_returns_none_for_empty_or_unreadable_process_table(monkeypatch):
    monkeypatch.setattr(processes.psutil, "process_iter", lambda **_kwargs: (_ for _ in ()).throw(OSError("table unavailable")))
    assert processes.find_exact_argv([]) is None
    assert processes.find_exact_argv([""]) is None
    assert processes.find_exact_argv(["python", "worker.py"]) is None


def test_process_liveness_normalizes_pid_and_fails_closed(monkeypatch):
    observed = []

    class Process:
        def __init__(self, pid):
            observed.append(("Process", pid))

        def is_running(self):
            return True

        def status(self):
            return "running"

    monkeypatch.setattr(processes.psutil, "pid_exists", lambda pid: observed.append(("exists", pid)) or True)
    monkeypatch.setattr(processes.psutil, "Process", Process)
    assert processes.process_is_live("42") is True
    assert processes.process_is_live(0) is False
    assert processes.process_is_live(-2) is False
    assert processes.process_is_live("not-a-pid") is False
    assert processes.process_is_live(None) is False
    assert observed == [("exists", 42), ("Process", 42)]

    monkeypatch.setattr(processes.psutil, "pid_exists", lambda _pid: (_ for _ in ()).throw(RuntimeError("psutil failure")))
    assert processes.process_is_live(42) is False


def test_linux_termination_removes_a_real_parent_and_grandchild_tree():
    if os.name != "posix":
        import pytest

        pytest.skip("this process-tree integration contract targets Linux/POSIX")

    child_code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        "print(p.pid, flush=True); time.sleep(120)"
    )
    root = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    child_pid = None
    try:
        assert root.stdout is not None
        ready, _, _ = select.select([root.stdout], [], [], 5.0)
        assert ready, "child process did not report its grandchild PID"
        child_pid = int(root.stdout.readline().strip())
        assert processes.process_is_live(child_pid)

        processes.terminate_process_tree(root, timeout_seconds=0.5)

        assert root.poll() is not None
        assert not processes.process_is_live(child_pid)
    finally:
        if root.poll() is None:
            processes.terminate_process_tree(root, timeout_seconds=0.5)
        if child_pid is not None and processes.process_is_live(child_pid):
            try:
                processes.psutil.Process(child_pid).kill()
            except Exception:
                pass


def test_linux_fixed_grace_cleanup_removes_real_parent_and_grandchild_tree():
    if os.name != "posix":
        import pytest

        pytest.skip("this process-tree integration contract targets Linux/POSIX")

    child_code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        "print(p.pid, flush=True); time.sleep(120)"
    )
    root = subprocess.Popen(
        [sys.executable, "-c", child_code], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, start_new_session=True,
    )
    child_pid = None
    try:
        assert root.stdout is not None
        ready, _, _ = select.select([root.stdout], [], [], 5.0)
        assert ready, "child process did not report its grandchild PID"
        child_pid = int(root.stdout.readline().strip())
        assert processes.process_is_live(child_pid)

        processes.terminate_process_tree_after_grace(root.pid, grace_seconds=0.05)

        root.wait(timeout=5)
        assert not processes.process_is_live(child_pid)
    finally:
        if root.poll() is None:
            processes.terminate_process_tree_after_grace(root.pid, grace_seconds=0.05)
        if child_pid is not None and processes.process_is_live(child_pid):
            try:
                processes.psutil.Process(child_pid).kill()
            except Exception:
                pass


def test_reconcile_trusts_live_saved_pid_without_verifying_command(monkeypatch, tmp_path):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="route", seed=7,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe", ModelRoute("ADP-W-All"),
        "split.npz", str(tmp_path / "results"), {}, ("python", "worker.py", "--route", "expected"),
    )
    runtime = JobRuntime(spec)
    record = {
        "status": "running", "pid": 4242, "attempt": 1,
        "command": ["python", "different-worker.py", "--route", "unrelated"],
        "device_mode": "cpu",
    }
    calls = []
    monkeypatch.setattr(processes, "process_is_live", lambda pid: calls.append(("live", pid)) or True)
    monkeypatch.setattr(
        processes, "find_exact_argv",
        lambda _command: (_ for _ in ()).throw(AssertionError("live saved PID must skip argv verification")),
    )

    location = processes.reconcile_runtime(runtime, record)

    assert location == "active"
    assert runtime.pid == 4242
    assert isinstance(runtime.process, processes.RecoveredProcessHandle)
    assert runtime.status.value == "running"
    assert calls == [("live", 4242)]


def test_legacy_recovery_defaults_missing_or_falsey_lane_and_pause_to_gpu_empty(
    monkeypatch, tmp_path,
):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="legacy", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="route", seed=7,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "legacy"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(tmp_path / "results"), {},
        ("python", "worker.py"),
    )
    monkeypatch.setattr(processes, "process_is_live", lambda _pid: True)

    store = JobStateStore(tmp_path / "run")
    for fields in ({}, {"device_mode": None, "pause_reason": None},
                   {"device_mode": "", "pause_reason": ""}):
        runtime = JobRuntime(spec)
        record = {"status": "launching", "pid": 4242, "attempt": 1, **fields}
        assert processes.reconcile_runtime(runtime, record, state_store=store) == "active"
        assert runtime.device_mode == "gpu"
        assert runtime.pause_reason == ""
        saved = store.load(runtime.job_id)
        assert saved["status"] == "running"
        assert saved["device_mode"] == "gpu"
        assert saved["pause_reason"] == ""


def test_reconcile_can_recover_a_paused_record_as_running_by_exact_argv(monkeypatch, tmp_path):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route="route", seed=7,
        split_sha256="split", run_signature="signature",
    )
    command = ("python", "worker.py", "--route", "expected")
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe", ModelRoute("ADP-W-All"),
        "split.npz", str(tmp_path / "results"), {}, command,
    )
    runtime = JobRuntime(spec)
    matched = SimpleNamespace(pid=5151)
    monkeypatch.setattr(processes, "process_is_live", lambda _pid: False)
    monkeypatch.setattr(processes, "find_exact_argv", lambda actual: matched if tuple(actual) == command else None)

    location = processes.reconcile_runtime(runtime, {
        "status": "paused", "pid": 1234, "attempt": 1, "pause_reason": "gpu_memory_pressure",
        "device_mode": "gpu", "command": list(command),
    })

    assert location == "active"
    assert runtime.pid == matched.pid
    assert runtime.status.value == "running"
    assert runtime.recovered_process is True


def test_emergency_cleanup_kills_unique_descendants_in_reverse_then_roots(monkeypatch):
    killed = []

    class Process:
        def __init__(self, pid, descendants=(), *, command=None):
            self.pid = pid
            self._descendants = list(descendants)
            self.info = {
                "pid": pid, "cmdline": command or ["python", "-m", "simple_adp.runner.worker", "--spec", "job.json"],
                "cwd": "/repo",
            }

        def children(self, *, recursive):
            assert recursive is True
            return self._descendants

        def kill(self):
            killed.append(self.pid)

    shared = Process(33)
    first_child = Process(31)
    second_child = Process(32)
    third_child = Process(34)
    first = Process(10, [first_child, shared, second_child])
    second = Process(11, [shared, third_child])
    self_process = Process(900)
    monkeypatch.setattr(process_cleanup.psutil, "Process", lambda: SimpleNamespace(pid=900))
    calls = []

    def process_iter(attrs):
        calls.append(attrs)
        return iter([self_process, first, second])

    monkeypatch.setattr(process_cleanup.psutil, "process_iter", process_iter)
    monkeypatch.setattr(process_cleanup, "owns_process", lambda _command, _cwd, _root: True)

    assert process_cleanup.kill_owned_processes() == 6
    assert calls == [("pid", "cmdline", "cwd")]
    assert killed == [32, 33, 31, 10, 34, 11]


def test_emergency_cleanup_swallows_only_documented_process_races(monkeypatch):
    psutil = process_cleanup.psutil
    killed = []

    class Process:
        def __init__(self, pid, *, children=(), child_error=None, kill_error=None, info_error=None):
            self.pid = pid
            self._children = list(children)
            self._child_error = child_error
            self._kill_error = kill_error
            self._info_error = info_error
            self.info = {"pid": pid, "cmdline": ["python", "-m", "simple_adp.runner.worker", "--spec"], "cwd": "/repo"}

        def children(self, *, recursive):
            assert recursive is True
            if self._child_error:
                raise self._child_error
            return self._children

        def kill(self):
            if self._kill_error:
                raise self._kill_error
            killed.append(self.pid)

    inaccessible_record = Process(1, info_error=psutil.AccessDenied(pid=1))
    inaccessible_record.info = RecordAccessError(psutil.AccessDenied(pid=1))
    denied_child = Process(3, kill_error=psutil.AccessDenied(pid=3))
    denied_enumeration = Process(2, children=[denied_child], child_error=psutil.AccessDenied(pid=2))
    root_gone = Process(4, kill_error=psutil.NoSuchProcess(pid=4))
    denied_child_parent = Process(6, children=[denied_child])
    good = Process(5)
    self_process = Process(900)
    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(pid=900))
    monkeypatch.setattr(psutil, "process_iter", lambda *_args, **_kwargs: iter([
        inaccessible_record, denied_enumeration, root_gone, denied_child_parent,
        good, self_process,
    ]))
    monkeypatch.setattr(process_cleanup, "owns_process", lambda _command, _cwd, _root: True)

    assert process_cleanup.kill_owned_processes() == 3
    # Failed child enumeration omits descendants; the root is still killed.
    # AccessDenied/NoSuchProcess kill races are swallowed; the later root works.
    assert killed == [2, 6, 5]


class RecordAccessError:
    def __init__(self, error):
        self.error = error

    def __getitem__(self, _key):
        raise self.error

    def get(self, _key, default=None):
        return default


def test_emergency_cleanup_propagates_unexpected_tree_and_top_level_errors(monkeypatch):
    import pytest

    class Process:
        pid = 10
        info = {"pid": 10, "cmdline": ["python", "-m", "simple_adp.runner.worker", "--spec"], "cwd": "/repo"}

        def children(self, *, recursive):
            assert recursive is True
            raise RuntimeError("unexpected tree error")

    monkeypatch.setattr(process_cleanup.psutil, "Process", lambda: SimpleNamespace(pid=900))
    monkeypatch.setattr(process_cleanup, "owns_process", lambda *_args: True)
    monkeypatch.setattr(process_cleanup.psutil, "process_iter", lambda *_args, **_kwargs: iter([Process()]))
    with pytest.raises(RuntimeError, match="unexpected tree error"):
        process_cleanup.kill_owned_processes()

    monkeypatch.setattr(
        process_cleanup.psutil, "process_iter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("iterator failure")),
    )
    with pytest.raises(RuntimeError, match="iterator failure"):
        process_cleanup.kill_owned_processes()

    class KillFailure(Process):
        pid = 11

        def children(self, *, recursive):
            assert recursive is True
            return []

        def kill(self):
            raise RuntimeError("unexpected kill error")

    monkeypatch.setattr(process_cleanup.psutil, "process_iter", lambda *_args, **_kwargs: iter([KillFailure()]))
    with pytest.raises(RuntimeError, match="unexpected kill error"):
        process_cleanup.kill_owned_processes()
def test_outer_watchdog_signal_cleanup_rescans_tree_after_fixed_grace(monkeypatch):
    events = []

    class Process:
        def __init__(self, pid, children=()):
            self.pid = pid
            self.children_list = list(children)
            self.scans = []

        def children(self, recursive=False):
            assert recursive is False
            self.scans.append(recursive)
            if self is root and len(root.scans) == 1:
                return [branch]
            if self is root and len(root.scans) == 2:
                return [branch, late]
            return self.children_list

        def send_signal(self, sig):
            events.append(("signal", self.pid, sig))

    leaf = Process(12)
    branch = Process(11, [leaf])
    late = Process(13)
    root = Process(10)
    monkeypatch.setattr(processes.psutil, "Process", lambda _pid: root)
    monkeypatch.setattr(processes.time, "sleep", lambda duration: events.append(("sleep", duration)))

    processes.terminate_process_tree_after_grace(10, grace_seconds=2)

    import signal
    assert events == [
        ("signal", 12, signal.SIGTERM),
        ("signal", 11, signal.SIGTERM),
        ("signal", 10, signal.SIGTERM),
        ("sleep", 2.0),
        ("signal", 12, signal.SIGKILL),
        ("signal", 11, signal.SIGKILL),
        ("signal", 13, signal.SIGKILL),
        ("signal", 10, signal.SIGKILL),
    ]


