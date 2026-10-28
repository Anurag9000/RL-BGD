from types import SimpleNamespace

import psutil
import pytest

import simple_adp.runner.resources as resources


class FakeProcess:
    def __init__(self, pid, name, cmdline, rss, *, fail_identity=False, cached_memory=True):
        self.pid = pid
        self._name = name
        self._cmdline = cmdline
        self._rss = rss
        self.fail_identity = fail_identity
        self.info = {
            "pid": pid, "name": name, "cmdline": cmdline,
            "memory_info": SimpleNamespace(rss=rss) if cached_memory else None,
        }

    def name(self):
        if self.fail_identity:
            if self.fail_identity == "runtime":
                raise RuntimeError("identity lookup failed unexpectedly")
            raise psutil.AccessDenied(self.pid)
        return self._name

    def cmdline(self):
        if self.fail_identity:
            if self.fail_identity == "runtime":
                raise RuntimeError("identity lookup failed unexpectedly")
            raise psutil.AccessDenied(self.pid)
        return self._cmdline

    def memory_info(self):
        return SimpleNamespace(rss=self._rss)


def test_external_host_filter_uses_cached_identity_and_excludes_python_prefixes(monkeypatch):
    processes = [
        FakeProcess(1, "python3", ["python3", "train.py"], 1000),
        FakeProcess(2, "PyTorchWorker", ["service"], 2000),
        FakeProcess(3, "trainer", ["/usr/bin/python", "task.py"], 3000),
        FakeProcess(4, "worker", ["worker"], 4000, fail_identity=True),
        FakeProcess(5, "worker", ["worker"], 5000),
        FakeProcess(6, "worker", ["worker"], 6000),
        FakeProcess(7, "  PyTorchWorker ", ["service"], 7000),
        FakeProcess(8, "trainer", ["  py-helper"], 8000),
        FakeProcess(9, "worker", ["worker"], 9000, cached_memory=False),
        FakeProcess(10, "worker", ["worker"], 10_000, fail_identity="runtime"),
    ]
    observed_attrs = []

    def process_iter(attrs=None):
        observed_attrs.append(attrs)
        return iter(processes)

    monkeypatch.setattr(resources.psutil, "process_iter", process_iter)
    result = resources.sample_external_host_memory({5})
    # The source reads identity from Process.info populated by process_iter;
    # it never calls name()/cmdline() on these objects. The two fake method
    # failures therefore do not remove otherwise eligible processes.
    assert observed_attrs == [["pid", "name", "cmdline", "memory_info"]]
    assert result.pid_set == frozenset({4, 6, 9, 10})
    assert result.host_rss_mib == (4000 + 6000 + 9000 + 10_000) / resources.MIB


def test_external_host_sampler_skips_per_process_inspection_errors(monkeypatch):
    class UnreadableProcess:
        pid = 11

        @property
        def info(self):
            raise psutil.AccessDenied(11)

    readable = FakeProcess(12, "worker", ["worker"], 2048)
    monkeypatch.setattr(resources.psutil, "process_iter", lambda attrs=None: iter([UnreadableProcess(), readable]))

    sample = resources.sample_external_host_memory(())

    # The reference catches inspection errors inside the loop and keeps
    # accounting for later readable processes.
    assert sample.pid_set == frozenset({12})
    assert sample.host_rss_mib == 2048 / resources.MIB


def test_external_host_sampler_propagates_process_iterator_failure(monkeypatch):
    def process_iter(attrs=None):
        raise psutil.AccessDenied(pid=0)

    monkeypatch.setattr(resources.psutil, "process_iter", process_iter)

    # The source has no outer try around process_iter construction/iteration;
    # this distinction is part of the copied runner's observed behavior.
    try:
        resources.sample_external_host_memory(())
    except psutil.AccessDenied:
        pass
    else:
        raise AssertionError("top-level process_iter failure must propagate")


def test_external_host_sampler_propagates_mid_iteration_failure(monkeypatch):
    readable = FakeProcess(12, "worker", ["worker"], 2048)

    def process_iter(attrs=None):
        yield readable
        raise psutil.AccessDenied(pid=13)

    monkeypatch.setattr(resources.psutil, "process_iter", process_iter)

    try:
        resources.sample_external_host_memory(())
    except psutil.AccessDenied:
        pass
    else:
        raise AssertionError("mid-iteration process_iter failure must propagate")


def test_host_and_swap_sampling_floors_byte_counters_to_whole_mib(monkeypatch):
    mib = int(resources.MIB)
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: SimpleNamespace(
        total=3 * mib + mib - 1,
        available=2 * mib + mib - 1,
        percent=37.5,
    ))
    monkeypatch.setattr(resources.psutil, "swap_memory", lambda: SimpleNamespace(
        total=5 * mib + mib - 1,
        used=4 * mib + mib - 1,
        percent=62.5,
    ))

    sample = resources.sample_host_memory()

    assert sample.total_mib == 3
    assert sample.available_mib == 2
    assert sample.used_percent == 37.5
    assert sample.swap_total_mib == 5
    assert sample.swap_used_mib == 4
    assert sample.swap_percent == 62.5


@pytest.mark.parametrize("failed_sampler", ["virtual_memory", "swap_memory"])
def test_host_sampler_propagates_os_memory_errors(monkeypatch, failed_sampler):
    def fail():
        raise psutil.Error("memory sensor unavailable")

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: SimpleNamespace(
        total=4 * int(resources.MIB), available=3 * int(resources.MIB), percent=25.0,
    ))
    monkeypatch.setattr(resources.psutil, "swap_memory", lambda: SimpleNamespace(
        total=0, used=0, percent=0.0,
    ))
    monkeypatch.setattr(resources.psutil, failed_sampler, fail)

    with pytest.raises(psutil.Error):
        resources.sample_host_memory()


def test_external_gpu_filter_excludes_python_names_and_owned_pids(monkeypatch):
    monkeypatch.setattr(resources, "_gpu_process_rows", lambda _index: [
        (10, "python", 100.0),
        (11, "PyTorchApp", 200.0),
        (12, "owned-worker", 300.0),
        (13, "external-worker", 400.0),
        (14, "  PyTorchWorker  ", 500.0),
    ])
    result = resources.sample_external_gpu_memory(0, {12})
    assert result.pid_set == frozenset({13})
    assert result.gpu_vram_mib == 400.0


def test_external_gpu_sensor_failure_returns_empty_sample(monkeypatch):
    monkeypatch.setattr(resources, "_gpu_process_rows", lambda _index: (_ for _ in ()).throw(OSError("missing telemetry")))
    result = resources.sample_external_gpu_memory(0, ())
    assert result.pid_set == frozenset()
    assert result.gpu_vram_mib == 0.0


def test_process_tree_preserves_known_root_when_psutil_inspection_fails(monkeypatch):
    monkeypatch.setattr(
        resources.psutil, "Process",
        lambda _pid: (_ for _ in ()).throw(psutil.AccessDenied(77)),
    )
    assert resources.process_tree_pids(77) == frozenset({77})

    class UnreadableTree:
        def children(self, recursive):
            assert recursive is True
            raise RuntimeError("child enumeration failed")

    monkeypatch.setattr(resources.psutil, "Process", lambda _pid: UnreadableTree())
    assert resources.process_tree_pids(88) == frozenset({88})


def test_gpu_memory_sampling_matches_ledger_queries_and_integer_accounting(monkeypatch):
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return b"1000.9, 250.8\n"

    monkeypatch.setattr(resources.subprocess, "check_output", check_output)
    sample = resources.sample_gpu_memory(3)
    assert sample == resources.GPUMemorySample(1000, 250, 750, 25.0)
    assert calls == [(
        ["nvidia-smi", "--id=3", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        {"stderr": resources.subprocess.DEVNULL},
    )]

    calls.clear()
    monkeypatch.setattr(resources.subprocess, "check_output", lambda command, **kwargs: calls.append((command, kwargs)) or b"0, 4\n")
    assert resources.sample_gpu_memory(0) == resources.GPUMemorySample(0, 4, 0, 0.0)


def test_gpu_used_percentage_is_not_clamped_when_used_exceeds_total(monkeypatch):
    monkeypatch.setattr(resources.subprocess, "check_output", lambda *_args, **_kwargs: b"100, 125\n")

    sample = resources.sample_gpu_memory(0)

    assert sample == resources.GPUMemorySample(100, 125, 0, 125.0)


@pytest.mark.parametrize("output", [b"", b"not-a-number, 20\n", b"100,\n"])
def test_gpu_memory_parse_failures_report_sensor_unavailable(monkeypatch, output):
    monkeypatch.setattr(resources.subprocess, "check_output", lambda *_args, **_kwargs: output)
    assert resources.sample_gpu_memory(0) is None


@pytest.mark.parametrize("error", [
    OSError("nvidia-smi unavailable"),
    resources.subprocess.CalledProcessError(1, "nvidia-smi"),
    RuntimeError("sensor runtime failure"),
])
def test_gpu_memory_query_exception_boundaries_return_unavailable_or_zero(monkeypatch, error):
    monkeypatch.setattr(
        resources.subprocess, "check_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    assert resources.sample_gpu_memory(0) is None
    assert resources._gpu_process_rows(0) == []
    assert resources.sample_external_gpu_memory(0, ()).gpu_vram_mib == 0.0

    monkeypatch.setattr(resources, "process_tree_pids", lambda _pid: frozenset({12}))
    monkeypatch.setattr(resources.psutil, "Process", lambda _pid: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=int(resources.MIB)),
    ))
    child = resources.sample_child_memory(12, gpu_device_index=0, gpu_mode=True)
    assert child.host_rss_mib == 1.0
    assert child.gpu_vram_mib == 0.0


def test_child_memory_sums_tree_rss_and_only_queries_gpu_for_gpu_jobs(monkeypatch):
    rss_by_pid = {10: 3 * int(resources.MIB), 11: 2 * int(resources.MIB)}
    monkeypatch.setattr(resources, "process_tree_pids", lambda _pid: frozenset({10, 11}))
    monkeypatch.setattr(
        resources.psutil, "Process",
        lambda pid: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=rss_by_pid[pid])),
    )
    gpu_calls = []
    monkeypatch.setattr(resources, "_gpu_process_rows", lambda index: gpu_calls.append(index) or [
        (10, "root", 120.0), (11, "descendant", 80.0), (12, "unrelated", 900.0),
    ])

    cpu_sample = resources.sample_child_memory(10, gpu_device_index=2, gpu_mode=False)
    gpu_sample = resources.sample_child_memory(10, gpu_device_index=2, gpu_mode=True)

    assert cpu_sample.host_rss_mib == 5.0
    assert cpu_sample.gpu_vram_mib == 0.0
    assert gpu_sample.host_rss_mib == 5.0
    assert gpu_sample.gpu_vram_mib == 200.0
    assert gpu_calls == [2]


def test_child_memory_keeps_tree_identity_when_one_rss_lookup_fails(monkeypatch):
    monkeypatch.setattr(resources, "process_tree_pids", lambda _pid: frozenset({10, 11}))

    def process(pid):
        if pid == 10:
            raise psutil.AccessDenied(pid)
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=int(resources.MIB)))

    monkeypatch.setattr(resources.psutil, "Process", process)
    monkeypatch.setattr(resources, "_gpu_process_rows", lambda _index: (_ for _ in ()).throw(AssertionError("CPU job queried GPU")))

    sample = resources.sample_child_memory(10, gpu_device_index=0, gpu_mode=False)

    assert sample.pid_set == frozenset({10, 11})
    assert sample.host_rss_mib == 1.0
    assert sample.gpu_vram_mib == 0.0


def test_coordinator_rechecks_child_liveness_at_memory_sample_boundary(tmp_path):
    from types import SimpleNamespace

    from simple_adp.runner.config import RunnerSettings
    from simple_adp.runner.scheduler import SuiteCoordinator

    class Provider:
        def __init__(self):
            self.child_calls = []

        def sample_child(self, pid, *, gpu_device_index, gpu_mode):
            self.child_calls.append((pid, gpu_device_index, gpu_mode))
            return resources.ProcessMemorySample(host_rss_mib=500, gpu_vram_mib=100)

    class Runtime:
        process = object()
        pid = 44
        is_gpu = True

        def __init__(self):
            self.poll_count = 0

        @property
        def live(self):
            self.poll_count += 1
            return self.poll_count == 1

    provider = Provider()
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="liveness", runs=()), RunnerSettings(),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=provider,
    )
    runtime = Runtime()
    assert runtime.live  # candidate/latest selection sees it alive
    sample = coordinator._sample_runtime_child(runtime)  # sampler's own poll sees exit

    assert sample == resources.ProcessMemorySample()
    assert provider.child_calls == []


def test_gpu_process_query_has_no_timeout_and_sensor_errors_become_empty(monkeypatch):
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return b"12, worker, 50\nmalformed\n13, other, bad\n"

    monkeypatch.setattr(resources.subprocess, "check_output", check_output)
    assert resources._gpu_process_rows(2) == [(12, "worker", 50.0)]
    assert calls == [(
        ["nvidia-smi", "--id=2", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        {"stderr": resources.subprocess.DEVNULL},
    )]
    monkeypatch.setattr(resources.subprocess, "check_output", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("sensor failed")))
    assert resources._gpu_process_rows(2) == []


def test_gpu_process_query_uses_source_first_three_csv_fields_and_swallows_decode_errors(monkeypatch):
    monkeypatch.setattr(
        resources.subprocess, "check_output",
        lambda *_args, **_kwargs: b"12, worker, 50, trailing-field\nmalformed\n",
    )
    assert resources._gpu_process_rows(0) == [(12, "worker", 50.0)]

    monkeypatch.setattr(
        resources.subprocess, "check_output",
        lambda *_args, **_kwargs: object(),
    )
    assert resources._gpu_process_rows(0) == []


def test_coordinator_accepts_synthetic_resource_provider(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from simple_adp.runner.config import RunnerSettings
    from simple_adp.runner.resources import (
        GPUMemorySample,
        HostMemorySample,
        ProcessMemorySample,
        ResourceProvider,
    )
    from simple_adp.runner.scheduler import SuiteCoordinator
    import simple_adp.runner.scheduler as scheduler_module

    class SyntheticResources:
        def __init__(self):
            self.calls = []

        def sample_host(self):
            self.calls.append(("host",))
            return HostMemorySample(8192, 4096, 50, 0, 0, 0)

        def sample_gpu(self, device_index):
            self.calls.append(("gpu", device_index))
            return GPUMemorySample(8192, 7000, 1192, 85.4)

        def process_tree(self, pid):
            self.calls.append(("tree", pid))
            return frozenset({pid, pid + 1})

        def sample_active_rss(self, pids):
            self.calls.append(("active_rss", frozenset(pids)))
            return 250.0

        def sample_child(self, pid, *, gpu_device_index, gpu_mode):
            self.calls.append(("child", pid, gpu_device_index, gpu_mode))
            return ProcessMemorySample(250, 100, frozenset({pid, pid + 1}))

        def sample_external_host(self, excluded_pids):
            self.calls.append(("external_host", frozenset(excluded_pids)))
            return ProcessMemorySample(host_rss_mib=800, pid_set=frozenset({90}))

        def sample_external_gpu(self, device_index, excluded_pids):
            self.calls.append(("external_gpu", device_index, frozenset(excluded_pids)))
            return ProcessMemorySample(gpu_vram_mib=300, pid_set=frozenset({91}))

    provider = SyntheticResources()
    concurrency_file = tmp_path / "concurrency.txt"
    concurrency_file.write_text("10\n", encoding="utf-8")
    original_resolve_concurrency = RunnerSettings.resolve_concurrency
    concurrency_reads = []

    def record_concurrency_read(settings):
        value = original_resolve_concurrency(settings)
        if settings.concurrency_file == str(concurrency_file):
            concurrency_reads.append(value)
        return value

    monkeypatch.setattr(RunnerSettings, "resolve_concurrency", record_concurrency_read)
    original_capacity = scheduler_module.resolve_launch_capacity
    original_pressure = scheduler_module.hard_pressure

    def record_capacity(*args, **kwargs):
        provider.calls.append(("capacity",))
        capacity = original_capacity(*args, **kwargs)
        # The source performs another read for its active-count ceiling. Make
        # the file unbounded between those two source reads.
        concurrency_file.write_text("0\n", encoding="utf-8")
        return capacity

    def record_pressure(*args, **kwargs):
        provider.calls.append(("hard_pressure",))
        return original_pressure(*args, **kwargs)

    monkeypatch.setattr(scheduler_module, "resolve_launch_capacity", record_capacity)
    monkeypatch.setattr(scheduler_module, "hard_pressure", record_pressure)
    assert isinstance(provider, ResourceProvider)
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="synthetic", runs=()),
        RunnerSettings(concurrency=10, concurrency_file=str(concurrency_file)),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=provider,
    )
    runtime = SimpleNamespace(
        live=True, job_id="synthetic-job", pid=10, is_gpu=True,
        started_at=1.0, attempt=1, process=object(),
    )
    # Two live routes need a capacity of 10 here: capacity reads 10, then the
    # source's separate active-count check rereads the new unbounded value 0.
    runtimes = [runtime, SimpleNamespace(
        live=True, job_id="second-job", pid=11, is_gpu=True,
        started_at=2.0, attempt=1, process=object(),
    )]
    snapshot, latest, latest_gpu = coordinator._sample_snapshot(runtimes, [], 10)
    assert latest is latest_gpu and latest.job_id == "second-job"
    assert snapshot.active_rss_mib == 250
    assert snapshot.host.available_mib == 4096
    # Capacity uses OS-reported available RAM directly; active RSS is diagnostic.
    assert snapshot.launch_capacity == 10
    assert concurrency_reads == [10, 10**9]
    assert snapshot.latest_child_vram_mib == 100
    assert snapshot.external_host_mib == 800
    assert snapshot.external_gpu_mib == 300
    assert ("tree", 10) in provider.calls
    assert ("active_rss", frozenset({10, 11, 12})) in provider.calls
    assert ("external_host", frozenset({10, 11, 12})) in provider.calls
    assert ("external_gpu", 0, frozenset({10, 11, 12})) in provider.calls
    # The reference computes capacity and strict pressure flags before child
    # and external memory sampling in each scheduler pass.
    call_names = [call[0] for call in provider.calls]
    assert call_names.index("capacity") < call_names.index("child")
    assert call_names.index("hard_pressure") > call_names.index("capacity")
    assert call_names.index("capacity") < call_names.index("external_host")
    assert call_names.index("hard_pressure") < call_names.index("child")

    provider.calls.clear()
    stale_and_suspended = [
        SimpleNamespace(pid=None, process=None, live=False),
        SimpleNamespace(pid=99, process=SimpleNamespace(), live=False),
    ]
    coordinator._sample_snapshot([], stale_and_suspended, 1)
    assert not any(call[0] == "tree" for call in provider.calls)
    assert ("external_host", frozenset()) in provider.calls


def test_active_rss_sum_returns_zero_if_any_process_lookup_fails(monkeypatch):
    def process(pid):
        if pid == 2:
            raise psutil.AccessDenied(pid)
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=int(resources.MIB)))

    monkeypatch.setattr(resources.psutil, "Process", process)
    assert resources.sample_active_rss_mib({1, 2}) == 0.0


def test_latest_child_remaining_memory_diagnostics_clamp_at_zero(tmp_path):
    from simple_adp.runner.resources import (
        GPUMemorySample,
        HostMemorySample,
        ProcessMemorySample,
    )
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.runner.config import RunnerSettings

    class OvercommittedResources:
        def sample_host(self):
            return HostMemorySample(1000, 50, 95.0, 0, 0, 0.0)

        def sample_gpu(self, _device_index):
            return GPUMemorySample(1000, 980, 20, 98.0)

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_child(self, _pid, *, gpu_device_index, gpu_mode):
            assert gpu_device_index == 0 and gpu_mode
            return ProcessMemorySample(host_rss_mib=100, gpu_vram_mib=80)

        def sample_external_host(self, _excluded_pids):
            return ProcessMemorySample()

        def sample_external_gpu(self, _device_index, _excluded_pids):
            return ProcessMemorySample()

    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="signature", runs=()),
        RunnerSettings(child_host_gate_buffer_mib=0, child_gpu_gate_buffer_mib=0),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=OvercommittedResources(),
    )
    runtime = SimpleNamespace(
        live=True, job_id="overcommitted", pid=123, is_gpu=True,
        started_at=1.0, attempt=1, process=object(),
    )

    coordinator._sample_snapshot([runtime], [], 1)

    assert coordinator.memory.values["last_host_remaining_mib"] == 0
    assert coordinator.memory.values["last_gpu_remaining_mib"] == 0
    assert coordinator.last_resource_snapshot["latest_child_host_remaining_mib"] == 0
    assert coordinator.last_resource_snapshot["latest_child_gpu_remaining_mib"] == 0


def test_snapshot_samples_only_latest_child_outside_pressure_victim_ranking(tmp_path):
    from simple_adp.runner.config import RunnerSettings
    from simple_adp.runner.resources import (
        GPUMemorySample,
        HostMemorySample,
        ProcessMemorySample,
    )
    from simple_adp.runner.scheduler import SuiteCoordinator

    class Resources:
        def __init__(self):
            self.child_calls = []

        def sample_host(self):
            return HostMemorySample(8000, 5000, 37.5, 0, 0, 0)

        def sample_gpu(self, _device_index):
            return GPUMemorySample(1000, 200, 800, 20)

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_active_rss(self, _pids):
            # Source active RSS is a separate aggregate, not the latest-child
            # footprint used by soft gates.
            return 678.0

        def sample_child(self, pid, *, gpu_device_index, gpu_mode):
            self.child_calls.append((pid, gpu_device_index, gpu_mode))
            assert pid == 20
            return ProcessMemorySample(host_rss_mib=120.0, gpu_vram_mib=0.0)

        def sample_external_host(self, _excluded_pids):
            return ProcessMemorySample()

        def sample_external_gpu(self, _device_index, _excluded_pids):
            return ProcessMemorySample()

    resources = Resources()
    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="signature", runs=()),
        RunnerSettings(), run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=resources,
    )
    older = SimpleNamespace(
        live=True, job_id="older-gpu", pid=10, is_gpu=True,
        started_at=1.0, attempt=1, process=object(),
    )
    latest = SimpleNamespace(
        live=True, job_id="latest-cpu", pid=20, is_gpu=False,
        started_at=2.0, attempt=1, process=object(),
    )

    snapshot, _, _ = coordinator._sample_snapshot([older, latest], [], 2)

    assert resources.child_calls == [(20, 0, False)]
    assert snapshot.active_rss_mib == 678.0
    assert snapshot.latest_child_rss_mib == 120.0


def test_latest_cpu_child_can_mask_latest_gpu_child_in_soft_gate_sample(tmp_path):
    """Pin the source's mixed-lane latest-child sampling quirk."""
    from simple_adp.runner.config import RunnerSettings
    from simple_adp.runner.resources import (
        GPUMemorySample,
        HostMemorySample,
        ProcessMemorySample,
    )
    from simple_adp.runner.scheduler import SuiteCoordinator

    class MixedLaneResources:
        def sample_host(self):
            return HostMemorySample(16_000, 8_000, 50.0, 0, 0, 0.0)

        def sample_gpu(self, _device_index):
            # The latest GPU child leaves 350 MiB free. Its 300 MiB use plus
            # the 100 MiB gate buffer should close the GPU gate if measured.
            return GPUMemorySample(1000, 650, 350, 65.0)

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_child(self, pid, *, gpu_device_index, gpu_mode):
            assert gpu_device_index == 0
            if gpu_mode:
                assert pid == 101
                return ProcessMemorySample(host_rss_mib=200, gpu_vram_mib=300)
            assert pid == 202
            return ProcessMemorySample(host_rss_mib=100, gpu_vram_mib=0)

        def sample_external_host(self, _excluded_pids):
            return ProcessMemorySample()

        def sample_external_gpu(self, _device_index, _excluded_pids):
            return ProcessMemorySample()

    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="mixed-lane-signature", runs=()),
        RunnerSettings(child_gpu_gate_buffer_mib=100, gpu_stop_avail_mib=100),
        run_root=tmp_path / "run",
        results_dir=tmp_path / "results",
        repository_root=tmp_path,
        resource_provider=MixedLaneResources(),
    )
    latest_gpu = SimpleNamespace(
        live=True, job_id="gpu-child", pid=101, is_gpu=True,
        started_at=1.0, attempt=1, process=object(),
    )
    latest_cpu = SimpleNamespace(
        live=True, job_id="cpu-child", pid=202, is_gpu=False,
        started_at=2.0, attempt=1, process=object(),
    )

    coordinator._sample_snapshot([latest_gpu, latest_cpu], [], 2)

    assert coordinator.last_resource_snapshot["latest_child_job_id"] == "cpu-child"
    assert coordinator.last_resource_snapshot["latest_child_vram_mib"] == 0
    assert coordinator.admission.gpu_gate_closed is False
    # The ledger's controller associates the CPU-derived footprint with the
    # separately selected latest GPU job when updating the GPU gate.
    assert coordinator.admission.gpu_gate_lock_job_id is None


def test_gate_transition_logs_close_then_external_reopen_in_source_order(tmp_path, caplog):
    import logging

    from simple_adp.runner.config import RunnerSettings
    from simple_adp.runner.resources import (
        GPUMemorySample, HostMemorySample, ProcessMemorySample,
    )
    from simple_adp.runner.scheduler import SuiteCoordinator

    class TransitionResources:
        def sample_host(self):
            return HostMemorySample(16_000, 1_000, 93.75, 0, 0, 0.0)

        def sample_gpu(self, _device_index):
            return GPUMemorySample(1_000, 950, 50, 95.0)

        def process_tree(self, pid):
            return frozenset({pid})

        def sample_child(self, _pid, **_kwargs):
            return ProcessMemorySample(host_rss_mib=10, gpu_vram_mib=10)

        def sample_external_host(self, _excluded_pids):
            return ProcessMemorySample()

        def sample_external_gpu(self, _device_index, _excluded_pids):
            return ProcessMemorySample()

    coordinator = SuiteCoordinator(
        SimpleNamespace(run_signature="gate-log-signature", runs=()),
        RunnerSettings(
            host_ram_stop_avail_mib=1_024, gpu_stop_avail_mib=100,
            child_host_gate_buffer_mib=500, child_gpu_gate_buffer_mib=100,
        ),
        run_root=tmp_path / "run", results_dir=tmp_path / "results",
        repository_root=tmp_path, resource_provider=TransitionResources(),
    )
    coordinator.memory.values.update({
        "external_host_mib": 1_000,
        "external_gpu_mib": 1_000,
        "external_host_pids": [80],
        "external_gpu_pids": [81],
    })
    runtime = SimpleNamespace(
        live=True, job_id="active-gpu", pid=42, is_gpu=True,
        started_at=1.0, attempt=1, process=object(),
    )

    with caplog.at_level(logging.INFO, logger="simple_adp.runner"):
        coordinator._sample_snapshot([runtime], [], 1)

    gate_messages = [record.getMessage() for record in caplog.records if "[GATE]" in record.getMessage()]
    assert [message.split("] ", 1)[1].split(" admission", 1)[0] for message in gate_messages] == [
        "gpu", "host", "gpu", "host",
    ]
    assert "gpu admission closed job=active-gpu" in gate_messages[0]
    assert "host admission closed job=active-gpu" in gate_messages[1]
    assert "gpu admission reopened by external non-python memory drop" in gate_messages[2]
    assert "host admission reopened by external non-python memory drop" in gate_messages[3]
