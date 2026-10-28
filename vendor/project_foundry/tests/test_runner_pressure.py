import os
import signal
import sys
import time

import pytest

import simple_adp.runner.pressure_wrapper as pressure_module
from simple_adp.runner.config import StandalonePressureSettings
from simple_adp.runner.pressure_wrapper import PressureAwareWrapper, PressureSample, _gpu_pressure_sample


def _sample(host, gpu, host_external=1000.0, gpu_external=1000.0):
    return PressureSample(
        host_percent=host,
        host_used_mib=host_external,
        swap_percent=70.0,
        gpu_percent=gpu,
        gpu_used_mib=gpu_external,
        python_host_mib=0.0,
        python_gpu_mib=0.0,
    )


def _mock_wrapper_samples(monkeypatch, samples):
    monkeypatch.setattr(pressure_module, "_sample_host_gpu_pressure", lambda _device: next(samples))
    monkeypatch.setattr(pressure_module, "_sample_python_usage", lambda _device: (0.0, 0.0))


def _mock_live_pid(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(pressure_module.psutil, "Process", lambda pid: SimpleNamespace(pid=pid))


def test_standalone_pressure_defaults_validation_and_missing_gpu_is_zero(monkeypatch):
    settings = StandalonePressureSettings()
    assert (settings.host_pressure_pct, settings.host_resume_pct) == (90.0, 85.0)
    assert (settings.gpu_pressure_pct, settings.gpu_resume_pct) == (90.0, 85.0)
    assert settings.external_memory_drop_mib == 500.0
    assert settings.poll_seconds == 0.5
    with pytest.raises(ValueError):
        StandalonePressureSettings(host_pressure_pct=101)
    with pytest.raises(ValueError):
        StandalonePressureSettings(gpu_resume_pct=91)
    monkeypatch.setattr(pressure_module.subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert _gpu_pressure_sample(0) == (0.0, 0.0)


def test_standalone_wrapper_requires_both_resume_domains_and_keeps_child_identity(monkeypatch):
    _mock_live_pid(monkeypatch)
    events = []
    samples = iter((_sample(91, 91), _sample(80, 90), _sample(80, 85), _sample(80, 80)))

    class FakeChild:
        pid = 9001
        def __init__(self):
            self.polls = 0
        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 6 else None

    children = []
    def launch(*args, **kwargs):
        child = FakeChild()
        children.append(child)
        return child

    monkeypatch.setattr(pressure_module.subprocess, "Popen", launch)
    _mock_wrapper_samples(monkeypatch, samples)
    monkeypatch.setattr(pressure_module, "_pause_child", lambda child, paused: events.append((child.pid, paused)))
    wrapper = PressureAwareWrapper(["probe"], StandalonePressureSettings(), sleeper=lambda _seconds: None)
    assert wrapper.run() == 0
    assert len(children) == 1
    assert events == [(9001, True), (9001, False)]
    assert wrapper.process is None


def test_standalone_missing_gpu_sensor_counts_as_ready_when_host_resumes(monkeypatch):
    from types import SimpleNamespace

    _mock_live_pid(monkeypatch)
    host_samples = iter((91.0, 85.0, 80.0))
    events = []

    class FakeChild:
        pid = 9010

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 6 else None

    child = FakeChild()
    monkeypatch.setattr(pressure_module.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(
        pressure_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=next(host_samples), used=1000 * pressure_module.MIB),
    )
    monkeypatch.setattr(
        pressure_module.psutil,
        "swap_memory",
        lambda: SimpleNamespace(percent=0.0),
    )
    monkeypatch.setattr(pressure_module.psutil, "process_iter", lambda *_a, **_k: iter(()))
    monkeypatch.setattr(
        pressure_module.subprocess,
        "check_output",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi unavailable")),
    )
    monkeypatch.setattr(
        pressure_module,
        "_pause_child",
        lambda process, paused: events.append((process.pid, paused)),
    )

    wrapper = PressureAwareWrapper(
        ["probe"], StandalonePressureSettings(), sleeper=lambda _seconds: None,
    )
    assert wrapper.run() == 0
    assert events == [(9010, True), (9010, False)]


def test_standalone_wrapper_requires_both_domains_even_when_each_has_external_drop(monkeypatch):
    _mock_live_pid(monkeypatch)
    events = []
    samples = iter((
        _sample(91, 80, 1000, 1000),  # host pressure pauses; record both external peaks
        _sample(90, 90, 500, 1000),  # host drop qualifies, GPU domain still blocks resume
        _sample(90, 90, 1000, 500),  # GPU drop qualifies, host domain still blocks resume
        _sample(90, 90, 500, 500),   # both domains now qualify by their 500 MiB drops
    ))

    class FakeChild:
        pid = 9002

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 5 else None

    child = FakeChild()
    monkeypatch.setattr(pressure_module.subprocess, "Popen", lambda *_a, **_k: child)
    _mock_wrapper_samples(monkeypatch, samples)
    monkeypatch.setattr(pressure_module, "_pause_child", lambda process, paused: events.append((process.pid, paused)))
    wrapper = PressureAwareWrapper(["probe"], StandalonePressureSettings(), sleeper=lambda _seconds: None)

    assert wrapper.run() == 0
    assert events == [(9002, True), (9002, False)]
    assert wrapper.host_peak_external == wrapper.gpu_peak_external == 0


def test_standalone_wrapper_tracks_external_peaks_while_paused_and_resamples_immediately(monkeypatch, capsys):
    _mock_live_pid(monkeypatch)
    events = []
    samples = iter((
        _sample(91, 80, 1000, 1000),  # pause on host pressure
        _sample(90, 90, 1500, 1500),  # paused external usage reaches a new peak
        _sample(90, 90, 1000, 1000),  # 500 MiB drops from that peak in both domains
        _sample(80, 80, 1000, 1000),  # resumed child is sampled again
    ))

    class FakeChild:
        pid = 9005

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 6 else None

    child = FakeChild()
    sleeps = []
    monkeypatch.setattr(pressure_module.subprocess, "Popen", lambda *_a, **_k: child)
    _mock_wrapper_samples(monkeypatch, samples)
    monkeypatch.setattr(pressure_module, "_pause_child", lambda process, paused: events.append((process.pid, paused)))
    wrapper = PressureAwareWrapper(
        ["probe"], StandalonePressureSettings(), sleeper=lambda seconds: sleeps.append(seconds),
    )

    assert wrapper.run() == 0
    assert events == [(9005, True), (9005, False)]
    assert sleeps == [0.5, 0.5]
    assert wrapper.host_peak_external == wrapper.gpu_peak_external == 0
    output = capsys.readouterr().out
    assert "host_drop=500.0 gpu_drop=500.0 host=90.00 swap=70.00 gpu=90.00" in output
    assert output.count("[PRESSURE] Pause requested.") == 1
    assert "[PRESSURE] host=90.0%" not in output


def test_standalone_wrapper_pressure_threshold_is_strict_and_swap_is_telemetry_only(monkeypatch):
    _mock_live_pid(monkeypatch)
    events = []

    class FakeChild:
        pid = 9003

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 2 else None

    child = FakeChild()
    monkeypatch.setattr(pressure_module.subprocess, "Popen", lambda *_a, **_k: child)
    # Equality to both 90% pressure limits must not pause, and swap is not
    # part of the standalone wrapper's pressure decision.
    monkeypatch.setattr(
        pressure_module, "_sample_host_gpu_pressure",
        lambda _device: PressureSample(90, 1000, 100, 90, 1000, 0, 0),
    )
    monkeypatch.setattr(pressure_module, "_pause_child", lambda process, paused: events.append((process.pid, paused)))
    wrapper = PressureAwareWrapper(["probe"], StandalonePressureSettings(), sleeper=lambda _seconds: None)

    assert wrapper.run() == 0
    assert events == []


def test_wrapper_samples_python_usage_only_when_pausing_or_waiting_to_resume(monkeypatch):
    _mock_live_pid(monkeypatch)
    samples = iter((_sample(40, 40), _sample(91, 40), _sample(80, 80), _sample(40, 40)))
    python_samples = []
    events = []
    call_order = []

    class Child:
        pid = 9030

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 6 else None

    child = Child()
    monkeypatch.setattr(pressure_module.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(pressure_module, "_sample_host_gpu_pressure", lambda _device: next(samples))
    monkeypatch.setattr(
        pressure_module, "_sample_python_usage",
        lambda _device: (python_samples.append("sample") or call_order.append("python") or (0.0, 0.0)),
    )
    monkeypatch.setattr(
        pressure_module, "_pause_child",
        lambda process, paused: (events.append(paused), call_order.append("pause" if paused else "resume")),
    )
    wrapper = PressureAwareWrapper(["probe"], StandalonePressureSettings(), sleeper=lambda _seconds: None)

    assert wrapper.run() == 0
    assert events == [True, False]
    assert python_samples == ["sample", "sample"]
    assert call_order[:2] == ["pause", "python"]


def test_standalone_wrapper_pause_error_is_reported_and_immediately_resampled(monkeypatch, capsys):
    _mock_live_pid(monkeypatch)
    class Child:
        pid = 9006

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 5 else None

    child = Child()
    samples = iter((_sample(91, 80), _sample(80, 80), _sample(80, 80)))
    sleeps = []
    monkeypatch.setattr(pressure_module.subprocess, "Popen", lambda *_a, **_k: child)
    _mock_wrapper_samples(monkeypatch, samples)

    def fail_pause(_child, paused):
        assert paused is True
        raise OSError("process group disappeared")

    monkeypatch.setattr(pressure_module, "_pause_child", fail_pause)
    wrapper = PressureAwareWrapper(
        ["probe"], StandalonePressureSettings(), sleeper=lambda seconds: sleeps.append(seconds),
    )

    assert wrapper.run() == 0
    assert "Failed to terminate for pause: process group disappeared" in capsys.readouterr().out
    assert sleeps == [0.5]


def test_standalone_wrapper_clamps_poll_sleep_to_one_tenth_second(monkeypatch):
    _mock_live_pid(monkeypatch)
    class Child:
        pid = 9007

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 3 else None

    child = Child()
    sleeps = []
    monkeypatch.setattr(pressure_module.subprocess, "Popen", lambda *_a, **_k: child)
    _mock_wrapper_samples(monkeypatch, iter((_sample(10, 10), _sample(10, 10))))
    wrapper = PressureAwareWrapper(
        ["probe"], StandalonePressureSettings(poll_seconds=0.01),
        sleeper=lambda seconds: sleeps.append(seconds),
    )

    assert wrapper.run() == 0
    assert sleeps == [0.1, 0.1]


def test_standalone_posix_pause_resolves_process_group_before_signaling(monkeypatch):
    events = []
    child = type("Child", (), {"pid": 4321, "poll": lambda _self: None})()

    monkeypatch.setattr(
        pressure_module.os, "getpgid",
        lambda pid: events.append(("getpgid", pid)) or 8765,
    )
    monkeypatch.setattr(
        pressure_module.os, "killpg",
        lambda pgid, sig: events.append(("killpg", pgid, sig)),
    )

    pressure_module._pause_child(child, True)
    pressure_module._pause_child(child, False)

    assert events == [
        ("getpgid", child.pid), ("killpg", 8765, pressure_module.signal.SIGSTOP),
        ("getpgid", child.pid), ("killpg", 8765, pressure_module.signal.SIGCONT),
    ]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signals are Linux-specific here")
def test_standalone_posix_wrapper_stops_and_resumes_a_real_child_group(monkeypatch, tmp_path):
    """Exercise real STOP/CONT signaling while keeping resource inputs deterministic."""
    marker = tmp_path / "child-finished"
    samples = iter((_sample(91, 10),))

    def sample_once_then_ready(_device):
        try:
            return next(samples)
        except StopIteration:
            return _sample(80, 10)

    monkeypatch.setattr(
        pressure_module,
        "_sample_host_gpu_pressure",
        sample_once_then_ready,
    )
    monkeypatch.setattr(pressure_module, "_sample_python_usage", lambda _device: (0.0, 0.0))

    real_killpg = os.killpg
    sent_signals = []

    def record_real_signal(process_group, signum):
        sent_signals.append(signum)
        return real_killpg(process_group, signum)

    monkeypatch.setattr(pressure_module.os, "killpg", record_real_signal)
    child_script = (
        "import pathlib, sys, time; "
        "time.sleep(0.35); pathlib.Path(sys.argv[1]).write_text('finished')"
    )
    wrapper = PressureAwareWrapper(
        [sys.executable, "-c", child_script, str(marker)],
        StandalonePressureSettings(poll_seconds=0.01),
        sleeper=time.sleep,
    )

    assert wrapper.run() == 0
    assert sent_signals == [signal.SIGSTOP, signal.SIGCONT]
    assert marker.read_text() == "finished"


def test_standalone_wrapper_checks_live_pid_after_poll_before_sampling(monkeypatch):
    events = []

    class Child:
        pid = 4322

        def poll(self):
            events.append("poll")
            return None

        def wait(self):
            events.append("wait")
            return 7

    child = Child()
    monkeypatch.setattr(pressure_module.subprocess, "Popen", lambda *_a, **_k: child)

    def process(pid):
        events.append(("Process", pid))
        raise pressure_module.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(pressure_module.psutil, "Process", process)
    monkeypatch.setattr(
        pressure_module, "_sample_host_gpu_pressure",
        lambda *_args: pytest.fail("vanished process must exit before pressure sampling"),
    )
    wrapper = PressureAwareWrapper(["probe"], StandalonePressureSettings())

    assert wrapper.run() == 7
    assert events == [("Process", child.pid), "wait"]


@pytest.mark.parametrize("paused", [False, True])
def test_standalone_pause_helper_does_not_signal_child_that_exited_at_boundary(monkeypatch, paused):
    class ExitedChild:
        def poll(self):
            return 23

    monkeypatch.setattr(
        pressure_module.os, "getpgid",
        lambda _pid: pytest.fail("must not inspect an exited child's process group"),
    )
    monkeypatch.setattr(
        pressure_module.os, "killpg",
        lambda *_args: pytest.fail("must not signal an exited child's process group"),
    )
    monkeypatch.setattr(
        pressure_module.psutil, "Process",
        lambda *_args: pytest.fail("must not inspect an exited child's process tree"),
    )

    pressure_module._pause_child(ExitedChild(), paused)


def test_standalone_wrapper_relaunches_child_that_exits_while_pressure_paused(monkeypatch):
    _mock_live_pid(monkeypatch)
    samples = iter((_sample(91, 80), _sample(80, 80), _sample(80, 80), _sample(80, 80)))
    launches = []
    pause_events = []

    class Child:
        def __init__(self, pid, exit_on_poll):
            self.pid = pid
            self.poll_count = 0
            self.exit_on_poll = exit_on_poll

        def poll(self):
            self.poll_count += 1
            return 17 if self.poll_count >= self.exit_on_poll else None

    def launch(*_args, **_kwargs):
        child = Child(9011 + len(launches), 2 if not launches else 3)
        launches.append(child)
        return child

    monkeypatch.setattr(pressure_module.subprocess, "Popen", launch)
    _mock_wrapper_samples(monkeypatch, samples)
    monkeypatch.setattr(
        pressure_module, "_pause_child",
        lambda child, paused: pause_events.append((child.pid, paused)),
    )
    wrapper = PressureAwareWrapper(["probe"], StandalonePressureSettings(), sleeper=lambda _seconds: None)

    assert wrapper.run() == 17
    assert [child.pid for child in launches] == [9011, 9012]
    assert pause_events == [(9011, True)]


def test_standalone_wrapper_interrupt_resumes_paused_child_before_tree_termination(monkeypatch):
    _mock_live_pid(monkeypatch)
    events = []

    class FakeChild:
        pid = 9004

        def poll(self):
            return None

    child = FakeChild()
    monkeypatch.setattr(pressure_module.subprocess, "Popen", lambda *_a, **_k: child)
    _mock_wrapper_samples(monkeypatch, iter((_sample(91, 80), _sample(90, 80))))
    monkeypatch.setattr(
        pressure_module, "_pause_child",
        lambda process, paused: events.append(("continue" if not paused else "pause", process.pid)),
    )
    monkeypatch.setattr(
        pressure_module, "terminate_standalone_pressure_tree",
        lambda process: events.append(("terminate_tree", process.pid)),
    )

    def interrupt(_seconds):
        raise KeyboardInterrupt

    wrapper = PressureAwareWrapper(["probe"], StandalonePressureSettings(), sleeper=interrupt)

    assert wrapper.run() == 130
    assert events == [
        ("pause", 9004),
        ("continue", 9004),
        ("terminate_tree", 9004),
    ]


@pytest.mark.skipif(os.name != "posix", reason="Linux process-group integration contract")
def test_standalone_wrapper_stops_and_continues_a_real_child_process_group(monkeypatch, capsys):
    import sys

    samples = iter((_sample(91, 40), _sample(80, 80)))

    def sample(_device):
        try:
            return next(samples)
        except StopIteration:
            return _sample(40, 40)

    monkeypatch.setattr(pressure_module, "_sample_host_gpu_pressure", sample)
    monkeypatch.setattr(pressure_module, "_sample_python_usage", lambda _device: (0.0, 0.0))
    wrapper = PressureAwareWrapper(
        [sys.executable, "-c", "import time; time.sleep(0.25)"],
        StandalonePressureSettings(poll_seconds=0.01),
    )

    assert wrapper.run() == 0
    output = capsys.readouterr().out
    assert "[PRESSURE] Pause requested. host=91.00%>90.0" in output
    assert "[STATE] admission gate reopened" in output


def test_standalone_signal_handler_terminates_immediately_without_resume(monkeypatch):
    events = []

    class Child:
        pid = 9005

    wrapper = PressureAwareWrapper(["probe"], StandalonePressureSettings())
    wrapper.process = Child()
    wrapper.paused = True
    monkeypatch.setattr(
        pressure_module, "terminate_standalone_pressure_tree",
        lambda process: events.append(("terminate_tree", process.pid)),
    )
    monkeypatch.setattr(
        pressure_module, "_pause_child",
        lambda process, paused: events.append(("pause" if paused else "continue", process.pid)),
    )

    with pytest.raises(SystemExit) as error:
        wrapper._interrupt(pressure_module.signal.SIGTERM, None)

    assert error.value.code == 130
    assert events == [("terminate_tree", 9005)]


def test_standalone_wrapper_rejects_empty_command():
    with pytest.raises(ValueError, match="command is required"):
        PressureAwareWrapper([], StandalonePressureSettings())


def test_standalone_cli_rejects_empty_command_with_source_error():
    with pytest.raises(SystemExit, match="No command provided"):
        pressure_module.main([])


def test_standalone_python_gpu_usage_filters_by_process_name_only(monkeypatch):
    from types import SimpleNamespace

    class PythonProcess:
        pid = 10
        info = {"pid": 10, "name": "python3", "memory_info": SimpleNamespace(rss=1024)}

        def memory_info(self):
            return self.info["memory_info"]

    monkeypatch.setattr(pressure_module.psutil, "virtual_memory", lambda: SimpleNamespace(percent=50.0, used=1000 * pressure_module.MIB))
    monkeypatch.setattr(pressure_module.psutil, "swap_memory", lambda: SimpleNamespace(percent=0.0))
    monkeypatch.setattr(pressure_module.psutil, "process_iter", lambda *_args, **_kwargs: iter([PythonProcess()]))
    monkeypatch.setattr(pressure_module, "_gpu_pressure_sample", lambda _device: (10.0, 1000.0))
    monkeypatch.setattr(
        pressure_module.subprocess, "check_output",
        lambda *_args, **_kwargs: b"custom_runtime,200\npython,300\n",
    )
    sample = pressure_module.sample_pressure(0)
    assert sample.python_host_mib == pytest.approx(1024 / pressure_module.MIB)
    assert sample.python_gpu_mib == 300.0


def test_standalone_python_host_sampler_skips_only_documented_process_errors(monkeypatch):
    from types import SimpleNamespace

    class Process:
        def __init__(self, info=None, error=None):
            self._info = info
            self._error = error

        @property
        def info(self):
            if self._error:
                raise self._error
            return self._info

    processes = [
        Process(error=pressure_module.psutil.AccessDenied(1)),
        Process({"name": "python3", "memory_info": SimpleNamespace(rss=2 * pressure_module.MIB)}),
    ]
    monkeypatch.setattr(pressure_module.psutil, "virtual_memory", lambda: SimpleNamespace(percent=20.0, used=1000 * pressure_module.MIB))
    monkeypatch.setattr(pressure_module.psutil, "swap_memory", lambda: SimpleNamespace(percent=0.0))
    monkeypatch.setattr(pressure_module.psutil, "process_iter", lambda attrs: iter(processes))
    monkeypatch.setattr(pressure_module, "_gpu_pressure_sample", lambda _device: (0.0, 0))
    monkeypatch.setattr(pressure_module.subprocess, "check_output", lambda *_args, **_kwargs: b"")

    sample = pressure_module.sample_pressure(0)

    assert sample.python_host_mib == 2.0

    # The source uses the cached memory_info value directly rather than
    # issuing a second Process.memory_info() lookup when it is missing.
    monkeypatch.setattr(
        pressure_module.psutil, "process_iter",
        lambda _attrs: iter([Process({"name": "python", "memory_info": None})]),
    )
    with pytest.raises(AttributeError):
        pressure_module.sample_pressure(0)


def test_standalone_python_host_sampler_propagates_unspecified_record_failures(monkeypatch):
    from types import SimpleNamespace

    class Process:
        @property
        def info(self):
            raise RuntimeError("unexpected process inspection failure")

    monkeypatch.setattr(pressure_module.psutil, "virtual_memory", lambda: SimpleNamespace(percent=20.0, used=1000 * pressure_module.MIB))
    monkeypatch.setattr(pressure_module.psutil, "swap_memory", lambda: SimpleNamespace(percent=0.0))
    monkeypatch.setattr(pressure_module.psutil, "process_iter", lambda attrs: iter([Process()]))

    with pytest.raises(RuntimeError, match="unexpected process inspection failure"):
        pressure_module.sample_pressure(0)


def test_standalone_gpu_pressure_sampler_matches_source_query_and_clamping(monkeypatch):
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return b"100, 125.9\n"

    monkeypatch.setattr(pressure_module.subprocess, "check_output", check_output)
    assert _gpu_pressure_sample(2) == (100.0, 125)
    assert calls == [(
        ["nvidia-smi", "--id=2", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        {"stderr": pressure_module.subprocess.DEVNULL},
    )]

    monkeypatch.setattr(
        pressure_module.subprocess, "check_output",
        lambda *_args, **_kwargs: b"100, -2.9\n",
    )
    assert _gpu_pressure_sample(0) == (0.0, -2)


def test_standalone_python_gpu_sampler_uses_source_two_column_query(monkeypatch):
    queries = []

    def check_output(command, **kwargs):
        queries.append((command, kwargs))
        if "query-gpu=" in command[2]:
            return b"1000, 50\n"
        return b"python3, 12\ncustom_python_worker, 8\nother, 30\npython, 1.5\n"

    monkeypatch.setattr(pressure_module.subprocess, "check_output", check_output)
    monkeypatch.setattr(pressure_module.psutil, "virtual_memory", lambda: type("V", (), {"percent": 10.0, "used": 1})())
    monkeypatch.setattr(pressure_module.psutil, "swap_memory", lambda: type("S", (), {"percent": 0.0})())
    monkeypatch.setattr(pressure_module.psutil, "process_iter", lambda *_args, **_kwargs: iter(()))

    sample = pressure_module.sample_pressure(3)
    assert sample.python_gpu_mib == 20.0
    assert queries[1] == (
        ["nvidia-smi", "--id=3", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
        {"stderr": pressure_module.subprocess.DEVNULL},
    )


def test_standalone_windows_pause_resumes_descendants_then_root_best_effort(monkeypatch):
    events = []

    class Process:
        def __init__(self, pid, *, fails=False):
            self.pid = pid
            self.fails = fails

        def suspend(self):
            events.append(("suspend", self.pid))
            if self.fails:
                raise pressure_module.psutil.AccessDenied(self.pid)

        def resume(self):
            events.append(("resume", self.pid))

    child_first = Process(30)
    child_second = Process(20, fails=True)
    class Root:
        def children(self, recursive):
            assert recursive is True
            return [child_first, child_second]

        def suspend(self):
            events.append(("suspend", 10))

        def resume(self):
            events.append(("resume", 10))

    from types import SimpleNamespace

    monkeypatch.setattr(pressure_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(pressure_module.psutil, "Process", lambda _pid: Root())
    child = type("Child", (), {"pid": 10, "poll": lambda _self: None})()

    pressure_module._pause_child(child, True)
    pressure_module._pause_child(child, False)

    assert events == [
        ("suspend", 30), ("suspend", 20), ("suspend", 10),
        ("resume", 30), ("resume", 20), ("resume", 10),
    ]
