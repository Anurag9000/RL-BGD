import hashlib
from types import SimpleNamespace

import pytest

import simple_adp.runner.runtime_tuning as tuning


def test_cpu_range_partition_hash_and_thread_budget_contracts():
    assert tuning.parse_cpu_list("0-2, 6, 9-7") == (0, 1, 2, 6, 7, 8, 9)
    cpus = (2, 4, 6, 8, 10)
    assert tuning.partition_cpus(cpus, 2, 0) == (2, 4, 6)
    assert tuning.partition_cpus(cpus, 2, 1) == (8, 10)
    assert tuning.partition_cpus(cpus, 99, 99) == (10,)
    expected = int.from_bytes(hashlib.sha1(b"job-key").digest()[:8], "big") % 7
    assert tuning.stable_cpu_slot("job-key", 7) == expected
    assert tuning.resolve_thread_budget(16, None) == 16
    assert tuning.resolve_thread_budget(16, 1) == 16
    assert tuning.resolve_thread_budget(16, 5) == 3
    assert tuning.resolve_num_workers(16) == 0


@pytest.mark.parametrize("explicit", [0, -2])
def test_nonpositive_explicit_concurrency_hint_falls_back_to_environment(monkeypatch, explicit):
    monkeypatch.setenv("TABULAR_CPU_JOB_CONCURRENCY", "4")
    assert tuning.current_concurrency_hint(explicit) == 4


def test_positive_explicit_concurrency_hint_precedes_environment_and_invalid_environment_is_absent(monkeypatch):
    monkeypatch.setenv("TABULAR_CPU_JOB_CONCURRENCY", "bad")
    assert tuning.current_concurrency_hint(2) == 2
    assert tuning.current_concurrency_hint(None) is None


def test_detected_cpus_uses_affinity_and_frequency_source_precedence(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(tuning.os, "sched_getaffinity", lambda _pid: {7, 4, 3, 2, 1})
    frequencies = {
        (7, "cpuinfo_max_freq"): "1000000",  # Must win even though scaling says 4 GHz.
        (7, "scaling_max_freq"): "4000000",
        (2, "cpuinfo_max_freq"): "1500000.5",  # Invalid integer falls through to the next source.
        (2, "scaling_max_freq"): "3500000",
        (3, "scaling_max_freq"): "2500000",
        (4, "scaling_max_freq"): "2500000",  # Equal frequency breaks by CPU ID.
        (4, "base_frequency"): "5000000",
    }

    def read_frequency(path, *args, **kwargs):
        parts = path.parts
        cpu = int(next(part[3:] for part in parts if part.startswith("cpu") and part[3:].isdigit()))
        name = path.name
        try:
            return frequencies[(cpu, name)]
        except KeyError as exc:
            raise OSError("frequency file unavailable") from exc

    monkeypatch.setattr(Path, "read_text", read_frequency)
    assert tuning.detected_cpus() == (2, 3, 4, 7, 1)


@pytest.mark.parametrize(("cpu_count", "expected"), [(4, (0, 1, 2, 3)), (None, (0,))])
def test_detected_cpus_falls_back_to_online_ids_or_one_cpu(monkeypatch, cpu_count, expected):
    from pathlib import Path

    monkeypatch.setattr(tuning.os, "sched_getaffinity", lambda _pid: (_ for _ in ()).throw(OSError("affinity unavailable")))
    monkeypatch.setattr(tuning.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no frequency data")))
    assert tuning.detected_cpus() == expected


def test_detected_cpus_falls_back_for_unexpected_affinity_errors(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(
        tuning.os, "sched_getaffinity",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("unexpected affinity failure")),
    )
    monkeypatch.setattr(tuning.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        Path, "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no frequency data")),
    )
    assert tuning.detected_cpus() == (0, 1)


def test_invalid_cpu_affinity_falls_back_best_effort_to_online_ids(monkeypatch):
    applied = []
    monkeypatch.setenv("TABULAR_CPU_AFFINITY_CPUS", "not-a-cpu-list")
    monkeypatch.setattr(tuning.os, "cpu_count", lambda: 3)
    monkeypatch.setattr(tuning, "_apply_affinity", lambda cpus: applied.append(cpus))
    tuning._apply_affinity_from_env()
    assert applied == [(0, 1, 2)]


def test_rejected_configured_affinity_retries_all_online_cpus(monkeypatch):
    attempts = []
    monkeypatch.setenv("TABULAR_CPU_AFFINITY_CPUS", "4-5")
    monkeypatch.setattr(tuning.os, "cpu_count", lambda: 3)

    def apply(cpus):
        attempts.append(cpus)
        return len(attempts) > 1

    monkeypatch.setattr(tuning, "_apply_affinity", apply)
    tuning._apply_affinity_from_env()
    assert attempts == [(4, 5), (0, 1, 2)]


def test_child_environment_partitions_without_mutating_parent_and_supports_shared_mode(monkeypatch):
    monkeypatch.setattr(tuning, "detected_cpus", lambda: (0, 1, 2, 3, 4, 5))
    monkeypatch.setattr(tuning, "_apply_affinity", lambda _cpus: pytest.fail("child environment must not mutate parent affinity"))
    partitioned = tuning.launcher_child_env(
        {"CUDA_VISIBLE_DEVICES": "0"}, job_key="", launch_budget=3,
        affinity_slot=1, device_index=0,
    )
    assert partitioned["TABULAR_CPU_AFFINITY_CPUS"] == "2,3"
    assert partitioned["OMP_NUM_THREADS"] == "2"
    assert partitioned["TABULAR_CPU_THREADS"] == "2"
    assert partitioned["TABULAR_CPU_CORES"] == "6"
    assert partitioned["TABULAR_CPU_JOB_CONCURRENCY"] == "3"
    assert "OMP_PROC_BIND" not in partitioned

    unbounded = tuning.launcher_child_env(
        {"CUDA_VISIBLE_DEVICES": "0"}, job_key="unbounded-job",
        launch_budget=10**9,
    )
    assert unbounded["OMP_NUM_THREADS"] == "1"
    # The helper sees the large sentinel; launch_job later normalizes this
    # metadata field to one for the child, as the ledger records.
    assert unbounded["TABULAR_CPU_JOB_CONCURRENCY"] == str(10**9)

    shared = tuning.launcher_child_env(
        {"CUDA_VISIBLE_DEVICES": "0", "TABULAR_CHILD_SHARED_CPU": "1"},
        job_key="ignored-in-shared-mode", launch_budget=3,
    )
    assert shared["TABULAR_CPU_AFFINITY_CPUS"] == "0,1,2,3,4,5"
    assert shared["OMP_NUM_THREADS"] == "6"
    assert shared["GOMP_CPU_AFFINITY"] == "0,1,2,3,4,5"
    assert shared["OMP_PROC_BIND"] == "spread"
    assert shared["OMP_PLACES"] == "cores"
    assert shared["KMP_AFFINITY"] == "granularity=fine,compact,1,0"
    assert shared["TABULAR_CPU_JOB_CONCURRENCY"] == "1"


def test_child_environment_uses_parent_affinity_as_fallback_but_prefers_base_env(monkeypatch):
    monkeypatch.setattr(tuning, "detected_cpus", lambda: (0, 1, 2, 3, 4, 5))
    monkeypatch.setenv("TABULAR_CPU_AFFINITY_CPUS", "2,3,4,5")

    inherited = tuning.launcher_child_env(
        {"CUDA_VISIBLE_DEVICES": "0"}, job_key="", launch_budget=2,
        affinity_slot=1,
    )
    explicit = tuning.launcher_child_env(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "TABULAR_CPU_AFFINITY_CPUS": "0,1,2,3",
        },
        job_key="",
        launch_budget=2,
        affinity_slot=1,
    )

    assert inherited["TABULAR_CPU_AFFINITY_CPUS"] == "4,5"
    assert explicit["TABULAR_CPU_AFFINITY_CPUS"] == "2,3"

    # Source behavior is an OR: an inherited truthy value still selects shared
    # mode even when the caller passes shared_cpu=False. The launcher later
    # rewrites the flag to "0", but these calculated thread/affinity controls
    # remain from shared mode.
    partitioned_with_inherited_shared = tuning.launcher_child_env(
        {"CUDA_VISIBLE_DEVICES": "0", "TABULAR_CHILD_SHARED_CPU": "1"},
        job_key="central-job", launch_budget=3, shared_cpu=False,
    )
    assert partitioned_with_inherited_shared["OMP_NUM_THREADS"] == "6"
    assert partitioned_with_inherited_shared["TABULAR_CPU_JOB_CONCURRENCY"] == "1"
    assert partitioned_with_inherited_shared["OMP_PROC_BIND"] == "spread"


def test_gpu_child_environment_preserves_single_mask_and_selects_from_multi_mask(monkeypatch):
    monkeypatch.setattr(tuning, "detected_cpus", lambda: (0, 1, 2, 3))

    single = tuning.launcher_child_env(
        {"CUDA_VISIBLE_DEVICES": " 2 ", "TRAINING_CONTROL_CPU_ONLY": "0"},
        job_key="single-mask", launch_budget=2, device_index=0, device_mode="gpu",
    )
    assert single["CUDA_VISIBLE_DEVICES"] == "2"
    assert single["TABULAR_BACKEND"] == "gpu"
    assert single["TABULAR_LOCAL_GPU_INDEX"] == "0"
    assert "TRAINING_CONTROL_CPU_ONLY" not in single

    multi = tuning.launcher_child_env(
        {"CUDA_VISIBLE_DEVICES": "0, 2,3"},
        job_key="multi-mask", launch_budget=2, device_index=2, device_mode="gpu",
    )
    assert multi["CUDA_VISIBLE_DEVICES"] == "2"
    assert multi["TABULAR_BACKEND"] == "gpu"
    assert multi["TABULAR_LOCAL_GPU_INDEX"] == "0"

    child = tuning.launcher_child_env({}, job_key="unmasked", launch_budget=2, device_index=3, device_mode="gpu")
    assert child["CUDA_VISIBLE_DEVICES"] == "3"

    for inherited in ("", " -1 "):
        with pytest.raises(RuntimeError, match="cannot override central CPU admission"):
            tuning.launcher_child_env(
                {"CUDA_VISIBLE_DEVICES": inherited},
                job_key="cpu-masked", launch_budget=2, device_index=3, device_mode="gpu",
            )

    with pytest.raises(ValueError, match="not in the inherited CUDA device mask"):
        tuning.launcher_child_env(
            {"CUDA_VISIBLE_DEVICES": "0,2"},
            job_key="missing-device", launch_budget=2, device_index=1, device_mode="gpu",
        )


def test_explicit_gpu_launch_rejects_cpu_admission_and_cpu_launch_masks_accelerators():
    with pytest.raises(RuntimeError, match="cannot override central CPU admission"):
        tuning.launcher_child_env(
            {"CUDA_VISIBLE_DEVICES": "0"}, job_key="cpu-only", cpu_only=True,
            device_mode="gpu",
        )

    cpu = tuning.launcher_child_env(
        {"CUDA_VISIBLE_DEVICES": "0"}, job_key="cpu-worker", device_mode="cpu",
    )
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu["HIP_VISIBLE_DEVICES"] == ""
    assert cpu["ROCR_VISIBLE_DEVICES"] == ""
    assert cpu["JAX_PLATFORMS"] == "cpu"
    assert cpu["JAX_PLATFORM_NAME"] == "cpu"
    assert cpu["TABULAR_BACKEND"] == "cpu"


def test_bootstrap_exports_full_thread_environment_without_child_partition(monkeypatch):
    import torch

    setter_calls = []

    monkeypatch.setattr(tuning, "maybe_reexec_systemd_scope", lambda **_kwargs: False)
    monkeypatch.setattr(tuning, "_apply_affinity_from_env", lambda: None)
    monkeypatch.setattr(tuning, "detected_cpus", lambda: (0, 1, 2, 3))

    def thread_setter(_count):
        setter_calls.append("threads")
        raise RuntimeError("injected setter failure")

    def interop_setter(_count):
        setter_calls.append("interop")

    monkeypatch.setattr(torch, "set_num_threads", thread_setter)
    monkeypatch.setattr(torch, "set_num_interop_threads", interop_setter)
    for key in ("TABULAR_CPU_JOB_CONCURRENCY", "TABULAR_CPU_AFFINITY_CPUS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TABULAR_CHILD_SHARED_CPU", "0")
    result = tuning.bootstrap_runtime("test-bootstrap", apply_os_tuning=False)
    assert setter_calls == ["threads", "interop"]
    assert result.cores == 4 and result.torch_threads == 4
    assert __import__("os").environ["OMP_PROC_BIND"] == "spread"
    assert __import__("os").environ["OMP_PLACES"] == "cores"
    assert __import__("os").environ["KMP_AFFINITY"] == "granularity=fine,scatter"
    assert __import__("os").environ["NUMEXPR_NUM_THREADS"] == "4"
    assert __import__("os").environ["TABULAR_CPU_THREADS"] == "4"
    assert __import__("os").environ["TABULAR_CPU_WORKERS"] == "0"


def test_bootstrap_attempts_systemd_scope_before_affinity_and_thread_tuning(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tuning, "maybe_reexec_systemd_scope",
        lambda *, label: calls.append(("systemd", label)) or False,
    )
    monkeypatch.setattr(tuning, "_apply_affinity_from_env", lambda: calls.append(("affinity", None)))
    monkeypatch.setattr(tuning, "detected_cpus", lambda: (0, 1))
    monkeypatch.setattr(tuning, "current_concurrency_hint", lambda: None)
    monkeypatch.setattr(tuning, "_apply_priority", lambda: calls.append(("priority", None)))
    monkeypatch.setattr(tuning, "_enforce_no_swap", lambda: calls.append(("no_swap", None)))
    import torch
    monkeypatch.setattr(torch, "set_num_threads", lambda _count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _count: None)

    tuning.bootstrap_runtime("worker-runtime", apply_os_tuning=True)

    assert calls[:2] == [("systemd", "worker-runtime"), ("affinity", None)]
    assert calls[2:] == [("priority", None), ("no_swap", None)]


@pytest.mark.parametrize(
    ("version_output", "expected"),
    [
        ("systemd 252 (test)", False),
        ("systemd 253 (test)", True),
        ("systemd 253 (test)\nextra version 300", True),
        ("other 300\nsystemd 253 (test)", False),
        ("systemd invalid", False),
    ],
)
@pytest.mark.parametrize("exec_error", [OSError("mock exec fallback"), RuntimeError("unexpected exec fallback")])
def test_systemd_scope_adds_zswap_gate_only_from_first_version_line(monkeypatch, version_output, expected, exec_error):
    captured = {}
    monkeypatch.setattr(tuning.os, "name", "posix")
    monkeypatch.delenv("TABULAR_SKIP_SYSTEMD_SCOPE", raising=False)
    monkeypatch.delenv("TABULAR_SYSTEMD_SCOPED", raising=False)
    monkeypatch.setattr(tuning, "shutil_which", lambda _name: "/usr/bin/systemd-run")
    monkeypatch.setattr(tuning, "detected_cpus", lambda: (1, 3))
    def run(*args, **kwargs):
        captured["run_args"] = args
        captured["run_kwargs"] = kwargs
        return SimpleNamespace(stdout=version_output)
    monkeypatch.setattr(tuning.subprocess, "run", run)
    def fail_exec(executable, argv, env):
        captured["executable"] = executable
        captured["command"] = argv
        captured["env"] = env
        raise exec_error
    monkeypatch.setattr(tuning.os, "execvpe", fail_exec)
    assert tuning.maybe_reexec_systemd_scope(["simple-adp-suite"]) is False
    assert captured["executable"] == "systemd-run"
    assert captured["command"][0] == "systemd-run"
    props = [captured["command"][index + 1] for index, value in enumerate(captured["command"][:-1]) if value == "-p"]
    assert ("MemoryZSwapMax=0" in props) is expected
    assert "MemorySwapMax=0" in props
    assert "AllowedCPUs=1,3" in props
    assert captured["run_kwargs"] == {
        "check": False,
        "stdout": tuning.subprocess.PIPE,
        "stderr": tuning.subprocess.DEVNULL,
        "text": True,
    }
    assert captured["env"]["TABULAR_SYSTEMD_SCOPED"] == "1"
    expected_properties = [
        "CPUWeight=10000", "StartupCPUWeight=10000", "IOWeight=10000",
        "StartupIOWeight=10000", "MemorySwapMax=0", "TasksMax=infinity",
        "CPUAccounting=yes", "IOAccounting=yes", "MemoryAccounting=yes",
        "TasksAccounting=yes",
    ]
    if expected:
        expected_properties.append("MemoryZSwapMax=0")
    expected_properties.append("AllowedCPUs=1,3")
    expected_command = [
        "systemd-run", "--user", "--scope", "--quiet", "--same-dir", "--collect",
        "--slice=app-mlps-training.slice", "--description=tabular:runner",
    ]
    for prop in expected_properties:
        expected_command.extend(["-p", prop])
    expected_command.extend([tuning.sys.executable, "simple-adp-suite"])
    assert captured["command"] == expected_command


def test_systemd_scope_preserves_python_module_invocation_on_reexec(monkeypatch):
    captured = {}
    monkeypatch.setattr(tuning.os, "name", "posix")
    monkeypatch.delenv("TABULAR_SKIP_SYSTEMD_SCOPE", raising=False)
    monkeypatch.delenv("TABULAR_SYSTEMD_SCOPED", raising=False)
    monkeypatch.setattr(tuning, "shutil_which", lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None)
    monkeypatch.setattr(tuning, "detected_cpus", lambda: (0,))
    monkeypatch.setattr(tuning.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout="systemd 252"))
    monkeypatch.setattr(tuning.sys, "argv", ["/repo/src/simple_adp/runner/launcher.py", "--help"])
    monkeypatch.setattr(
        tuning.sys,
        "orig_argv",
        [tuning.sys.executable, "-u", "-m", "simple_adp.runner.launcher", "--help"],
        raising=False,
    )

    def capture_exec(executable, argv, _env):
        captured["executable"] = executable
        captured["argv"] = argv
        raise OSError("mock exec fallback")

    monkeypatch.setattr(tuning.os, "execvpe", capture_exec)
    assert tuning.maybe_reexec_systemd_scope() is False
    assert captured["executable"] == "systemd-run"
    assert captured["argv"][-5:] == [
        tuning.sys.executable, "-u", "-m", "simple_adp.runner.launcher", "--help",
    ]


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("not_posix", False),
        ("skip_env", False),
        ("already_scoped", False),
        ("interactive_c", False),
        ("interactive_stdin", False),
        ("missing_systemd_run", False),
        ("exec_error", False),
    ],
)
def test_systemd_scope_reexec_skip_and_fallback_conditions(monkeypatch, scenario, expected):
    calls = []
    monkeypatch.setattr(tuning.os, "name", "posix")
    monkeypatch.delenv("TABULAR_SKIP_SYSTEMD_SCOPE", raising=False)
    monkeypatch.delenv("TABULAR_SYSTEMD_SCOPED", raising=False)
    argv = ["simple-adp-suite"]
    if scenario == "not_posix":
        monkeypatch.setattr(tuning.os, "name", "nt")
    elif scenario == "skip_env":
        monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    elif scenario == "already_scoped":
        monkeypatch.setenv("TABULAR_SYSTEMD_SCOPED", "1")
    elif scenario == "interactive_c":
        argv = ["-c"]
    elif scenario == "interactive_stdin":
        argv = ["-"]
    elif scenario == "missing_systemd_run":
        monkeypatch.setattr(tuning, "shutil_which", lambda _name: None)
    elif scenario == "exec_error":
        monkeypatch.setattr(tuning, "shutil_which", lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None)
        monkeypatch.setattr(tuning.os, "execvpe", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("exec failed")))

    monkeypatch.setattr(tuning, "detected_cpus", lambda: (0,))
    monkeypatch.setattr(tuning.subprocess, "run", lambda *_args, **_kwargs: calls.append("version-query"))
    if scenario != "exec_error" and scenario != "missing_systemd_run":
        monkeypatch.setattr(tuning, "shutil_which", lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None)

    assert tuning.maybe_reexec_systemd_scope(argv) is expected
    assert calls == []


def test_affinity_application_swallows_unexpected_os_errors(monkeypatch):
    monkeypatch.setattr(tuning.os, "sched_setaffinity", lambda *_args: (_ for _ in ()).throw(RuntimeError("affinity failed")))
    tuning._apply_affinity((1, 2))


def test_posix_priority_attempts_exact_controls_independently_and_best_effort(monkeypatch):
    calls = []

    def deny_nice(value):
        calls.append(("nice", value))
        raise PermissionError("nice denied")

    def record_run(command, **kwargs):
        calls.append(("ionice", command, kwargs))
        raise OSError("ionice unavailable")

    def deny_scheduler(pid, policy, parameter):
        calls.append(("scheduler", pid, policy, parameter.sched_priority))
        raise PermissionError("scheduler denied")

    monkeypatch.setattr(tuning.os, "name", "posix")
    monkeypatch.setattr(tuning.os, "nice", deny_nice)
    monkeypatch.setattr(tuning, "shutil_which", lambda name: "/usr/bin/ionice" if name == "ionice" else None)
    monkeypatch.setattr(tuning.subprocess, "run", record_run)
    monkeypatch.setattr(tuning.os, "SCHED_BATCH", 3, raising=False)
    monkeypatch.setattr(tuning.os, "sched_setscheduler", deny_scheduler)
    monkeypatch.setattr(tuning.os, "getpid", lambda: 4321)

    tuning._apply_priority()

    assert calls[0] == ("nice", -20)
    assert calls[1] == (
        "ionice", ["ionice", "-c2", "-n0", "-p", "4321"],
        {"check": False, "stdout": tuning.subprocess.DEVNULL, "stderr": tuning.subprocess.DEVNULL},
    )
    assert calls[2] == ("scheduler", 0, 3, 0)


def test_linux_no_swap_attempt_uses_mlockall_current_and_future_best_effort(monkeypatch):
    import ctypes

    class LibC:
        def __init__(self):
            self.flags = []

        def mlockall(self, flags):
            self.flags.append(flags)
            return -1

    libc = LibC()
    monkeypatch.setattr(tuning.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: libc)

    tuning._enforce_no_swap()

    assert libc.flags == [3]  # MCL_CURRENT | MCL_FUTURE; failure is deliberately ignored.


def test_windows_memory_priority_attempt_is_independent_of_priority_class(monkeypatch):
    import ctypes

    calls = []

    class Kernel:
        def GetCurrentProcess(self):
            return 7

        def SetPriorityClass(self, *_args):
            raise RuntimeError("priority class denied")

        def SetProcessInformation(self, *_args):
            calls.append("memory-priority")
            return 1

    monkeypatch.setattr(tuning.os, "name", "nt")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=Kernel()), raising=False)
    tuning._apply_priority()
    assert calls == ["memory-priority"]
