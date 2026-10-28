from simple_adp.runner import watchdog


def test_watchdog_master_liveness_matches_kill_zero_semantics(monkeypatch):
    import errno

    calls = []

    def kill(pid, signal):
        calls.append((pid, signal))
        if pid == 11:
            return None  # kill -0 succeeds for an unreaped zombie too.
        raise OSError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(watchdog.os, "kill", kill)
    assert watchdog.master_alive(11) is True
    assert watchdog.master_alive(12) is False
    assert calls == [(11, 0), (12, 0)]


def test_linux_watchdog_rescans_descendants_after_grace_before_kill(tmp_path, monkeypatch):
    events = []

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.enumerations = 0

        def children(self, recursive=False):
            assert recursive is True
            self.enumerations += 1
            if self.enumerations == 1:
                return [early_child]
            return [early_child, late_child]

        def terminate(self):
            events.append(("terminate", self.pid))

        def kill(self):
            events.append(("kill", self.pid))

        def is_running(self):
            return True

    early_child = Process(11)
    late_child = Process(12)
    root = Process(10)
    monkeypatch.setattr(watchdog.psutil, "Process", lambda _pid: root)
    alive = iter((True, False))

    def master_alive(_pid):
        events.append("master_check")
        return next(alive)

    monkeypatch.setattr(watchdog, "master_alive", master_alive)
    monkeypatch.setattr(watchdog.time, "sleep", lambda seconds: events.append(("sleep", seconds)))

    assert watchdog.monitor(
        1, 10, tmp_path / "watchdog.log", initial_delay=2,
        poll_seconds=3, grace_seconds=15,
    ) == 0

    assert root.enumerations == 2
    assert events == [
        ("sleep", 2),
        ("sleep", 3),
        "master_check",
        ("sleep", 3),
        "master_check",
        ("terminate", 11),
        ("terminate", 10),
        ("sleep", 15),
        ("kill", 12),
        ("kill", 11),
        ("kill", 10),
    ]
    assert "[WATCHDOG] master 1 gone; cleaning python children" in (
        tmp_path / "watchdog.log"
    ).read_text()


def test_linux_watchdog_reaps_real_orphaned_process_tree(tmp_path):
    import multiprocessing
    import os
    import subprocess
    import sys

    import psutil

    if os.name != "posix":
        return

    master = subprocess.Popen(["sleep", "60"])
    child_code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import time; time.sleep(60)']); "
        "print(p.pid, flush=True); time.sleep(60)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    grandchild_pid = int(child.stdout.readline().strip())
    monitor = multiprocessing.Process(
        target=watchdog.monitor,
        args=(master.pid, child.pid, tmp_path / "watchdog.log"),
        kwargs={"initial_delay": 0.01, "poll_seconds": 0.01, "grace_seconds": 0.05},
    )
    monitor.start()
    master.terminate()
    master.wait(timeout=5)
    try:
        monitor.join(timeout=5)
        assert not monitor.is_alive()
        assert monitor.exitcode == 0
        child.wait(timeout=5)
        try:
            grandchild = psutil.Process(grandchild_pid)
            assert not grandchild.is_running() or grandchild.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
        assert "[WATCHDOG] master" in (tmp_path / "watchdog.log").read_text()
    finally:
        if monitor.is_alive():
            monitor.terminate()
            monitor.join(timeout=2)
        for pid in (grandchild_pid, child.pid):
            try:
                process = psutil.Process(pid)
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    process.kill()
            except psutil.Error:
                pass
        child.stdout.close()
