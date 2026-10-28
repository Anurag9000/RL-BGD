"""Detached master-liveness watcher for launcher-owned scheduler children."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import psutil

from .processes import terminate_process_tree


def master_alive(pid: int) -> bool:
    try:
        # The Linux shell reference uses `kill -0 PID`. That reports a zombie
        # as present until it is reaped, so do not substitute psutil's
        # is_running/status interpretation here.
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def monitor(master_pid: int, child_pid: int, log_path: str | Path, *, initial_delay: float = 2.0, poll_seconds: float = 3.0, grace_seconds: float = 15.0) -> int:
    log = Path(log_path)
    time.sleep(initial_delay)
    while True:
        time.sleep(poll_seconds)
        if not master_alive(master_pid):
            break
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(
                f"[WATCHDOG] master {master_pid} gone; cleaning python children\n"
            )
            stream.flush()
    except OSError:
        pass
    try:
        root = psutil.Process(child_pid)
    except psutil.Error:
        return 0
    if os.name == "nt":
        try:
            root.terminate()
        except psutil.Error:
            pass
        time.sleep(grace_seconds)
        try:
            descendants = root.children(recursive=True)
        except psutil.Error:
            descendants = []
        for process in sorted(descendants, key=lambda item: item.pid, reverse=True):
            try:
                process.kill()
            except psutil.Error:
                pass
    else:
        try:
            descendants = root.children(recursive=True)
        except psutil.Error:
            descendants = []
        for process in reversed(descendants):
            try:
                process.terminate()
            except psutil.Error:
                pass
        try:
            root.terminate()
        except psutil.Error:
            pass
        time.sleep(grace_seconds)
        # The shell reference walks the tree again for KILL. Children can
        # spawn during the grace interval, so the TERM snapshot is not enough.
        try:
            kill_descendants = root.children(recursive=True)
        except psutil.Error:
            kill_descendants = []
        for process in reversed(kill_descendants):
            try:
                process.kill()
            except psutil.Error:
                pass
    try:
        if root.is_running():
            root.kill()
    except psutil.Error:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=int, required=True)
    parser.add_argument("--child", type=int, required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args(argv)
    return monitor(args.master, args.child, args.log)


if __name__ == "__main__":
    raise SystemExit(main())
