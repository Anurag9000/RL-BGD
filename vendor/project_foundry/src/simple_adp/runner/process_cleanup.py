"""Ownership matching and forceful cleanup for emergency runner recovery."""

from __future__ import annotations

from pathlib import Path

import psutil

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODULE_ENTRYPOINT_MARKERS = {
    "simple_adp.runner.launcher": {
        "--output", "--run-root", "--source-run-root", "--settings-json",
        "--set", "--list-suites", "--status", "--resume", "--report",
        "--dry-run",
    },
    "simple_adp.runner.controller": {"--plan"},
    "simple_adp.runner.worker": {"--spec"},
}
SCRIPT_ENTRYPOINTS = {
    "simple-adp-suite.sh",
    "simple-adp-suite.ps1",
}
LAUNCHER_MARKER_FLAGS = MODULE_ENTRYPOINT_MARKERS["simple_adp.runner.launcher"]


def _entrypoint_markers(command: list[str]) -> tuple[int, set[str]] | None:
    """Return the argument offset and markers valid for this runner entrypoint."""
    if not command:
        return None
    executable = Path(command[0].strip('"')).name.lower()
    if executable.startswith("python"):
        for index, token in enumerate(command[1:-1], start=1):
            if token == "-c":
                return None
            if token == "-m":
                module = command[index + 1]
                markers = MODULE_ENTRYPOINT_MARKERS.get(module)
                return (index + 2, markers) if markers is not None else None
    if Path(command[0].strip('"')).name in SCRIPT_ENTRYPOINTS:
        return 1, LAUNCHER_MARKER_FLAGS
    for index, token in enumerate(command[1:], start=1):
        if Path(token.strip('"')).name not in SCRIPT_ENTRYPOINTS:
            continue
        previous = command[index - 1].strip('"').lower()
        if index == 1 and executable in {"bash", "sh", "dash", "zsh"}:
            return index + 1, LAUNCHER_MARKER_FLAGS
        if previous in {"-file", "-f"} and executable in {"pwsh", "powershell", "powershell.exe", "pwsh.exe"}:
            return index + 1, LAUNCHER_MARKER_FLAGS
    return None


def _argv_contains_root(command: list[str], root: Path) -> bool:
    """Recognize root paths in argv, including ``--option=/path`` values."""
    for token in command:
        candidate = token.split("=", 1)[1] if "=" in token else token
        try:
            path = Path(candidate.strip('"')).expanduser()
            if path.is_absolute() and path.resolve().is_relative_to(root):
                return True
        except (OSError, RuntimeError, ValueError):
            continue
    return False


def owns_process(command: list[str], cwd: str | None, root: Path = REPOSITORY_ROOT) -> bool:
    """Match an owned runner entrypoint, marker, and repository location."""
    if not command:
        return False
    entrypoint = _entrypoint_markers(command)
    if entrypoint is None:
        return False
    argument_start, marker_flags = entrypoint
    arguments = command[argument_start:]
    if not any(
        token in marker_flags or any(token.startswith(f"{flag}=") for flag in marker_flags)
        for token in arguments
    ):
        return False
    root = root.resolve()
    try:
        cwd_inside = cwd is not None and Path(cwd).resolve().is_relative_to(root)
    except (OSError, RuntimeError):
        cwd_inside = False
    return cwd_inside or _argv_contains_root(command, root)


def kill_owned_processes(root: Path = REPOSITORY_ROOT) -> int:
    current_pid = psutil.Process().pid
    owned: list[psutil.Process] = []
    for process in psutil.process_iter(("pid", "cmdline", "cwd")):
        try:
            if process.info["pid"] != current_pid and owns_process(
                process.info.get("cmdline") or [], process.info.get("cwd"), root
            ):
                owned.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    seen: set[int] = set()
    killed = 0
    for process in owned:
        try:
            descendants = process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            descendants = []
        for descendant in reversed(descendants):
            if descendant.pid in seen:
                continue
            seen.add(descendant.pid)
            try:
                descendant.kill()
                killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if process.pid in seen:
            continue
        seen.add(process.pid)
        try:
            process.kill()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return killed
