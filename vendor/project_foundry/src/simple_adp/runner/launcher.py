"""Outer suite launcher: settings bridge, bundle handling, manifest, watchdog."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
from dataclasses import fields
from pathlib import Path
from typing import Any

from .bundles import export_results_bundle, rehydrate_sharded_telemetry, restore_results_bundle
from .config import RunnerSettings, UNBOUNDED_CONCURRENCY
from .plan import SuitePlan
from .scheduler import SuiteCoordinator
from .processes import terminate_process_tree, terminate_process_tree_after_grace
from .runtime_tuning import bootstrap_runtime, launcher_child_env
from .logging_context import configure_console_logging

LOG = logging.getLogger("simple_adp.launcher")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "y"}:
        return True
    if normalized in {"0", "false", "no", "off", "n", ""}:
        return False
    return default


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the generic pressure-aware simple-adp suite")
    parser.add_argument("--settings-json", help="JSON object with any RunnerSettings field")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override any setting using JSON scalar/list syntax")
    parser.add_argument(
        "--dataset-suites", default="cc18,tabzilla,ctr23",
        help="comma-separated suite names (bare names use OpenML) or provider:suite selectors",
    )
    parser.add_argument("--output")
    parser.add_argument("--run-root")
    parser.add_argument("--source-run-root")
    parser.add_argument("--profiles")
    parser.add_argument("--seeds")
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--gpu-device-index", type=int)
    parser.add_argument("--cpu-only", action="store_true", default=None)
    parser.add_argument("--smoke", action="store_true", help="one epoch and one full-training-partition batch")
    parser.add_argument("--dry-run", action="store_true", help="write plan and run metadata without scheduling children")
    parser.add_argument("--detach", action="store_true", help="detach outer launcher and return its PID")
    parser.add_argument("--list-suites", action="store_true")
    parser.add_argument("--status", action="store_true", help="print the current run status and exit")
    parser.add_argument("--resume", action="store_true", help="resume the exact persisted suite plan and run state")
    parser.add_argument("--report", action="store_true", help="regenerate reports from persisted state without scheduling children")
    return parser


def _settings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> RunnerSettings:
    values: dict[str, Any] = {}
    known = {field.name for field in fields(RunnerSettings)}
    for name in known:
        env_name = f"SIMPLE_ADP_{name.upper()}"
        if env_name in os.environ:
            values[name] = _parse_value(os.environ[env_name])
    if args.settings_json:
        try:
            settings_file = json.loads(Path(args.settings_json).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read settings JSON: {exc}")
        if not isinstance(settings_file, dict):
            parser.error("settings JSON must be an object")
        unknown = set(settings_file) - known
        if unknown:
            parser.error(f"unknown RunnerSettings field(s): {', '.join(sorted(unknown))}")
        values.update(settings_file)
    for pair in args.overrides:
        if "=" not in pair:
            parser.error("--set requires KEY=VALUE")
        key, raw = pair.split("=", 1)
        if key not in known:
            parser.error(f"unknown RunnerSettings field {key!r}")
        values[key] = _parse_value(raw)
    direct = {
        "results_dir": args.output,
        "run_root": args.run_root,
        "source_run_root": args.source_run_root,
        "generic_profiles": tuple(args.profiles.split(",")) if args.profiles else None,
        "model_seeds": tuple(int(part) for part in args.seeds.split(",")) if args.seeds else None,
        "task_limit": args.task_limit,
        "fold_limit": args.fold_limit,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "gpu_device_index": args.gpu_device_index,
        "cpu_only": args.cpu_only,
    }
    values.update({key: value for key, value in direct.items() if value is not None})
    if args.smoke:
        values["smoke_test"] = True
    if args.dry_run:
        values["dry_run"] = True
    values.setdefault("tasks", ("all",))
    return RunnerSettings(**values)


def _apply_runtime_environment_defaults(settings: RunnerSettings) -> None:
    """Apply configured runtime defaults when their inherited values are empty.

    This mirrors the shell wrapper's ``${VAR:-default}`` behavior while
    keeping explicitly inherited values authoritative. Applying defaults
    before runtime bootstrap also ensures imports and subprocesses see them.
    """
    defaults = {
        "CUDA_VISIBLE_DEVICES": (settings.cuda_visible_devices, "0"),
        "PYTORCH_CUDA_ALLOC_CONF": (settings.pytorch_cuda_alloc_conf, "expandable_segments:True,max_split_size_mb:128"),
        "CUBLAS_WORKSPACE_CONFIG": (settings.cublas_workspace_config, ":4096:8"),
    }
    for key, (configured, fallback) in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = str(configured or fallback)


def _dataset_suites(value: str) -> tuple[tuple[str, str], ...]:
    """Resolve CLI suite selectors; bare names remain OpenML-compatible.

    Use ``provider:suite`` for registered non-OpenML providers. A bare suite
    keeps the original CLI behavior and targets the built-in OpenML provider.
    """
    selections: list[tuple[str, str]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            provider, suite = (part.strip() for part in token.split(":", 1))
        else:
            provider, suite = "openml", token
        if not provider or not suite:
            raise ValueError(f"invalid dataset suite selector {token!r}; expected suite or provider:suite")
        selections.append((provider, suite))
    if not selections:
        raise ValueError("at least one dataset suite is required")
    return tuple(selections)


def _registered_dataset_providers(registry: Any) -> Any:
    """Return the provider mapping from the dataset-specific registry API."""
    providers = getattr(registry, "providers", None)
    if providers is not None:
        return providers
    # Keep compatibility with small third-party registry facades that expose
    # the same mapping under ``adapters``.
    return getattr(registry, "adapters", {})


def _detach(args: argparse.Namespace, settings: RunnerSettings) -> int:
    results_dir = Path(settings.results_dir).expanduser().resolve()
    run_root = results_dir / "run_root"  # shell detach contract ignores configured RUN_ROOT
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "launcher.log"
    pid_path = run_root / "launcher.pid"
    command = [sys.executable, "-u", "-m", "simple_adp.runner.launcher"]
    command.extend(token for token in sys.argv[1:] if token != "--detach")
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    try:
        with pid_path.open("w", encoding="utf-8") as stream:
            stream.write(f"{process.pid}\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        # Match the source branch: a PID-file error does not terminate Popen.
        print(f"detached launcher started as {process.pid}, but PID file failed: {exc}", file=sys.stderr, flush=True)
    print(f"detached launcher PID={process.pid} log={log_path}", flush=True)
    return 0


def _run_with_watchdog(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    child = subprocess.Popen(command, cwd=cwd, env=env)
    watchdog_command = [
        sys.executable, "-u", "-m", "simple_adp.runner.watchdog",
        "--master", str(os.getpid()), "--child", str(child.pid), "--log", str(log_path),
    ]
    watchdog = None
    return_code = None
    signal_cleanup = False
    previous_handlers: dict[int, Any] = {}

    def stop_for_signal(signum: int, _frame: Any) -> None:
        nonlocal signal_cleanup
        if signal_cleanup:
            return
        signal_cleanup = True
        print(
            f"[RUNNER] {signal.Signals(signum).name} received. "
            f"Stopping controller (PID {child.pid})...",
            file=sys.stderr,
            flush=True,
        )
        try:
            if child.poll() is None:
                terminate_process_tree_after_grace(child.pid, grace_seconds=2.0)
        finally:
            if watchdog is not None:
                try:
                    watchdog.kill()
                except OSError:
                    pass
        raise SystemExit(128 + signum)

    try:
        # The Linux shell reference traps these signals and performs immediate
        # TERM, a two-second grace period, KILL, then watcher cleanup.
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
                previous_handlers[signum] = signal.signal(signum, stop_for_signal)
        watchdog = subprocess.Popen(
            watchdog_command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return_code = int(child.wait())
    except BaseException:
        # The shell wrapper's EXIT/signal trap kills the owned controller
        # tree. Preserve that cleanup if Python is interrupted or watchdog
        # startup fails before it can supervise the child.
        if not signal_cleanup:
            try:
                if child.poll() is None:
                    terminate_process_tree(child)
            except Exception:
                LOG.exception("failed to terminate controller after launcher interruption")
        raise
    finally:
        if watchdog is not None and not signal_cleanup:
            try:
                watchdog.terminate()
                watchdog.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    watchdog.kill()
                except OSError:
                    pass
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if return_code == 0:
        try:
            log_path.unlink()
        except OSError:
            pass
    assert return_code is not None
    return return_code


def load_registries():
    """Import model/provider plugins lazily after runtime defaults are set."""
    from .plugins import load_registries as _load_registries

    return _load_registries()


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    settings = _settings(args, parser)
    _apply_runtime_environment_defaults(settings)
    if args.detach:
        return _detach(args, settings)
    if args.status:
        status_path = Path(settings.run_root or (Path(settings.results_dir) / "run_root")) / "status.json"
        try:
            print(status_path.read_text(encoding="utf-8"), end="")
            return 0
        except OSError as exc:
            parser.error(f"cannot read status: {exc}")
    if args.resume and args.report:
        parser.error("--resume and --report cannot be used together")
    configure_console_logging()
    bootstrap_runtime("simple-adp-suite-launcher")
    datasets, models = load_registries()
    try:
        provider_suites = _dataset_suites(args.dataset_suites)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list_suites:
        providers = _registered_dataset_providers(datasets)
        available = {
            f"{provider}:{suite}" if provider != "openml" else suite:
                [ref.task_id for ref in providers[provider].list_tasks(suite)]
            for provider, suite in provider_suites
            if provider in providers
        }
        missing = sorted({provider for provider, _suite in provider_suites} - set(providers))
        if missing:
            parser.error(f"dataset provider(s) are not registered: {', '.join(missing)}")
        print(json.dumps(available, indent=2))
        return 0

    results_dir = Path(settings.results_dir).expanduser().resolve()
    run_root = Path(settings.run_root).expanduser().resolve() if settings.run_root else results_dir / "run_root"
    source_root = Path(settings.source_run_root).expanduser().resolve() if settings.source_run_root else None
    plan_path = run_root / "suite_plan.json"
    if args.report:
        if not plan_path.is_file():
            parser.error(f"cannot generate report without persisted plan: {plan_path}")
        try:
            plan = SuitePlan.from_manifest(plan_path)
        except Exception as exc:
            parser.error(f"cannot load persisted suite plan: {exc}")
        if plan.manifest.get("results_dir") != str(results_dir) or plan.manifest.get("run_root") != str(run_root):
            parser.error("report paths differ from the persisted suite plan")
        coordinator = SuiteCoordinator(
            plan,
            settings,
            run_root=run_root,
            results_dir=results_dir,
            repository_root=REPOSITORY_ROOT,
        )
        summary = coordinator.write_persisted_state_report()
        print(json.dumps(summary, sort_keys=True, default=str), flush=True)
        return 0
    # Source ordering intentionally creates the nested run root before restore.
    run_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if settings.auto_resume_bundle:
        restore_results_bundle(results_dir, REPOSITORY_ROOT)
    rehydrated_parts = rehydrate_sharded_telemetry(results_dir, REPOSITORY_ROOT)
    if rehydrated_parts:
        print(f"rehydrated {rehydrated_parts} sharded telemetry file(s) before launch", flush=True)
    launch_manifest = {
        "repository_root": str(REPOSITORY_ROOT),
        "results_dir": str(results_dir),
        "run_root": str(run_root),
        "source_run_root": str(source_root) if source_root else None,
        "dataset_suites": tuple(part.strip() for part in args.dataset_suites.split(",") if part.strip()),
        "providers_and_suites": provider_suites,
        "runner_settings": settings.as_dict(),
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    (run_root / "suite_launch_manifest.json").write_text(json.dumps(launch_manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    if args.resume:
        if not plan_path.is_file():
            parser.error(f"cannot resume without persisted suite plan: {plan_path}")
        try:
            plan = SuitePlan.from_manifest(plan_path)
        except Exception as exc:
            parser.error(f"cannot load persisted suite plan: {exc}")
        saved_settings = json.dumps(plan.manifest.get("runner_settings"), sort_keys=True, default=str)
        requested_settings = json.dumps(settings.as_dict(), sort_keys=True, default=str)
        if saved_settings != requested_settings:
            parser.error("resume settings differ from the persisted suite plan")
        saved_suites = plan.manifest.get("providers_and_suites", [])
        requested_suites = [list(selection) for selection in provider_suites]
        if saved_suites != requested_suites:
            parser.error("resume dataset suites differ from the persisted suite plan")
        if plan.manifest.get("results_dir") != str(results_dir) or plan.manifest.get("run_root") != str(run_root):
            parser.error("resume paths differ from the persisted suite plan")
        current_model_adapters = set(models.adapters)
        current_dataset_providers = set(_registered_dataset_providers(datasets))
        missing_models = set(plan.manifest.get("model_adapters", [])) - current_model_adapters
        missing_providers = {provider for provider, _suite in saved_suites} - current_dataset_providers
        if missing_models or missing_providers:
            parser.error(
                "resume plan references unregistered plugins: "
                f"model_adapters={sorted(missing_models)}, dataset_providers={sorted(missing_providers)}"
            )
    else:
        plan = SuitePlan.build(
            settings=settings,
            results_dir=results_dir,
            run_root=run_root,
            dataset_providers=tuple(_registered_dataset_providers(datasets).values()),
            dataset_suites=provider_suites,
            model_adapters=tuple(models.adapters.values()),
        )
        plan_path.write_text(json.dumps(plan.manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    controller_command = [
        sys.executable, "-u", "-m", "simple_adp.runner.controller",
        "--plan", str(plan_path),
        "--run-root", str(run_root),
        "--results-dir", str(results_dir),
        "--repository-root", str(REPOSITORY_ROOT),
    ]
    if source_root is not None:
        controller_command.extend(["--source-run-root", str(source_root)])
    budget = settings.effective_concurrency
    child_env = launcher_child_env(
        os.environ.copy(),
        job_key="simple-adp-central-controller",
        launch_budget=budget,
        shared_cpu=False,
        cpu_only=settings.cpu_only,
        device_index=settings.gpu_device_index,
    )
    child_env["TABULAR_CHILD_SHARED_CPU"] = "0"
    child_env["TABULAR_CPU_WORKERS"] = "0"
    child_env["TABULAR_CPU_JOB_CONCURRENCY"] = "1" if budget >= UNBOUNDED_CONCURRENCY else str(max(1, budget))
    child_env["SIMPLE_ADP_RUN_ROOT"] = str(run_root)
    child_env["SIMPLE_ADP_RESULTS_DIR"] = str(results_dir)
    # The source wrapper pairs automatic restore/export behind the same switch.
    # Export failures never mask the controller's return code.
    try:
        return_code = _run_with_watchdog(
            controller_command,
            cwd=REPOSITORY_ROOT,
            env=child_env,
            log_path=Path("/tmp") / f"simple-adp-watchdog-{os.getpid()}.log",
        )
    finally:
        if settings.auto_resume_bundle:
            try:
                export_results_bundle(results_dir, REPOSITORY_ROOT)
            except Exception as exc:
                print(f"automatic result export failed: {exc}", file=sys.stderr, flush=True)
    return return_code


def cli_main(argv: list[str] | None = None) -> int:
    """Match the source launcher’s final KeyboardInterrupt notice and exit."""
    try:
        return main(argv)
    except KeyboardInterrupt:
        print(
            "\n[INTERRUPT] Caught KeyboardInterrupt. Launcher-owned jobs were cleaned up and marked resumable.",
            flush=True,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(cli_main())
