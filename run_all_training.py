"""Native training jobs controlled by the original, unmodified Project Foundry."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)

_SCIENTIFIC_SOURCE_PATHS = (
    "src", "scripts", "foundry_integration", "run_all_training.py",
    "configs", "vendor/project_foundry", "pyproject.toml",
    "requirements-foundry.txt",
)


def _source_revision():
    """Require an auditable committed source tree for scientific execution."""
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if revision.returncode != 0 or len(revision.stdout.strip()) != 40:
        raise RuntimeError("Foundry scientific execution requires a Git HEAD revision")
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *_SCIENTIFIC_SOURCE_PATHS],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if dirty.returncode == 1:
        raise RuntimeError("uncommitted scientific source changes: commit before execution")
    if dirty.returncode != 0:
        raise RuntimeError("cannot verify scientific source worktree cleanliness")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *_SCIENTIFIC_SOURCE_PATHS],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if untracked.returncode != 0 or untracked.stdout.strip():
        raise RuntimeError("untracked scientific source files or Git index unavailable")
    return revision.stdout.strip()


def _scientific_run_signature(jobs, source_revision):
    if not isinstance(source_revision, str) or len(source_revision) != 40:
        raise ValueError("source revision must be a 40-character Git SHA")
    if any(char not in "0123456789abcdef" for char in source_revision.lower()):
        raise ValueError("source revision must be a hexadecimal Git SHA")
    return hashlib.sha256(canonical({
        "native_jobs": jobs,
        "source_revision": source_revision.lower(),
        "version": 2,
    }).encode()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)

def _verify_scientific_child(command, job):
    """Validate the native artifact contract when this is an RL-BGD one-job child."""
    if len(command) < 4 or command[1:3] != [
        "-m", "foundry_integration.rl_bgd_single"
    ]:
        return None
    from foundry_integration.rl_bgd_single import verify_native_job

    def required(flag):
        if command.count(flag) != 1:
            raise ValueError(f"native command must contain exactly one {flag}")
        index = command.index(flag)
        if index + 1 >= len(command):
            raise ValueError(f"missing native command value for {flag}")
        return command[index + 1]

    suite = required("--suite")
    native_job_id = required("--job-id")
    seed = int(required("--seed"))
    output_root = required("--output-root")
    if job.get("id") != f"{suite}__{native_job_id}" or job.get("seed") != seed:
        raise ValueError("Foundry selection does not match native scientific job identity")
    return verify_native_job(suite, native_job_id, seed, output_root)


def _archive_old_worker_status(result_dir):
    """Move previous attempt indicators out of the authoritative current paths."""
    previous = [result_dir / "result.json", result_dir / "native_process.json"]
    if not any(path.exists() for path in previous):
        return None
    archive = result_dir / "attempt_history"
    archive.mkdir(parents=True, exist_ok=True)
    index = 1
    while any((archive / f"attempt_{index:06d}_{path.name}").exists()
              for path in previous):
        index += 1
    for path in previous:
        if path.exists():
            os.replace(path, archive / f"attempt_{index:06d}_{path.name}")
    return index


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    run_id = spec["run_id"]
    result_dir = Path(spec["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    # An invalid config, command, revision or child launch is still a failed
    # *attempt*. Clear stale success before any preflight step can reject it.
    previous_attempt = _archive_old_worker_status(result_dir)
    raw_command = spec.get("command")
    record = {"job_id": run_id, "scientific_job": spec.get("scientific_job"),
              "command": raw_command, "status": "running",
              "archived_previous_attempt": previous_attempt}
    status_path = result_dir / "native_process.json"
    write_json(status_path, record)
    try:
        if os.environ.get("SIMPLE_ADP_JOB_ID") not in (None, run_id):
            raise RuntimeError("Foundry job identity mismatch")
        native = spec["scientific_job"]
        if not isinstance(native, dict):
            raise TypeError("scientific job must be an object")
        if (not isinstance(raw_command, list) or not raw_command
                or not all(isinstance(part, str) and part for part in raw_command)):
            raise ValueError("native command must be a nonempty argv array")

        config_payload = native.get("config_payload")
        config_file = result_dir / "native_config.yaml"
        if config_payload is not None:
            import yaml
            if config_file.exists():
                if yaml.safe_load(config_file.read_text(encoding="utf-8")) != config_payload:
                    raise ValueError("resolved native config changed for existing run")
            else:
                config_file.write_text(
                    yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8"
                )
        command = [
            part.replace("{CONFIG_FILE}", str(config_file)) for part in raw_command
        ]
        record["command"] = command
        write_json(status_path, record)

        native_module = "foundry_integration.rl_bgd_single"
        is_native_command = command[1:3] == ["-m", native_module]
        is_registered_native = (
            native.get("source") == "src/rl_bgd/experiments/suites.py:SUITES"
            or "scientific_contract" in native
            or "scientific_fingerprint" in native
        )
        if (is_registered_native or native_module in command) and not is_native_command:
            raise ValueError("registered native scientific job requires canonical wrapper argv")

        recorded_revision = spec.get("source_revision")
        if recorded_revision is not None:
            if not isinstance(recorded_revision, str):
                raise TypeError("scientific source revision must be a string")
            if _source_revision() != recorded_revision:
                raise ValueError("scientific source revision changed since native plan creation")
        elif is_native_command:
            raise ValueError("native scientific child lacks mandatory source revision")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH")
        )))
        env["FOUNDRY_NATIVE_JOB_ID"] = run_id
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        code = result.returncode
        if code == 0:
            # A job can run for hours: fail closed if source moved while the
            # child ran, even if its native canonical artifacts are valid.
            if recorded_revision is not None and _source_revision() != recorded_revision:
                raise ValueError("scientific source revision changed during native execution")
            verified = _verify_scientific_child(command, native)
            if is_registered_native and verified is None:
                raise ValueError("registered native job lacks strict canonical verification")
            if verified is not None:
                record["verified_native_run"] = verified
    except Exception as exc:
        code = 1
        record["exception"] = f"{type(exc).__name__}: {exc}"
    record.update(return_code=code, status="completed" if code == 0 else "failed")
    write_json(status_path, record)
    if code == 0:
        write_json(result_dir / "result.json", record)
    return code

def report(out):
    counts = {}
    for path in (out / "controller" / "jobs").glob("**/job_state.json"):
        try:
            record = json.loads(path.read_text())
            status = record.get("status", "unknown")
        except (OSError, ValueError):
            status = "invalid-state"
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"state_counts": counts}, indent=2, sort_keys=True))
    return 0

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", default="inventory",
                   choices=("inventory", "execute", "resume", "status", "report"))
    p.add_argument("--execute", action="store_true",
                   help="Mandatory opt-in to run training jobs")
    p.add_argument("--accept-restart-risk", action="store_true",
                   help="Permit routes without verified exact checkpoint recovery")
    p.add_argument("--out", default="artifacts/foundry_training")
    p.add_argument("--write-inventory", action="store_true", help="Persist the source-derived catalog as JSON without executing jobs")
    p.add_argument("--job", action="append", default=[])
    p.add_argument("--seed", action="append", type=int, default=[])
    p.add_argument("--config", action="append", default=[],
                   help="Native config or plan for supporting adapters (not RL-BGD SUITES)")
    p.add_argument("--data", action="append", default=[],
                   help="DATASET=PATH for adapters with external-data support (not RL-BGD SUITES)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cpu-only", action="store_true")
    args = p.parse_args(argv)
    out = (ROOT / args.out).resolve()
    if args.action in ("status", "report"):
        return report(out)
    if args.limit < 0:
        p.error("--limit must be nonnegative")
    sys.path.insert(0, str(ROOT / "src"))
    from foundry_integration.catalog import discover
    jobs = discover(ROOT, args)
    if len({(j["id"], int(j.get("seed", 0))) for j in jobs}) != len(jobs):
        raise RuntimeError("native catalog has duplicate scientific IDs/seeds")
    selected = [j for j in jobs
                if (not args.job or j["id"] in args.job)
                and (not args.seed or j.get("seed", 0) in args.seed)]
    blocked = [j for j in selected if j.get("blocker")]
    if args.action != "inventory" and args.execute and not args.job:
        # Preserve every blocked route in inventory, schedule only executable ones.
        selected = [j for j in selected if not j.get("blocker")]
    if args.limit:
        selected = selected[:args.limit]
    print(json.dumps({
        "catalog_total": len(jobs), "selected": len(selected),
        "blocked_in_full_selection": len(blocked),
        "source": "vendored Project Foundry SuiteCoordinator",
        "jobs": [{"id": j["id"], "seed": j.get("seed", 0),
                  "method": j.get("method"), "dataset": j.get("dataset"),
                  "resume_mode": j.get("resume_mode", "NOT_RESUMABLE"),
                  "blocker": j.get("blocker")} for j in selected],
    }, indent=2, sort_keys=True))
    if args.write_inventory:
        from collections import Counter
        fingerprint_ids = {
            j.get("scientific_fingerprint", j["id"]) for j in jobs
        }
        catalog = {
            "format": "foundry-native-training-catalog/v2",
            "repository": ROOT.name,
            "distinct_scientific_jobs": len(fingerprint_ids),
            "seed_expanded_training_runs": len(jobs),
            "expanded_training_runs": len(jobs),
            "by_method": dict(sorted(Counter(j.get("method") for j in jobs).items())),
            "by_dataset": dict(sorted(Counter(j.get("dataset") for j in jobs).items())),
            "blocked_runs": sum(bool(j.get("blocker")) for j in jobs),
            "jobs": jobs,
        }
        if ROOT.name.lower() == "rl-bgd":
            from foundry_integration.entrypoint_census import entrypoint_census
            entrypoints = entrypoint_census(ROOT)
            catalog["entrypoint_coverage"] = {
                key: value for key, value in entrypoints.items()
                if key not in {
                    "scripts", "runner_modules", "unmatched_import_candidates",
                    "independent_runner_candidates", "suite_only_targets",
                    "suite_targets_absent_from_public_runner_definitions",
                }
            }
            write_json(out / "native_entrypoint_coverage.json", entrypoints)
            from foundry_integration.scientific_inventory import (
                coverage_summary, inventory_records,
            )
            science = inventory_records()
            coverage = coverage_summary(science)
            catalog["scientific_coverage"] = coverage
            write_json(out / "scientific_job_coverage.json", coverage)
            import csv
            csv_path = out / "scientific_job_inventory.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            temp = csv_path.with_name("." + csv_path.name + ".tmp")
            fields = [
                "suite", "job_id", "native_run_id", "seed",
                "scientific_fingerprint", "execution_fingerprint",
                "classification", "method", "dataset",
                "optional_extra", "blocker",
            ]
            with temp.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows({key: record.get(key) for key in fields}
                                 for record in science)
            os.replace(temp, csv_path)
        write_json(out / "scientific_job_inventory.json", catalog)
    if args.action == "inventory":
        return 0
    if not args.execute:
        p.error("execution requires explicit --execute")
    if any(j.get("blocker") for j in selected):
        p.error("blocked jobs selected; choose supported jobs explicitly with --job")
    if any(j.get("resume_mode") != "EXACT_RESUME" for j in selected) and not args.accept_restart_risk:
        p.error("inexact resume possible; explicitly opt in with --accept-restart-risk")
    if not selected:
        p.error("no executable jobs selected")
    vendor = ROOT / "vendor" / "project_foundry" / "src"
    if not (vendor / "simple_adp" / "runner" / "scheduler.py").is_file():
        raise RuntimeError("Pinned Project Foundry scheduler missing")
    sys.path.insert(0, str(vendor))
    from simple_adp.runner.config import RunnerSettings
    from simple_adp.runner.datasets import DatasetTaskRef
    from simple_adp.runner.route import ModelRoute
    from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan, manifest_compatibility_digest
    from simple_adp.runner.scheduler import SuiteCoordinator
    settings = RunnerSettings(cpu_only=args.cpu_only, run_hardness_screen=False)
    source_revision = _source_revision()
    signature = _scientific_run_signature(jobs, source_revision)
    specs = []
    for j in selected:
        seed = int(j.get("seed", 0))
        identity = RunIdentity(
            provider="native", suite=ROOT.name, task_id=j["id"],
            split_repeat=0, fold=0, sample=0, suite_repeat=1,
            model_route=j.get("method", "native"), seed=seed,
            split_sha256="native-no-invented-split", run_signature=signature)
        native_result = (out / "results" / identity.run_id).resolve()
        command = [s.replace("{PYTHON}", sys.executable)
                   .replace("{RESULT_DIR}", str(native_result / "native"))
                   .replace("{ROOT}", str(ROOT)) for s in j["command"]]
        spec_path = out / "controller" / "jobs" / identity.run_id / "native_spec.json"
        payload = {"run_id": identity.run_id, "scientific_job": j,
                   "result_dir": str(native_result), "command": command,
                   "source_revision": source_revision}
        if spec_path.is_file():
            if json.loads(spec_path.read_text()) != payload:
                raise ValueError("existing immutable native spec differs: " + str(spec_path))
        else:
            write_json(spec_path, payload)
        specs.append(RunSpec(
            identity=identity,
            task_ref=DatasetTaskRef(provider="native", suite=ROOT.name, task_id=j["id"],
                                    metadata={"source": j.get("source"), "dataset": j.get("dataset")}),
            model_adapter="native-process", route=ModelRoute(model_id="native",
                label=j.get("method", "native"), suite_group="vanilla"),
            split_file="", result_dir=str(native_result), worker_settings={},
            command=(sys.executable, str(ROOT / "run_all_training.py"),
                     "__worker", str(spec_path.resolve())),
            device_capable=bool(j.get("gpu_capable", True))))
    manifest = {"run_signature": signature, "source_revision": source_revision,
                "runner_settings": settings.as_dict(),
                "job_count": len(specs), "jobs": [s.as_dict() for s in specs],
                "native_jobs": selected, "results_dir": str(out / "results"),
                "run_root": str(out / "controller")}
    manifest["manifest_digest"] = manifest_compatibility_digest(manifest)
    plan = SuitePlan(signature, manifest, tuple(specs))
    summary = SuiteCoordinator(plan, settings, run_root=out / "controller",
                               results_dir=out / "results", repository_root=ROOT).run()
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if int(summary.get("terminal_failures", 0)) == 0 else 1

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "__worker":
        raise SystemExit(worker(sys.argv[2]))
    raise SystemExit(main())
