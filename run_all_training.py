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

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)

def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    run_id = spec["run_id"]
    if os.environ.get("SIMPLE_ADP_JOB_ID") not in (None, run_id):
        raise RuntimeError("Foundry job identity mismatch")
    result_dir = Path(spec["result_dir"])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH"))))
    env["FOUNDRY_NATIVE_JOB_ID"] = run_id
    native = spec["scientific_job"]
    config_payload = native.get("config_payload")
    config_file = result_dir / "native_config.yaml"
    if config_payload is not None:
        import yaml
        config_file.parent.mkdir(parents=True, exist_ok=True)
        if config_file.exists():
            if yaml.safe_load(config_file.read_text(encoding="utf-8")) != config_payload:
                raise ValueError("resolved native config changed for existing run")
        else:
            config_file.write_text(yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8")
    command = [part.replace("{CONFIG_FILE}", str(config_file)) for part in spec["command"]]
    if not command or not all(isinstance(s, str) and s for s in command):
        raise ValueError("native command must be an argv array")
    result_dir.mkdir(parents=True, exist_ok=True)
    record = {"job_id": run_id, "scientific_job": spec["scientific_job"],
              "command": command, "status": "running"}
    write_json(result_dir / "native_process.json", record)
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        code = result.returncode
    except Exception as exc:
        code = 1
        record["exception"] = f"{type(exc).__name__}: {exc}"
    record.update(return_code=code, status="completed" if code == 0 else "failed")
    write_json(result_dir / "native_process.json", record)
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
    p.add_argument("--job", action="append", default=[])
    p.add_argument("--seed", action="append", type=int, default=[])
    p.add_argument("--config", action="append", default=[],
                   help="Native configuration or precomputed plan JSON")
    p.add_argument("--data", action="append", default=[],
                   help="DATASET=PATH; external authorized data")
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
    signature = hashlib.sha256(canonical({"native_jobs": selected, "version": 1}).encode()).hexdigest()
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
                   "result_dir": str(native_result), "command": command}
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
    manifest = {"run_signature": signature, "runner_settings": settings.as_dict(),
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
