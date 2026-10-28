"""Real per-repository training-job adapters; never fabricate scientific models."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

def row(id, command, *, method, dataset, seed=0, source="", resume_mode="RESTART_FROM_BEGINNING", blocker=None, gpu_capable=True, **more):
    return dict(id=str(id), command=[str(a) for a in command], method=str(method),
                dataset=str(dataset), seed=int(seed), source=str(source),
                resume_mode=resume_mode, blocker=blocker,
                gpu_capable=gpu_capable, **more)

def discover(root: Path, args):
    py = "{PYTHON}"
    name = root.name.lower()
    seeds = args.seed or [0]
    if any("=" not in item or not all(item.split("=", 1)) for item in args.data):\n        raise ValueError("--data entries must use DATASET=PATH")\n    data = dict(item.split("=", 1) for item in args.data)
    data = {key: str((root / value).resolve()) if not Path(value).is_absolute() else value
            for key, value in data.items()}
    results = []

    if name == "adp-cl":
        from adp_cl.experiments.job_inventory import discover_jobs
        inventory = discover_jobs(root, seed=0)
        for job in inventory["jobs"]:
            stream = job.case.stream
            for seed in seeds:
                command = [py, "scripts/run_jobs.py", "--execute", "--resume",
                           "--seeds", str(seed), "--job", job.job_id,
                           "--out", "{RESULT_DIR}"]
                if stream == "gas_drift" and data.get("gas_drift"):
                    command += ["--gas-root", data["gas_drift"]]
                results.append(row(job.job_id, command,
                    method=job.case.config.adp.method, dataset=stream, seed=seed,
                    source="src/adp_cl/experiments/job_inventory.py:discover_jobs",
                    resume_mode="ATOMIC_RESTART",
                    blocker=("gas_drift dataset requires --data gas_drift=PATH"
                             if stream == "gas_drift" and not data.get("gas_drift") else None)))
        return results

    if name == "rl-bgd":
        from rl_bgd.experiments.suites import SUITES
        for suite_name, suite in sorted(SUITES.items()):
            if any(label in suite_name.lower() for label in
                   ("ablation", "uncertainty_analysis", "mechanism_analysis", "compute_analysis")):
                continue
            for job in suite.jobs:
                for seed in job.seeds:
                    results.append(row(f"{suite_name}__{job.job_id}",
                        [py, "-m", "foundry_integration.rl_bgd_single",
                         "--suite", suite_name, "--job-id", job.job_id,
                         "--seed", str(seed), "--output-root", "{RESULT_DIR}"],
                        method=job.algorithm, dataset=job.environment, seed=seed,
                        source="src/rl_bgd/experiments/suites.py:SUITES",
                        resume_mode="RESTART_FROM_BEGINNING"))
        return results

    if name == "cl-benchmarks":
        # The native runner currently refuses non-smoke execution. The
        # broad method registry is NOT evidence of executable training jobs.
        results.append(row("implemented_smoke_default",
            [py, "-m", "cl_benchmarks.run", "--smoke"],
            method="native-default-smoke", dataset="native-smoke-stream",
            source="cl_benchmarks/run.py:main",
            resume_mode="RESTART_FROM_BEGINNING"))
        return results

    if name == "crl-benchmark":
        from crl_benchmark.run import load_config, validate_config, default_backend_name, build_backend_registry
        configs = [Path(p) for p in args.config] if args.config else sorted((root / "configs/suites").glob("*.yaml"))
        backends = build_backend_registry()
        for file in configs:
            if not file.is_absolute(): file = root / file
            if not file.is_file(): continue
            if any(w in file.name.lower() for w in ("smoke", "audit", "contract")) and not args.config:
                continue
            raw = load_config(file)
            reason = None
            try:
                validate_config(raw)
                backends.get(default_backend_name(raw)).validate_config(raw)
            except Exception as exc:
                reason = f"native validation blocked: {type(exc).__name__}: {exc}"
            for seed in (args.seed or [int(raw.get("seed", 0))]):
                job_id = file.stem
                command = [py, "-m", "crl_benchmark.run", "--config",
                           str(file), "--seed", str(seed),
                           "--run-id", job_id + "__s" + str(seed),
                           "--results-dir", "{RESULT_DIR}",
                           "--manifest-dir", "{RESULT_DIR}/manifests", "--resume"]
                results.append(row(job_id, command,
                    method=raw.get("method", "unknown"), dataset=raw.get("benchmark", file.stem),
                    seed=seed, source=str(file.relative_to(root)),
                    resume_mode="RESTART_FROM_BEGINNING", blocker=reason))
        return results

    if name == "crl-using-cl-methods":
        # A native planner JSON provides source-validated compatible runs;
        # parameter-free cartesian method x environment sweeps are forbidden.
        plans = [Path(p) for p in args.config]
        for plan in plans:
            if not plan.is_absolute(): plan = root / plan
            if not plan.exists(): raise FileNotFoundError(plan)
            payload = json.loads(plan.read_text())
            for run in payload.get("runs", []):
                compatible = run.get("compatible", False)
                native = run.get("manifest") or {}
                job_id = str(run["run_id"])
                seed = int(run.get("seed", native.get("seed", 0)))
                reasons = run.get("incompatibilities") or []
                results.append(row(job_id,
                    [py, "-m", "crl_cl.runners.execute", "--manifest", str(plan),
                     "--run-id", job_id, "--registry", "registry/methods.yaml",
                     "--output-root", "{RESULT_DIR}"],
                    method=run.get("method_id", native.get("method", "unknown")),
                    dataset=run.get("agent", "native_rl_env"), seed=seed,
                    source=str(plan),
                    resume_mode="RESTART_FROM_BEGINNING",
                    blocker=(None if compatible else "native incompatible: " + ",".join(reasons))))
        return results

    if name == "uncertainity-aware-crowd-navigation":
        specifications = [
            ("joint_scene_forecaster", "train_joint_scene_forecaster.py", "scenes_csv",
             "--csv", ["--dt", "0.25", "--history", "8", "--horizon", "12"]),
            ("linear_action_response", "train_action_linear_response.py", "action_jsonl",
             "--jsonl", []),
            ("bayesian_action_response", "train_bayesian_action_response.py", "action_jsonl",
             "--jsonl", []),
        ]
        for method, script, key, flag, options in specifications:
            found = data.get(key)
            blocker = None if found and Path(found).is_file() else (
                f"authorized dataset file required: --data {key}=PATH")
            for seed in (args.seed or [2026]):
                results.append(row(method,
                    [py, "scripts/" + script, flag, found or "{MISSING_DATA}",
                     "--seed", str(seed), "--output-dir", "{RESULT_DIR}", *options],
                    method=method, dataset=key, seed=seed,
                    source="scripts/" + script,
                    blocker=blocker, resume_mode="RESTART_FROM_BEGINNING",
                    gpu_capable=False))
        return results

    if name == "chunking":
        from chunking.experiments.manifest import build_manifest
        # Direct reproduction configurations are single native jobs. 
        files = ([Path(p) for p in args.config] if args.config else
                 sorted((root / "configs/reproduction/paper").glob("*.yaml")) +
                 [root / "configs/manifests/core_protocol_matrix.yaml"])
        for file in files:
            if not file.is_absolute(): file = root / file
            if not file.is_file(): continue
            if "/manifests/" in str(file):
                plan = build_manifest(file)
                for cell in plan["cells"]:
                    if cell.get("status") != "VALID": continue
                    config = cell["resolved_config"]
                    seed = int(config["runtime"]["seed"])
                    results.append(row(f"{file.stem}__{cell['run_id']}",
                        [py, "-m", "chunking.cli.main", "run",
                         "{CONFIG_FILE}", "--runs-dir", "{RESULT_DIR}"],
                        method=str(config.get("learner", {}).get("name", "native")),
                        dataset=str(config.get("data", {}).get("name", file.stem)),
                        seed=seed, source=str(file.relative_to(root)),
                        resume_mode="RESTART_FROM_BEGINNING",
                        config_payload=config))
            else:
                results.append(row(file.stem,
                    [py, "-m", "chunking.cli.main", "run", str(file),
                     "--runs-dir", "{RESULT_DIR}"],
                    method="native-paper-reproduction", dataset=file.stem,
                    source=str(file.relative_to(root)), resume_mode="RESTART_FROM_BEGINNING"))
        return results

    if name == "simple-adp-cl":
        # Reuse the repository's actual deduplicated, training-only planner.
        spec = importlib.util.spec_from_file_location("native_training_catalog",
                        root / "scripts/run_all_training_jobs.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        seeds_native = tuple(seeds)
        selected = {"explicit_adp", "bgd", "self_segmenting"}
        native_plan = module.build_plan(root, seeds_native, 1, selected, data)
        native_root = (root / args.out / "native_science").resolve()
        native_plan_file = native_root / "master_plan.json"
        # Inventory remains read-only; only explicit execution creates a
        # native immutable plan required by Bayesian BGD internal workers.
        if args.action in {"execute", "resume"}:
            if native_plan_file.is_file():
                saved = json.loads(native_plan_file.read_text())
                if [(j["job_id"], [r["run_id"] for r in j["runs"]]) for j in saved["jobs"]] != [
                    (j["job_id"], [r["run_id"] for r in j["runs"]]) for j in native_plan["jobs"]]:
                    raise ValueError("native scientific plan mismatch; choose a different --out")
            else:
                module.export_catalog(native_root, native_plan)
        for job in native_plan["jobs"]:
            for run in job["runs"]:
                native_command = module.command_for_job(
                    root, native_root, job, run, native_plan_file, data)
                family = job["scientific"]["family"]
                dataset = job["scientific"]["dataset"]
                deps = job.get("dependencies") or []
                results.append(row(run["run_id"], native_command,
                    method=family, dataset=dataset, seed=int(run["seed"]),
                    source="scripts/run_all_training_jobs.py:build_plan/command_for_job",
                    resume_mode="RESTART_FROM_BEGINNING",
                    blocker=("dependent native training job requires prior completion"
                             if deps else None)))
        return results
    raise RuntimeError(f"Unknown native research repository: {root.name}")
