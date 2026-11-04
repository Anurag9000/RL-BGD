# Foundry native-integration implementation ledger — RL-BGD

## Evidence-backed status

| Scope | Status | Evidence |
|---|---|---|
| Whole pinned Foundry tracked tree | COPIED; Git object identity previously checked | `vendor/project_foundry/`, `SOURCE_FILE_MANIFEST.json`, `verify_source.py` |
| Original Foundry control-plane code unchanged | PRESERVED | All 78 original files remain in the vendor directory |
| Central entrypoint | IMPLEMENTED, RUNTIME UNVERIFIED | `run_all_training.py` |
| Actual Foundry coordinator invoked | STATICALLY WIRED, RUNTIME UNVERIFIED | `SuiteCoordinator(...).run()` in `run_all_training.py` |
| Native scientific inventory adapter | IMPLEMENTED, EXHAUSTIVENESS UNVERIFIED | `foundry_integration/catalog.py` |
| Real native scientific subprocess bridge | IMPLEMENTED, END-TO-END UNVERIFIED | `run_all_training.py:worker` |
| Native success/failure propagation | IMPLEMENTED, bounded regression tests added; live integration pending | `foundry_integration/test_bridge.py` |
| Native algorithms and original CL/RL model math | UNMODIFIED | Original source files not replaced by Foundry scientific models |
| Resource admission/concurrency/pressure controls | VENDORED AND WIRED; DYNAMIC TEST UNVERIFIED | Original `simple_adp.runner.scheduler`, `admission`, `resources`, `processes` |
| Hardness-screening | NOT APPLICABLE TO GENERIC NATIVE JOBS | Native adapter explicitly disables ADP-only hardness screening, without changing source defaults |
| Checkpoint and exact resume | NOT CERTIFIED | Non-exact routes require `--accept-restart-risk` |
| Complete full non-ablation training-job census | NOT CERTIFIED | Native catalog discovery is an implementation, not evidence that no other routes exist |
| Scientific accuracy/regression and GPU pressure behavior | NOT VERIFIED | Real datasets, dependencies and hardware-dependent tests still required |

## Repository-specific integration

Native `SUITES` exposes implemented paper jobs. A single-job wrapper calls `execute_suite` with exactly one selected job/seed. It now additionally checks that the selected native run ID, Git revision, fully resolved configuration, run metadata, and strict canonical artifacts match the requested job using the native artifact loader. The outer Foundry worker independently verifies the child's canonical run identity before publishing `result.json`. Exit code zero alone is insufficient. When a job is retried, old `result.json` and `native_process.json` indicators are moved into `attempt_history/` before launch, so a failed attempt cannot inherit a stale success marker. This preserves older process-status evidence without treating an old attempt as the new run. All declared suite families, including ablations, uncertainty analysis, mechanistic experiments and compute-training jobs, are represented in the registry-derived Foundry catalog; this does not imply scientific completion or exhaustiveness beyond the registered suites.

## Commands and safety

Install native dependencies for this repository and the original Foundry runtime dependency floors:

```bash
python -m pip install -r requirements-foundry.txt
python docs/foundry_integration/verify_source.py
python -m py_compile run_all_training.py foundry_integration/catalog.py foundry_integration/rl_bgd_single.py
python foundry_integration/test_bridge.py
python run_all_training.py
```

The default command **only inventories**. Select and launch genuine native training explicitly:

```bash
python run_all_training.py execute --execute --accept-restart-risk --job stationary_core__stationary_sac_bgd --seed 0
```

The restart-risk flag is **not** evidence of safe checkpoint recovery. Foundry may terminate a child under memory pressure and relaunch it. Use only when repeated execution is safe for the particular selected trainer and output directories.

## Remaining acceptance gates

Run actual bounded native training for every exposed major family; validate no duplicate physical schedulers; compare original versus Foundry-controlled scientific outcomes; test device placement, CPU fallback, concurrency, OOM, pressure termination, retry, resume, idempotency, data provenance, GPU/host telemetry, reporting and checkpoint integrity. Do not report these unexecuted checks as passing.


## Source-revision binding for Foundry execution

The outer `SuiteCoordinator` plan signature now includes the complete native
scientific catalog **and the current committed Git HEAD**. Distinct revisions
receive distinct physical Foundry run identities even when their job arguments
are otherwise unchanged. Execution checks that native source/configuration,
the Foundry bridge, and pinned vendor runtime have no uncommitted tracked
changes or unexpected untracked source files. Inventory/report commands remain
read-only and do not require a clean Git worktree.

New worker specs record the revision; at launch, the worker verifies that the
current committed source revision still matches the planned one, before starting
the scientific subprocess. Failed revision validation archives any stale
success indicator and writes an explicit failure result. This closes a
provenance gap in which outer scheduler completion state could otherwise be
reused across different scientific code revisions.

The Git revision binds **committed source code** but does not independently
hash external datasets, system binaries, CUDA drivers, dynamic simulator
packages or all runtime-dependent data. These remain separate provenance
obligations and should not be described as scientifically validated merely
because the scheduler signature matches.

## Device-aware native admission

Foundry's physical GPU/CPU lane assignment is not the same as a native
scientific runner's model device. Each registered job now derives its
`gpu_capable` admission flag from the **resolved** native runner
configuration: `device="cpu"` is CPU-only, while an `auto` device remains
GPU-capable. This prevents explicit CPU-pinned development/diagnostic jobs
from occupying nominal GPU slots and distorting GPU-pressure scheduling.
Foundry's unchanged launcher masks accelerators for jobs admitted to CPU;
native `resolve_device("auto")` observes that mask. No scientific algorithm,
training budget, or benchmark protocol is altered by this classification.
The regression is a source-derived catalog contract test, not a live GPU
pressure/admission test. A CUDA-required native configuration cannot safely
migrate to CPU and must be evaluated separately before enabling such jobs.
