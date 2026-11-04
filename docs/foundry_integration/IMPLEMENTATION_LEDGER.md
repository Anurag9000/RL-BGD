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

## Prelaunch retry-status isolation

The worker now archives the prior attempt's `result.json` and
`native_process.json` **before** validating its native configuration,
argv, selected job identity or Git revision. Errors in these preflight
checks are captured as a new failed `native_process.json` status, rather
than escaping before removing the old success indicator. Bounded regression
tests cover changed configurations and invalid argv without invoking real
training. An unreadable/invalid spec that cannot identify a result directory
is necessarily outside this result-directory cleanup contract.

## Native-wrapper preflight enforcement

A Foundry job with RL-BGD's registered SUITES source/contract/fingerprint
must invoke `python -m foundry_integration.rl_bgd_single` in the exact
expected module position. The worker rejects noncanonical wrappers before
launch, rather than accepting an arbitrary zero-exit command without
canonical scientific artifact verification. A native module string shifted
behind interpreter flags is also rejected; the generated native job catalog
always uses the canonical argv. These are fail-closed contract checks for
scheduler plans, not an arbitrary-code security boundary against malicious
local users who can edit the scheduler or the repository.

## Bidirectional source-runner static census

The independent script-entrypoint audit now additionally parses top-level
public functions declared in every present `src/rl_bgd/runners/*.py` file.
It cross-references symbols in both directions against the explicit `SUITES`
registry and standalone script imports. A reproducible review queue identifies
script-imported unregistered callables and independently defined `run_*`
functions not yet represented as suite targets. No unregistered entrypoint is
automatically scheduled; scientific distinctness and compatibility must be
established first. This is static AST evidence, not executed call-graph or
benchmark validation.

## Reject unconsumed RL-BGD input overrides

The generic Foundry command line offers `--config` and `--data` for
other repository adapters, but the RL-BGD `SUITES` adapter consumes neither.
Previously they were silently ignored, which could falsely imply a different
scientific configuration or input dataset was used. RL-BGD now rejects these
arguments with explicit errors before inventory materialization or job launch.
To configure native scientific experiments, update their explicit and tested
suite registration; do not assume the generic Foundry flags remap native
runner kwargs or provide CORA archives. Other supported adapters retain their
existing CLI behavior.

## Post-child source-revision validation

Git source validation is now performed both **before launch** and, for
zero-exit children, **after the subprocess returns but before native artifact
verification or Foundry success publication**. If the committed source or
worktree changes during an hours-long training subprocess, the worker writes
failed status and withholds its success marker. This prevents a mixed-version
training attempt from being certified solely under its prelaunch Git identity.
The test simulates a changed revision across the two checks without launching
research experiments. As with any point-in-time hash check, a transient source
mutation made and reverted during execution is outside this guarantee.

## Persisted CI audit evidence

The Foundry native-contract workflow exports all source-derived training
catalog and independent runner-census files as a named GitHub Actions artifact
(`rl-bgd-native-scientific-inventory`, 30-day retention). The workflow log
also shows all presently unregistered runner candidates, unmatched script
imports and registered non-runner scientific targets, retaining enough evidence
to review actual source omissions rather than relying solely on aggregate
counts. The artifact is generated without launching training, installing
optional simulators, or claiming scientific results. The 30-day CI artifact
is not a permanent repository snapshot; rerun inventory locally for later
audits.

## Source-adjudicated unregistered runner candidates (4 November 2026)

The expanded AST census found 23 runner modules and 52 public callables.
At this snapshot, 15 function definitions directly match declared `SUITES`
targets; one registered analysis target is appropriately outside the runner
package. The review queue, after excluding `main()` CLI forwarding wrappers,
contains twelve source-inspected `run_*` entrypoints. Each is manually labelled
with its observed role and source-grounded reason in
`foundry_integration/entrypoint_census.py`.

The machine report now exposes `unreviewed_runner_candidates` and
`stale_manual_runner_reviews`. Native inventory tests reject either in the
current repository so newly introduced or renamed unregistered entrypoints
cannot disappear from review. Diagnostic functions, wrapper orchestration,
indirectly registered methods and oracle-boundary controls are kept distinct
from genuinely not-yet-registered task-agnostic or stationary research jobs.
The detailed adjudication and required follow-up are in
`docs/TRAINING_ONLY_JOB_CATALOG.md`.

This is a manually curated **source-only** review, not a claim that the
unregistered experiments were executed or that all allowable hyperparameter
combinations are enumerable. No new job is added to `SUITES` without verified
scientific comparability and native invocation contracts.

## Source-change-triggered inventory refresh

The native Foundry CI workflow now watches the entire first-party
`src/rl_bgd/**` and `scripts/**` trees, both targeted census regression
tests, and the bridge/registry documents. This is necessary because a new
callable, changed native default, or new script import may alter the source
inventory without touching `SUITES`. Superseded pushes cancel older Foundry
inventory runs, while the latest run writes and uploads a fresh audit artifact.
This extends **static audit trigger coverage**; it does not mean independent
runnable configurations are automatically scheduled or validated on GPU.

## Original structured quadratic native runner adapter

A thin `run_quadratic_suite` function now adapts the original
`QuadraticRunConfig` dataclass to the stable JSON-keyword experiment
launcher. The original algorithm, `main`, defaults and optimizer behavior
are unchanged. `mechanism_analysis` registers five seeded native runs of the
original configuration and exposes them through Foundry with strict canonical
artifacts and source provenance. The census treats the original function as
an indirect suite arm rather than double-scheduling it. A bounded CPU unit
exercise and mocked delegation contract cover the adapter; expensive seeded
mechanism campaigns were not executed.

## Matched native recurrent Adam/BGD SAC control

The standalone stationary recurrent BGD-SAC acceptance runner now has a native
Adam-SAC comparator using its exact LQR horizon, GRU/Q widths, SAC learning
rates, train/evaluation seeds, replay budget and sequence windows. The original
BGD runner is unchanged. Two explicit `stationary_core` source-derived jobs
provide five-seed `stationary_recurrent_sac` matching, separated from
feedforward SAC and recurrent PPO studies. A bounded mocked call tests Adam
agent construction and training arguments; no expensive benchmark campaign
was executed. The runner census removes the obsolete unregistered BGD entry
only because both source implementations are registered.

## Source-registered oracle phase-boundary SAC controls

The four available native EWC/Online-EWC/SI/MAS modes of the existing
`run_boundary_regularized_sac_recurring_lqr` are registered as 20 seed-level
jobs in `baseline_core`. The original runner and consolidation algorithm
remain unchanged; only the suite registrations, coverage review, tests and
documentation are updated. The jobs share an explicit `oracle_boundary`
information-access contract and distinct environment/comparison-group
labels, preventing accidental pooling with strict task-agnostic regularizers.
No external benchmark campaign or expensive multi-seed run was launched.

## Fail-closed explicit Foundry selection

The native launcher now checks every requested `--job` and `--seed`
against actual catalog entries before inventory export or scientific
execution. It also rejects a combined selection that matches no runs,
without silently ignoring bad entries in a multi-value request. Mock-catalog
CLI regression tests cover invalid jobs, invalid seeds, mixed valid/invalid
selections, incompatible filters, and a valid selection. No scientific
training is executed by these tests.

## Oracle zero-penalty sham experimental control

The existing boundary-regularized SAC runner accepts an additional explicit
`regularization_strength` parameter, defaulting to the unchanged value
`1.0`. At `0.0`, EWC remains the configured regularizer and oracle phase
boundaries remain supplied, but its loss penalty is zero; consolidation
computation is not bypassed. A five-seed `oracle_boundary_sac_ewc_sham`
entry pairs this within-implementation control with the previously registered
four oracle regularizers using the same benchmark protocol and metric. Tests
verify zero-strength wiring and rejection of negative strengths without
running an expensive research campaign. Its result must not be represented
as a standalone Adam baseline or a boundary-free method.

## Native scientific suite identity collision guard

`validate_suite_registry` now rejects duplicate job identifiers within any
suite, duplicate seeds within a job, and invalid non-integer/negative seeds.
These checks protect the canonical `suite__job__seed` run-directory key before
materialization or execution, preventing unrelated scientific configurations
from colliding at one result path. Synthetic in-memory registry tests exercise
duplicate job and seed cases without launching training, and the inventory
validator retains its independent expanded-identity check.
