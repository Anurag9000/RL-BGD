# Training-only scientific-job catalogue — RL-BGD

> **Source-grounded static inventory, not a claim of completed or exhaustive executable runs.** Snapshot: `main` at `4fb3d759145f0d0149ae70e82bfece555512f055`. This file was derived from the complete non-truncated tracked Git tree and the native repository sources; no external datasets, generated local plans, checkpoint stores or execution environments were available.

## Scope and identity

A distinct scientific **training job** is one reachable, compatible trainable method/model + dataset/task stream + learner-visible protocol + architecture/objective configuration that materially changes training. Repeat seeds and evaluation-only routing/analytics are not additional scientific jobs. Do **not** infer all combinations from a registry, file names, or paper descriptions. Suites labelled ablation/sweep/smoke require per-cell classification before inclusion; ablations are excluded from the primary count, not erased from this source ledger. Controls that train valid baselines may be legitimate jobs and are not automatically discarded.

**Verified exact distinct training-job count: NOT ESTABLISHED.** The following counts are of **tracked candidate source files, not materialized jobs**: 23 configuration/registry files; 31 candidate launch/training scripts; 4 selected internal registry/dataset/experiment files; 2 configuration paths explicitly named ablation/sweep/smoke. Even when a planner exists, its realized count requires using its native compatibility checks, actual selected datasets and resource/seed/manifest inputs.

## Native scientific authority and central training path

- Primary evidence: [`docs/EXPERIMENT_REGISTRY.md`](../docs/EXPERIMENT_REGISTRY.md).
- Existing launcher/entry point: [`scripts/run_paper_suite.py`](../scripts/run_paper_suite.py).
- **Read-only inventory or planning command:** `python scripts/run_paper_suite.py smoke --output-root artifacts/suites`.
- **Native execution example (NOT run for this audit):** `python scripts/run_paper_suite.py smoke --output-root artifacts/suites --execute`.
- Protocol-specific caveat: Curated suite planner (`rl_bgd.experiments.suites`) materializes explicit runs and distinguishes canonical SAC/PPO, BGD variants, recurrent agents and information controls. `run_paper_suite.py` defaults to planning/materialization; `--execute` launches training. Sweep/ablation suites, mechanistic-only analyses and repeated seeds are not additional primary scientific training jobs.

The native launcher is authoritative for validation, deterministic job identity, dependencies, resume, resource checks and artifact storage. This index does not replace that code with a generic runner, does not create empty fake job records and does not declare a job completed merely because a configuration exists.

## Registered inputs in tracked tree (23)

These paths are an exhaustive **path inventory of files matching** `configs|suites|manifests|registry|specs` JSON/YAML/TOML conventions; their contents are **not** all individually certified to be runnable training jobs. Preserve per-file role and filter ablation/evaluation/diagnostic material when building native plans.

- `configs/algorithms/information_access_strict.yaml` — blob `f933b7551998`
- `configs/algorithms/ppo_adam.yaml` — blob `2ba9b4e8d69a`
- `configs/algorithms/ppo_bgd.yaml` — blob `33d8c7d4507d`
- `configs/algorithms/sac_adam.yaml` — blob `5efe87c58447`
- `configs/algorithms/sac_bgd.yaml` — blob `2f88296e8e2a`
- `configs/bayes/bgd.yaml` — blob `4c0e3da2993f`
- `configs/benchmarks/continual_world_canonical_cw10.yaml` — blob `b0a06878d604`
- `configs/benchmarks/continual_world_canonical_cw20.yaml` — blob `6f88dd9daa6b`
- `configs/benchmarks/continual_world_ta_cw10.yaml` — blob `2a5c5ee7ed34`
- `configs/benchmarks/continual_world_ta_cw20.yaml` — blob `00aaac52b653`
- `configs/benchmarks/three_rl_style_cw10.yaml` — blob `b6a9a554a7df`
- `configs/benchmarks/three_rl_style_cw20.yaml` — blob `65e84fa17026`
- `configs/environments/carl_pendulum_abrupt.yaml` — blob `940ca9e76c88`
- `configs/environments/carl_pendulum_recurring.yaml` — blob `7b6db4b4dfe9`
- `configs/environments/carl_pendulum_smooth.yaml` — blob `c3d2aec9b404`
- `configs/environments/cw10_canonical.yaml` — blob `b1e23210f9ff`
- `configs/environments/cw20_canonical.yaml` — blob `a70111a0a30f`
- `configs/environments/lqr_recurring.yaml` — blob `9fba55e46180`
- `configs/environments/lqr_smoke.yaml` — blob `990393dd9d66`
- `configs/environments/ta_cw10.yaml` — blob `e48ab0e93944`
- `configs/environments/ta_cw20.yaml` — blob `ff0d3e21bb17`
- `configs/paper/artifact_pipeline.yaml` — blob `d0f9c6883505`
- `configs/sweeps/evidence_temperature_lqr.yaml` — blob `0943c44a50e4`

## Possible execution and orchestration entry points (31)

These paths were conservatively identified by filename, not executed. Some may be development, infrastructure or evaluation-only; verify callers/CLI before treating any as training.

- `scripts/run_adaptive_bgd_lqr_stream.py` — blob `a62cea6dc5c1`
- `scripts/run_bgd_ppo_lqr.py` — blob `ad36aaa06e28`
- `scripts/run_bgd_sac_lqr.py` — blob `211b882da9da`
- `scripts/run_canonical_continual_world_sac.py` — blob `d6aadfaa595a`
- `scripts/run_carl_pendulum_sac.py` — blob `3e68b114c43e`
- `scripts/run_evidence_temperature_sweep.py` — blob `55c4d6afae63`
- `scripts/run_hidden_context_sac_comparison.py` — blob `c52f4097e1ef`
- `scripts/run_mechanistic_analysis.py` — blob `3847e0587acc`
- `scripts/run_paper_suite.py` — blob `b4441004058b`
- `scripts/run_paper_suite_parallel.py` — blob `d43c7e01757f`
- `scripts/run_ppo_continual_lqr.py` — blob `b9a4e7de3d87`
- `scripts/run_ppo_lqr.py` — blob `328b8c75454c`
- `scripts/run_quadratic.py` — blob `575a581533f4`
- `scripts/run_recurrent_continual_world.py` — blob `2b52636e7c19`
- `scripts/run_recurrent_ppo_continual_lqr.py` — blob `2f72e6470ee1`
- `scripts/run_recurrent_sac_continual_lqr.py` — blob `717616eb941c`
- `scripts/run_recurrent_stationary_lqr.py` — blob `a1b30a2bd982`
- `scripts/run_regularized_sac_continual_lqr.py` — blob `355d8e87b5c0`
- `scripts/run_regularized_sac_lqr.py` — blob `d20f306dae18`
- `scripts/run_replay_evidence.py` — blob `12e687d9f152`
- `scripts/run_sac_lqr.py` — blob `10f779398773`
- `scripts/run_ta_continual_world_sac.py` — blob `111d66b76ca7`
- `scripts/run_ucl_ppo_lqr.py` — blob `370ddb2431be`
- `src/rl_bgd/agents/ppo/recurrent_train.py` — blob `af7ad13ce619`
- `src/rl_bgd/agents/ppo/train.py` — blob `d0d94310afe9`
- `src/rl_bgd/agents/sac/recurrent_train.py` — blob `9c4963cd93b9`
- `src/rl_bgd/agents/sac/regularized_train.py` — blob `9fd66d06a944`
- `src/rl_bgd/agents/sac/task_aware_train.py` — blob `ec453c502a7e`
- `src/rl_bgd/agents/sac/train.py` — blob `0c494eab7fc9`
- `src/rl_bgd/artifacts/suite.py` — blob `95b0b21b017f`
- `src/rl_bgd/experiments/suites.py` — blob `67a411d96cbb`

## Additional registry, dataset and experiment source anchors (4)

This selected source index helps trace registered methods, dataset builders, manifest validators and experiment kernels. It is not a proof of reachability.

- `src/rl_bgd/experiments/__init__.py` — blob `91db1790e140`
- `src/rl_bgd/experiments/invoke.py` — blob `2d79a46f3c5a`
- `src/rl_bgd/experiments/parallel.py` — blob `50713bd32bdb`
- `src/rl_bgd/experiments/suites.py` — blob `67a411d96cbb`

## Explicitly named ablation/sweep/smoke config sources (2)

File naming alone cannot distinguish a legitimate comparison from a hyperparameter ablation. The primary non-ablation job planner should inspect the actual expanded cells, classify them and deduplicate on canonical scientific training identity, not on suite name.

- `configs/environments/lqr_smoke.yaml` — blob `990393dd9d66`
- `configs/sweeps/evidence_temperature_lqr.yaml` — blob `0943c44a50e4`

## Coverage, unresolved items and execution contract

| Check | Status | Evidence needed for closure |
|---|---|---|
| Tracked tree path census | **Complete at captured commit** | Non-truncated Git tree paths and blob identities above |
| Native runner located | **Source-identified** | Exercise help/plan in a checked-out environment |
| All methods and datasets actually runnable | **Unverified** | Native registry compatibility + prepared real data + dependency checks |
| Unique concrete training-job records | **Unmaterialized here** | Generate native plan, export canonical IDs and deduplicate scientific aliases |
| Seed/repeat-expanded execution count | **Unmaterialized here** | Explicit seed list, repeats, budget and a validated immutable plan |
| Trained results and resume integrity | **Not run** | Native checkpoints, data fingerprints, output manifests and successful completion evidence |

### Reproducible next verification

From a clean checkout at the pinned SHA, inspect the native CLI and run the **planning-only** command above. Archive the plan JSON/CSV, resolved configurations, rejected/skipped cells with reasons, input dataset identities, seed list, and source revision. Compare method/dataset/protocol membership with this inventory; only then calculate **exact distinct jobs** versus **execution records**. Execution is a separate operator action. Do not automatically download datasets or treat optional provider availability as success.

## Live source-derived Foundry inventory (subsequent implementation)

The earlier static census above intentionally did **not** establish an executable
scientific configuration count. The newer `foundry_integration/scientific_inventory.py`
enumerates **every declared** `SUITES` job (including ablation, uncertainty,
mechanistic and compute suites), resolves runner signature defaults when
dependencies permit, and separates seed-independent scientific fingerprints
from seed-expanded execution identities.

Run `python run_all_training.py inventory --write-inventory --out artifacts/foundry_training`
to export `scientific_job_inventory.json`, `scientific_job_inventory.csv`,
and `scientific_job_coverage.json`. These are real source-derived registry
counts, not executed training results. Every seed and optional dependency
requirement remains visible, and duplicate cross-suite aliases are listed.
The export does not yet prove exhaustive discovery of independent training
runners not present in `SUITES`; those remain an explicit audit obligation.
No numbers should be copied from another source revision.


### Scientific fingerprint boundary

The training fingerprint contains the native target, resolved method and
environment, information-access protocol, and scientifically relevant runtime
parameters. Seeds, output paths, physical model-device assignment,
checkpoint/logging coordinates, and selected summary-metric names do not
create new scientific training methods. The execution fingerprint retains
suite/job/seed identity, and the full native result independently records
the actual runtime settings. Different training horizon, objective,
Bayesianization, replay/evidence mode or posterior hyperparameters remain
distinct scientific configurations. This registry-level deduplication does
not prove that all independently reachable runnable native scripts have
been classified.


### Independent script-entrypoint audit

`foundry_integration/entrypoint_census.py` now scans every **present**
first-party Python script under `scripts/` and statically records its
`__main__` guard and exact imported `rl_bgd.runners` module/symbol pairs.
The scanner compares those pairs to declared suite targets and retains
`EXACT_SUITE_TARGET`, `SAME_RUNNER_MODULE_ONLY`, and
`NO_SUITE_RUNNER_MODULE` distinctions. Imports that do not exactly match
a suite target are **review candidates**, not automatically extra training
jobs: wrappers may import a generic `main`, and some standalone scripts are
development, orchestration or diagnostics rather than scientific methods.

`run_all_training.py inventory --write-inventory` produces
`native_entrypoint_coverage.json` alongside the training-job JSON/CSV
exports. This closes the *static enumeration of present script wrappers*
without claiming complete reachability through dynamic entrypoints or
full execution compatibility. Neither script names nor imported symbols
are independently counted as trainable scientific configurations.

### Foundry CPU-pinned vs GPU-capable native jobs

Registered native jobs that resolve `device="cpu"` are now declared
`gpu_capable=false` for Foundry admission, preventing CPU-pinned training
from consuming an advertised GPU process slot. `device="auto"` retains GPU
capability and follows the Foundry CPU lane's CUDA visibility mask. This is a
static scheduling-contract correction; physical CUDA occupancy, GPU OOM
behavior, preemption and host/device migration are not certified by it.

### Bidirectional native runner function census

The `native_entrypoint_coverage.json` export now inspects each tracked,
top-level **public function definition** in `src/rl_bgd/runners/*.py`,
not merely functions imported by scripts. It records which symbols are exact
`SUITES` training targets, script-imported but unregistered symbols, and
unreferenced `run_*` definitions. It also reports suite targets lacking a
local public runner definition for explicit audit, without assuming such
targets are broken (module-level reexports or dynamic resolution are possible).

The machine-readable `independent_runner_candidates` list is a **review
queue**, not an automatic collection of new training jobs. Individual callable
definitions can be helpers, wrappers, nested experiment modes or diagnostic
functions; neither callability nor a `run_` prefix proves distinct scientific
configurations or successful execution. Parameter-space exploration and dynamic
callable resolution still require separate review.

The bidirectional audit **separates targets outside** `rl_bgd.runners`
(e.g., `rl_bgd.analysis.mechanistic`) from registered runner targets that
appear to lack local public definitions. A non-runner target is not a missing
runner function, and its presence must not fail runner-coverage assertions.

Top-level `main()` functions are explicitly classified as
`CLI_MAIN_WRAPPER` and retained in the complete callable census, but
excluded from the independent scientific-job candidate queue. Their
underlying `run_*` functions and imported symbols remain visible. This
prevents a script forwarding to `main()` from being counted as a distinct
unregistered research model.

## Adjudication of independently defined unregistered runner functions

The 4 November 2026 two-way source census identifies these twelve distinct
*function entrypoints*. This is **not** twelve new distinct scientific
configurations: a single function may expose multiple legitimate experimental
arms, and multiple wrappers may run the same underlying learner.

| Native target | Source-derived classification | Remaining work, if any |
|---|---|---|
| `rl_bgd.runners.evidence_temperature_sweep:run_evidence_temperature_sweep` | `DECOMPOSED_SUITE_ARMS` | Delegates to per-temperature BGD-SAC runs already represented by evidence-temperature suite jobs; not a separate learner. |
| `rl_bgd.runners.hidden_context_sac_comparison:run_hidden_context_sac_comparison` | `DECOMPOSED_SUITE_ARMS` | Loops over five registered hidden-context variant arms; wrapper is not an independent scientific configuration. |
| `rl_bgd.runners.ppo_continual_lqr:run_ppo_recurring_lqr` | `UNREGISTERED_TA_RESEARCH_RUNNER` | Direct recurrent-free Adam/BGD PPO on a recurring LQR stream; needs explicit matched-configuration registration if included. |
| `rl_bgd.runners.quadratic:run_quadratic` | `SYNTHETIC_EXPERIMENT_NEEDS_ADAPTER` | Valid posterior-learning stream but requires QuadraticRunConfig object, not JSON kwargs; a tested native adapter is needed. |
| `rl_bgd.runners.recurrent_bgd_stationary_lqr:run_recurrent_adam_ppo_lqr` | `UNREGISTERED_STATIONARY_ACCEPTANCE` | Stationary recurrent Adam-PPO acceptance control; not a separately registered paper-suite job. |
| `rl_bgd.runners.recurrent_bgd_stationary_lqr:run_recurrent_bgd_ppo_lqr` | `UNREGISTERED_STATIONARY_ACCEPTANCE` | Stationary recurrent BGD-PPO learning acceptance; requires matched control/protocol registration. |
| `rl_bgd.runners.recurrent_bgd_stationary_lqr:run_recurrent_bgd_sac_lqr` | `UNREGISTERED_STATIONARY_ACCEPTANCE` | Stationary recurrent BGD-SAC learning acceptance; requires matched control/protocol registration. |
| `rl_bgd.runners.recurrent_ppo_continual_lqr:run_recurrent_ppo_recurring_lqr` | `UNREGISTERED_TA_RESEARCH_RUNNER` | Recurrent Adam/BGD PPO on hidden recurring LQR; not directly represented by SUITES targets. |
| `rl_bgd.runners.recurrent_sac_continual_lqr:run_recurrent_sac_recurring_lqr` | `INDIRECT_SUITE_ARM` | Adam/BGD/adaptive-BGD modes are invoked by registered hidden-context comparison variants; standalone default budget is separate. |
| `rl_bgd.runners.regularized_sac_lqr:run_boundary_regularized_sac_recurring_lqr` | `ORACLE_BOUNDARY_CONTROL` | Ground-truth switch steps trigger consolidation; must never be labelled strict task-agnostic or grouped as such. |
| `rl_bgd.runners.replay_evidence:run_replay_evidence_demo` | `DIAGNOSTIC_DEMO` | Synthetic replay-reuse posterior diagnostic, not a standalone reinforcement-learning training condition. |
| `rl_bgd.runners.smoke:run_smoke` | `SMOKE_ONLY` | Small quadratic implementation check, not an independently claimed paper experiment. |

The review is persisted in `foundry_integration/entrypoint_census.py` and
checked by the inventory tests. `main()` forwarding functions are present
in the full callable census but intentionally absent from this queue. The
unregistered PPO/recurrent PPO and stationary recurrent acceptance families
require matched budgets, seed semantics, comparison groups, primary metrics,
artifact schema and actual native target-call compatibility before promotion.
The oracle-boundary method must remain isolated from strict task-agnostic
comparison groups. Synthetic demos and batch wrappers must not inflate the
scientific training-job count. No experiments were run to write this ledger.

### Registered recurring-LQR PPO scientific arms

Two formerly unregistered runner targets are now explicit five-seed jobs in
`task_agnostic_final`: feed-forward PPO Adam/BGD and recurrent PPO Adam/BGD.
The generated catalog includes four scientific configurations and twenty
seed-expanded run records for these arms. Matching is **within** each model
family, not across feed-forward and recurrent streams because their source
context schedules differ. The removed manual-review census candidates were
replaced by real SUITES jobs; unregistered diagnostic and standalone acceptance
runners remain separately classified. These are declared training jobs, not
completed benchmark executions.
