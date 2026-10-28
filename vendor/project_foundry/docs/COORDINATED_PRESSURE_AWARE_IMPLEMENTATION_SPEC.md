# Coordinated pressure-aware suite: implementation traceability

> **Distribution note (2026-10-23):** The source archive used during the original audit is excluded from this repository. Source-differential tests and archive-integrity tests were removed with it. Historical audit entries below describe evidence from the originating checkout; they are not current test results. The current suite uses self-contained contract and integration tests.

Current verification after removing the archive and its dependent differential tests: `python3 -m pytest -q` passed **18,362 tests in 153.61 sec** in the ProjectFoundry checkout. This result covers the remaining self-contained contracts and integrations; it does not include source-differential validation.

This file is the implementation-facing companion to
[`COORDINATED_PRESSURE_AWARE_RUNNER_LEDGER.md`](COORDINATED_PRESSURE_AWARE_RUNNER_LEDGER.md).
The ledger remains the detailed behavioral specification. This document maps its
source-observed contracts to the generic `simple_adp.runner` implementation and
records fidelity limits. It intentionally does not reproduce OPF task data,
case naming, model-head behavior, or OPF algorithm internals.

## Fidelity labels

- **Implemented** means a generic implementation exists in the named module.
- **Partial** means a path exists but some ledger details are absent, approximated,
  or have not yet been verified against focused contract evidence.
- **Excluded** means the mechanism is outside the user-defined generic scope.
- **Not implemented** means there is no corresponding implementation yet.

These labels describe source coverage, not runtime proof. Passing the unit suite
does not prove behavior on every OS, GPU driver, provider, or recovery state.

## Ledger-to-package map

| Ledger section | Generic implementation | Status and exact boundary |
|---|---|---|
| 1. Entry and wrapper chain | `runner/launcher.py`, `runner/controller.py`, `runner/watchdog.py`, `scripts/simple-adp-suite.sh`, `scripts/simple-adp-suite.ps1` | **Partial.** CLI/env/JSON precedence, dry run, detach, explicit exact-plan resume, report-only regeneration from persisted state, bundle ordering, manifest, controller child, watchdog, and POSIX/PowerShell entrypoints exist. Tests verify restore runs after creation of nested `run_root`, so the source empty-directory guard blocks default-path restore; disabling `auto_resume_bundle` skips restore and export but still rehydrates telemetry from the results directory; `test_launcher_rehydration_error_propagates_before_controller_even_when_bundles_disabled` pins setup-error propagation before controller launch/export; export still runs when enabled and the controller returns nonzero; resume reuses a persisted digest-checked plan, report-only mode does not schedule children, detach uses `<results>/run_root` while leaving a started process alive if PID-file creation fails, and `_run_with_watchdog` starts the controller before its watchdog, waits for the controller return code, then terminates and timeout-kills the watcher. The Bash entrypoint preserves caller thread settings, applies ledger defaults and best-effort priority, and retains the literal `python` fallback. The generic CLI intentionally has `suite` mode rather than OPF `case`/`massive`; task preparation is provider-owned. Linux launcher truth parsing now has parametrized coverage for every documented true/false token, whitespace/case normalization, and missing/unrecognized fallbacks. Windows wrapper behavior has static contract tests but no Windows execution evidence. |
| 2. Exact defaults and meanings | `runner/config.py`, `runner/plan.py`, `runner/launcher.py` | **Partial.** Central pressure defaults, capacities, retries, epoch-zero parsing, vanilla grids, and ADP grids are present. Initial width/depth coordinates now reject values below one or above their corresponding global caps, matching source validation. Tests assert documented defaults and deliberate validation holes; a 33-case differential contract compares shared `RunnerSettings` boundaries and permissive holes directly against the copied scheduler `validate_args`; zero-repeat and nondefault ADP/vanilla repeat asymmetry; concurrency-file valid/empty/zero/malformed/unreadable handling and live per-pass rereading, including proof that lowering the limit leaves already-live children running, plus env < settings JSON < `--set` < direct CLI precedence with explicit numeric zero preserved. Validation requires `missing_present_task_repeats >= 1` while still allowing `repeat_count=0`. All estimator-facing controls are explicit RunnerSettings values and are copied into each RunSpec: increments, patience/margins, learning rate and scheduler, AdamW settings, generic penalty weights, candidate cap, and depth seed width. The worker reconstructs SearchSettings from these values; smoke mode overrides only epochs and full-training-partition batch size. A future-provider plan contract verifies non-default settings survive planning, and the worker contract verifies those values reach the adapter. Before runtime bootstrap, the launcher applies configured CUDA visibility and allocator/workspace defaults only when their inherited environment values are unset or empty; nonempty inherited values retain precedence. Focused launcher contracts verify custom `--set` values reach bootstrap and the controller child environment, and cover absent, empty, and inherited values for all three settings. OPF-only no-op flags remain outside generic task scope. |
| 3. Run plan and deterministic identities | `runner/plan.py`, `runner/datasets.py`, `runner/models.py` | **Partial.** Deterministic identities include provider/task/split/route/seed; split hashes, explicit route enumeration, provider/task registration, and a full serialized-manifest SHA-256 digest exist. Job seeds use the ledger formula over generic identity fields: first eight SHA-256 bytes, modulo `2**31 - 1`, with the recorded nonzero fallback. `RunSpec.device_capable` is serialized with a default of true and is deliberately not consulted by device selection. `ModelRoute.suite_group` lets a registered adapter classify a new model without adding it to the built-in estimator registry; a synthetic provider/model contract now proves the planner creates its route. A new package entry-point integration contract discovers and registers an external dataset provider and model adapter alongside built-ins, then plans the custom route through the unchanged `SuitePlan` path. The built-in model adapter slices pandas tables positionally and NumPy-backed arrays by integer index, with classification/regression integration coverage for both provider shapes (including non-default DataFrame indices and unseen categories) and explicit metric-value assertions. A provider-plan contract exercises all three registered OpenML suite keys with multiple listed task IDs, verifies each task becomes model routes, and checks split hashes survive plan construction; task loading and split generation are fixture-backed, so this is not a live OpenML download or benchmark run. A separate classification/regression split contract verifies validation is carved only from the exact outer training indices, outer test indices stay unchanged and disjoint, classification labels are stratified, and the configured split seed is passed through. Manifest compatibility normalizes absolute symlink aliases only in job `argv[0]`; a top-level `python_executable` path remains exact, matching the ledger exception. The built-in adapter now has a direct route-count contract for all 14 canonical identities, both generic profiles for the eight mixed-order IDs, all 50 individual vanilla architectures plus the aggregate grid, and the 44 default screening routes; `test_hardness_screen_routes_are_additive_and_report_cannot_filter_plan` compares plans with screening enabled and disabled, then verifies an entirely incomplete/ineligible screen report leaves all experiment routes intact. Generic identity fields differ from OPF job IDs by design. Run settings, including estimator controls, are part of serialized run specs and the signed plan; provider/model plugin coverage verifies plan-level propagation, while worker tests verify estimator-control receipt. A launcher contract now passes `ex_k_width` and `learning_rate` through `--set`; a paired planning contract proves that changing `learning_rate` changes the signed plan identity and that the value reaches the worker specification. Runtime fixture coverage remains partial. |
| 4. Resource sensors and accounting | `runner/resources.py`, `runner/scheduler.py` | **Partial.** Host/swap, `nvidia-smi`, child-tree RSS/VRAM, external memory, and PID tracking exist behind a typed injectable `ResourceProvider`; external-memory exclusions include only active/suspended process trees with a live handle, matching source `current_pids`; child-tree sampling retains the known root PID if psutil inspection or descendant enumeration fails, matching the source fallback; active RSS is separately summed over all active child PIDs and returns zero if any RSS lookup fails; only the latest active child is sampled for soft-gate diagnostics, while pressure-victim candidates are sampled separately only when hard pressure is active; both paths recheck process liveness at the sampling boundary and return a zero footprint if the child has exited since candidate selection; external-host classification requests and reads cached `pid`, `name`, `cmdline`, and `memory_info` from `process_iter`; it applies stripped/lowercased source `python` substring and `py` prefix checks to name and joined command, falls back to `memory_info()` only when cached memory is absent, and skips a process on per-record errors. Focused contracts now distinguish the source's exception boundaries: an individual process inspection error is skipped and later processes are still counted, while an exception constructing/iterating the top-level process iterator propagates out of the sampler and controller pass. GPU queries use the source argument vectors without an added timeout, decode byte output, truncate total/used MiB through `int(float(...))`, preserve a zero-total sample with zero used percentage, and leave central used percentage above 100 when reported usage exceeds total while clamping free MiB at zero (`test_gpu_used_percentage_is_not_clamped_when_used_exceeds_total`); latest-child host/GPU remaining-memory diagnostics clamp at zero when the child footprint exceeds the sampled available/free amount, matching the source controller; the system callback adapter preserves the same sampler functions, while synthetic providers can drive coordinator accounting without OS or GPU dependencies. Mocked-source tests assert the exact cached-attribute request, Python-name/prefix/whitespace-normalized filtering, cached-memory fallback, and the fact that name()/cmdline() methods are not called, owned/Python GPU PID filtering, empty samples after GPU sensor failure, coordinator calls/exclusion sets, mixed-device latest-child sampling, host/GPU remaining-memory clamp behavior, and the latest-CPU-child footprint being reused for the GPU soft-gate estimate when the ledger's separate latest-GPU selection is active. The controller publishes the sampled resource snapshot to status and log records before changing gates; log overlays report current host/GPU usage and active CPU/GPU counts. The latest reconciliation found and corrected a source-order mismatch: immediately after active PID/RSS accounting, the source resolves launch capacity (including its own live concurrency-file read), applies the finite-concurrency `active_count + 1` ceiling using a separate concurrency-file reread, then computes strict host/GPU pressure, all before latest-child and external-process sampling. Contracts pin this ordering and simulate concurrency changing to the unbounded sentinel between the capacity and active-count reads. Gate-close side effects now match the source's transition check: an already-closed gate keeps its lock ID and lane-enabled value until a reopen/release path changes it; a regression contract covers the stale-lock/enabled-lane state that can follow completion. Additional Linux contracts now verify that host virtual/swap sensor errors propagate, malformed GPU-memory output becomes a missing sensor, child RSS sums the root and descendants while skipping inaccessible process memory, and only GPU-mode children query and aggregate VRAM for their own process tree. Default OS sampler failure boundaries are now covered, including top-level GPU memory query exceptions propagated through child and external GPU accounting; injected provider implementations retain their own documented exception behavior and need provider-specific contracts. |
| 5. Admission, gates, holds, and scheduler labels | `runner/admission.py`, `runner/scheduler.py` | **Partial.** Source capacity uses `int(max(0, resource) // reserve)`, including Python float floor-division edge behavior; minimum-one capacity, strict hard-pressure comparison, strict buffered soft gates, external PID-drop reopen, lane holds, source-specific scheduler labels (`pressure_aware` requires CPU-lane readiness for GPU launch), one launch per pass, and pressure victim selection exist. The regression `test_reserve_capacity_preserves_python_float_floor_division_boundaries` pins the `1.0 // 0.1 == 9` boundary that differs from `floor(1.0 / 0.1)`. GPU admission now also requires host availability to meet the host resume threshold, as specified by the ledger. Parent CPU-mask admission and child environment generation share one predicate that recognizes the source variables `TRAINING_CONTROL_BACKEND`, `CPU_ONLY`, `TRAINING_CONTROL_CPU_ONLY`, and `OPF_ADP_DISABLE_GPU_ACCELERATORS`, plus generic package aliases; live gate snapshots are included. Capacity tests cover host/GPU reserve floors, minimum-one capacity, GPU-sensor fallback to host slots, total active caps, finite active-count-plus-one growth, and unbounded concurrency. A synthetic live provider verifies active RSS remains diagnostic and OS available memory is used directly. An exhaustive 2^13 selector truth table crosses CPU-only and parent masks, host/GPU readiness and sensor presence, launch flags, gates, lane caps, total caps, and independent lane holds against a separate admission oracle. A 288-state gate-updater matrix crosses live-lane presence, GPU-sensor availability, hard-pressure flags, and below/equal/above child-buffer boundaries, checking lock IDs and lane-enable state. Table-driven tests also cover buffered equality, external-drop equality and strict-below threshold with positive-baseline/PID-disappearance requirements, host-resume equality and below-threshold rejection for GPU launches, scheduler-label behavior, including the CPU-hold distinction between `pressure_aware` and the other labels, and the source's unusual same-latest child sampling behavior, now including the mixed CPU/GPU footprint-to-gate association. A full controller-pass fixture now proves an exact external-PID drop reopens a closed GPU admission gate while a paused GPU child remains suspended below its lane's resume threshold. `test_full_pass_resumes_every_ready_job_before_caps_gates_and_holds` pins the source resume bypass: two ready GPU-paused jobs both relaunch in one pass despite total/GPU caps of one, closed host/GPU gates, active lane holds, and `batch_scale=0.2`; both receive the unscaled concurrency launch budget of five. A new 4-case CPU/GPU-by-PID-presence full-pass matrix asserts exact 500 MiB external-drop equality and exact lane-resume-threshold equality: a disappeared tracked PID reopens its gate, while a still-live PID leaves the gate closed even though an independently ready paused route can resume. Peer OOM and hard-pressure victim tie ordering both use the ledger's descending `(depth,width,job_id)` tuple from explicit route seed coordinates; the aggregate fixed-grid route uses its first architecture as its source-compatible initial coordinate. |
| 6. Child environment and process launch | `runner/runtime_tuning.py`, `runner/processes.py`, `runner/worker.py` | **Partial.** CPU/GPU environments, thread controls, seed, per-job command/log/state, process groups, and one-route worker are present. The Linux launch contract pins `start_new_session=True` for children. New queue launches now pass the sampled, resource-capped admission budget to child environment construction; pressure-resume launches pass the current concurrency as in the source helper. Environment construction occurs before log opening; command, metadata, and optional launch-context JSON are atomically written, with attempt/device/context recorded in metadata and job state. Logs use an append-mode UTF-8 text handle with the source job/attempt/device marker plus command/context lines; retry and finish markers use the same text handle, and retry closes the attempt log after writing its retry marker. Child environment construction returns an affinity partition without changing the coordinator process, supports explicit affinity slots and the shared-CPU helper mode, and preserves source CPU/thread metadata. **Affinity lookup matches the source fallback order:** a nonempty affinity in the supplied child environment wins; otherwise it falls back to the parent process environment, before partitioning. `test_child_environment_uses_parent_affinity_as_fallback_but_prefers_base_env` pins both cases. It reproduces the source OR semantics: inherited truthy `TABULAR_CHILD_SHARED_CPU` selects shared mode even when the caller passes `shared_cpu=False`; launch code later rewrites the variable to `0`, but the already-calculated thread/affinity settings remain shared-mode settings. Suite launchers normalize unbounded concurrency metadata to `1`. `current_concurrency_hint` now treats an invalid/nonpositive explicit hint as absent and falls back to `TABULAR_CPU_JOB_CONCURRENCY`; positive explicit hints retain precedence, with a focused contract for both paths. The selected worker `device_mode` is passed explicitly: CPU mode masks CUDA/HIP/ROCR, sets both JAX CPU selector variables, and selects CPU backend; GPU mode raises if central CPU admission is active, preserves an inherited single-device mask, requires the requested physical index in a multi-device mask, or assigns the requested index when no mask is set. Empty and `-1` CUDA masks count as CPU-only under the shared parent predicate, so an explicitly requested GPU child is rejected for those masks. Bootstrap independently applies its affinity and exports its own `scatter` OpenMP settings. OPF dispatcher import order, model alias registry, nested ported-process handoff, and OPF environment variables are excluded. The worker initializes backend hooks before loading plugin/model adapters. Torch probing falls back to CuPy; optional cuDF/cuML activation, NumPy/CuPy conversion, and backend-aware accuracy/seeding are available without making those packages dependencies. |
| 7. Job state, restart reconciliation, and resume | `runner/state.py`, `runner/processes.py`, `checkpoint.py`, `training.py`, `search.py`, `dedicated.py`, `generic_search.py`, `estimator.py` | **Partial.** Atomic JSON, broad read/parse-error fallback for persisted controller and job records, event CSV, candidate/method/task/replay compatibility views, live PID/exact-argv recovery using string-token/empty-token normalization and earliest-create-time/PID selection, unknown-exit requeue, source-root no-overwrite copy, and exact same-stage resume through canonical model routes exist. Runtime persistence now includes reason-specific GPU/host pause counters and the legacy `force_cpu_after_cuda_oom` field; recovery retains the counters and resets that legacy flag to false, so it never forces the retry lane. Recovered process handles return `None` while their PID is live and synthetic exit code 1 after disappearance; the scheduler converts disappearance into unknown-exit requeue. Completed zero-exit state is classified before recovered-live process handling using source-authoritative `last_exit_code`, preventing a stale PID from requeuing an already completed route even when current-attempt `exit_code` disagrees. A live process found for launching/retrying state is normalized and durably saved as running with recovered-from status/time; a running record with the same live PID is left unchanged. **Focused contracts pin three documented identity/reconciliation quirks:** a saved PID passing liveness is trusted without checking its command line, an exact-argv match can recover a persisted paused record as running because argv recovery precedes paused-state handling, and a persisted non-exact failure is queued on startup even when `retry_count` has reached `max_retries` (that limit is checked after the resumed attempt fails). The paused-record case remains a source-compatibility behavior, not a recovery guarantee. Malformed persisted numeric values safely default, and non-mapping launch-context/resource-snapshot values recover as empty mappings. Compatibility aggregation rejects non-mapping nested identities and tolerates non-mapping routes in task/replay views; a malformed legacy route does not prevent the current task summary from being written. Unknown job-state fields survive live-process recovery and re-save. Compatibility summaries now recursively scan sorted nested job-state paths, including states outside the current manifest (needed because generic run IDs contain path components), normalize unknown/launching statuses, select latest successful completion with the source predicate `ended.replace(".", "", 1).isdigit()` and then lexical string/job-ID sorting (including its permissive `.5` behavior), and retain first successful candidate ordering without metric ranking. Tests cover these aggregation rules and profiles. A contract fixture checks named top-level source-root files copy first, recursive files follow, existing destinations are preserved, and copied/skipped counts match. Conflicting completed-record exit fields now follow `last_exit_code` in reconciliation, with focused regression cases. Terminal exact-resume failures recognize both source status spellings (`failed` and `fail`) before attempting persisted-PID or exact-argv recovery; tests cover both forms with an apparently live PID. Checkpoint read, envelope, version, signature, controller-phase, architecture, global-best, optimizer/scheduler, and RNG-restore failures now carry the terminal exact-resume marker; focused tests prove the scheduler classifies a corrupt estimator checkpoint as `inexact_checkpoint_resume`. The nested controller remains outside the registered 14 routes, but its controller context, global-best snapshot, patience counters, and active inner candidate persist through stage callbacks and resume; an independent 3,456-case bounded oracle checks its width/depth trajectory, and interrupted/uninterrupted equivalence is verified at initial, first inner-depth, and later inner-depth stages. Pressure-resume launch context supplies the generic single signed estimator checkpoint path; it contains the current and best snapshots in place of OPF separate last/best/best-batch path names. The copied runner's OPF stage-sidecar schema is replaced by the estimator's signed checkpoint schema. |
| 8. Retry, failure, OOM, and shutdown | `runner/scheduler.py`, `runner/processes.py` | **Partial.** Last-attempt marker scan (including a three-case contract proving prior-retry markers are ignored), 64 KiB backward chunks, 200,000-byte fallback, last-4,000-character OOM classification, retry counts, OOM backoff floor, process-tree termination, signal shutdown, and inexact-resume terminal marker exist. OOM backoff now updates scale, reason, and the local-time `YYYYMMDD_HHMMSS` timestamp only when the floored next scale is strictly lower; at the `1e-6` floor it leaves the prior backoff record unchanged, matching the source early return. Non-OOM retry, successful completion, successful launch, and successful hard-pressure pause clear and immediately persist the timestamp with the backoff reason; an actual OOM reduction also persists its new scale and timestamp immediately. Tests cover reduction, floor no-op, persisted transition state, and non-OOM clearing. Device-mode gate release now matches the source helper on retry: CPU reopens only the host/CPU lane, while GPU reopens both host/CPU and GPU lanes; focused contracts assert the associated lock clearing and launch-enable flags. A paused-job relaunch error retains the runtime as paused with its original reason and persists `resume_launch_error`. `test_source_and_generic_resume_launch_failure_preserve_paused_state` directly compares the copied source helper with the generic coordinator for the failed-relaunch state fields (paused status, original pressure reason, cleared PID/exit code, and error reason); `test_full_pass_resume_launch_failure_keeps_pause_and_still_admits_queued_route` separately verifies that the generic full controller pass continues to ordinary queue admission after this failure. The copied main loop catches the helper exception and proceeds to later pass phases. `test_source_main_and_generic_resume_failure_still_launch_queued_route` now exercises that combined trajectory in both real controller loops through a subsequent successful queue launch, completion, and interrupt shutdown while retaining the failed route as paused. A real Linux parent/grandchild integration contract launches a child tree, invokes `terminate_process_tree`, and verifies both processes cease to be live. Retry state retains the prior attempt code and end time is cleared while retrying; retry events persist that code, including launch error code 1. Restart reconciliation reads the explicit prior code, with a fallback to legacy `exit_code`; failed process launches are appended to the queue tail and retried immediately inside the queue-fill loop; only one successful launch ends queue filling in each scheduling pass, while persistent uncapped launch failures retain the source's no-poll-sleep spin behavior. Focused contracts assert retry-to-tail ordering within a pass and central termination order: descendants terminate in reverse order, the root follows, wait timeouts are clamped to at least 0.1 seconds; if the first wait reports no survivors, only the root is waited on for 0.1 seconds; if processes survive, only those process objects are killed, waited again, and potentially killed a second time, then the root is waited with the configured timeout. The psutil path never calls `Popen.kill()`. A new contract pins the source edge where descendant enumeration errors propagate before any termination or wait call. The no-psutil-root branch is also tested to call `Popen.terminate()` and return without waiting. Shutdown preserves a paused job’s pressure reason, clears exit/end state, and records interruption separately. A synthetic arbitrary resource-sampler exception confirms the source behavior that unexpected controller exceptions propagate without invoking signal shutdown cleanup; the child remains represented as running in that failure path. Log flush and close are independent best-effort operations, matching the source helper. Linux contracts now dispatch each of SIGINT, SIGTERM, SIGHUP, and SIGQUIT through the installed handler and prove all previous handlers are restored; a separate KeyboardInterrupt run proves the same restoration and exit-130 path. A job-state write failure during shutdown is now injected: it propagates after child termination and job-state mutation, and prevents controller-state persistence, matching the direct shutdown call order. Peer CUDA OOM places the selected peer in the persisted suspended list and selects by the ledger's depth/width/job ID values. |
| 9. Standalone pressure wrapper | `runner/pressure_wrapper.py`, `runner/config.py` | **Linux behavior covered with source contracts and a real child-process-group stop/continue test under synthetic pressure samples; hardware-generated pressure events remain unverified.** Defaults `90/85%`, `500 MiB`, `0.5 sec`, both-domain resume, swap telemetry-only, GPU sensor failure as zero pressure, process-group STOP/CONT, and termination-on-signal are represented. An end-to-end regression contract pins that missing `nvidia-smi` yields zero GPU pressure and therefore satisfies the GPU half of resume once host readiness is met. The standalone GPU sampler matches the source query (no timeout), decodes the first output row, truncates total/used MiB with `int(float(...))`, clamps utilization to `[0,100]`, and returns `(0.0, 0)` on query/parse failure or nonpositive total. Python GPU usage uses the source two-column `process_name,used_memory` query, accepts digit-only MiB fields, and filters by process name alone. The generic wrapper samples this process-wide Python usage only after the successful pause operation and during resume polling; ordinary pressure polls query host/swap/GPU pressure without process enumeration or the second GPU query. Signal handlers immediately terminate the child tree and exit 130; the separate `KeyboardInterrupt` path continues a paused child before terminating it. Host Python memory is process-name based and reads cached `p.info["memory_info"].rss` directly; only `NoSuchProcess`, `AccessDenied`, `ZombieProcess`, `TypeError`, and `KeyError` are skipped, while other exceptions (including missing cached RSS as `AttributeError`) propagate. GPU Python memory is not associated with host PID identity. Console output prints the command at launch, pressure domains and thresholds when requesting pause, and external drops plus host/swap/GPU percentages when gates reopen; it emits no periodic pressure line during ordinary polling or while paused. Normal exit returns the child status; if the child exits while paused, the wrapper keeps waiting for both gates and launches a replacement after they reopen. Linux tests pin the live-poll versus paused-wait behavior, sampler arguments, truncation, clamping, missing-GPU behavior, same-child resume, signal-versus-keyboard-interrupt cleanup, both-domain resume gating, external-usage peak updates while paused, peak reset and immediate resampling after resume, process-name-only GPU accounting, exact host-RSS inspection exception boundaries, failed POSIX pause reporting with immediate resampling, the minimum 0.1-second poll sleep, helper-level child liveness checks before pause/resume signaling, POSIX `getpgid` lookup before `killpg`, the post-poll PID lookup and `NoSuchProcess` wait/exit path, and a mocked Windows process-tree path that suspends/resumes descendants before the root and ignores per-process access denial. `test_standalone_posix_wrapper_stops_and_resumes_a_real_child_group` additionally verifies real Linux process-group STOP/CONT and child completion with synthetic sensor samples. Physical host/GPU pressure and Windows runtime are outside acceptance scope. |
| 10. Runtime tuning and OS controls | `runner/runtime_tuning.py` | **Linux implementation and mocked contracts covered.** CPU discovery/frequency sort, swapped CPU-list ranges, even partitioning/remainder, explicit and SHA-1 slotting, shared/nonshared child thread environments, bootstrap affinity/thread exports, workers zero, best-effort priority, Linux memory locking, and systemd scope are implemented. Shared `bootstrap_runtime` now attempts systemd re-exec before affinity and thread configuration for launcher, controller, and worker entrypoints; the launcher no longer makes a separate duplicate attempt. Linux tests cover this ordering plus range/partition/hash/thread formulas, no parent-affinity mutation during child-env construction, shared mode, bootstrap variables, independent Torch thread-setter failures, broad affinity-error fallback, retrying all online CPU IDs when a syntactically valid configured mask is rejected, unexpected `sched_getaffinity` failure fallback, systemd probe arguments without an added timeout, first-line-only exact `systemd <integer>` version parsing, stdout capture/stderr discard, the `MemoryZSwapMax=0` gate at systemd >=253, and source-compatible fallback when `execvpe` raises either `OSError` or another exception; re-exec uses bare `systemd-run` argv[0] for PATH resolution, matching the ledger. Windows controls are outside this repository scope. Host-level priority, memory-locking, and systemd policy effects remain best-effort and are not asserted as applied during tests.
| 11. Results bundle, logs, and progress views | `runner/bundles.py`, `runner/telemetry.py`, `runner/scheduler.py`, `estimator.py`, `runner/worker.py`, `runner/models.py` | **Partial.** Level-1 gzip tar, 64 MiB parts, telemetry exclusion, empty-target restore guard, slash- and backslash-normalized traversal/link rejection, telemetry shard checksum/path checks, progress sidecar precedence, CSV/JSONL counts, regex fallback, and reports exist. Live-summary contracts now pin newest-sidecar selection, malformed-sidecar suppression of log/state fallback, and existing-log suppression of persisted-state fallback. The worker now writes generic `telemetry/<simple_adp>/<job_id>/telemetry.csv` and atomic `progress_state.json` callbacks for batch, epoch, and stage-state records; counters resume from progress JSON and fall back to CSV. Child telemetry stores architecture, stage, epoch/batch cursor, patience, best loss, checkpoint path, and state only, not task metrics. Final report Markdown is atomically written with the JSON path, renders batch/epoch progress totals, and includes the timestamp; suite-summary Markdown renders the same completion/failure/retry/progress totals without the timestamp. A focused contract checks normal and persisted-state output. The live status resource snapshot is recorded before gate mutation, and its values become the resource context for subsequent logs. Contract tests cover telemetry exclusion, nonempty restore guard including a failing directory-listing fallback, monolithic restore with a nonexistent destination parent, slash- and backslash-normalized traversal rejection, shard checksums/root containment, malformed-manifest skipping, missing-shard residue, checksum-failure cleanup, part restoration, the inclusive archive-size threshold (at the configured threshold the archive remains monolithic; above it, parts are emitted), stale bundle/part cleanup, CSV-over-JSONL precedence, callback integration, resume counters, live status aggregation including malformed overflowing persisted counters, report/task-summary contents, and prior attempt-code retention in retry state/events. The threshold test uses 128-byte fixtures; production `PART_SIZE_BYTES` remains 64 MiB. Multipart restore now creates the target parent before reading archive parts, matching the source ordering; a fixture verifies parent existence at the first part read. Restore ordering is covered through `launcher.main` with the real empty-directory guard and mocked controller; bundle archive extraction is covered separately. |
| 12. Backend admission guard | `runner/admission.py`, `runner/backend.py`, `runner/worker.py` | **Implemented/adapted.** Admission rejects contradictory CPU/GPU assignments and stale accelerator contexts. A usable backend probe allocates, fills, and synchronizes Torch first, then tries CuPy. Optional cuDF pandas and cuML hooks preserve import-order warnings and activation errors; CPU workers reject CuPy and accelerator tensors before inspection/conversion. NumPy/CuPy selection and conversion, backend-aware accuracy, four-name status reporting, seed precedence, and CPU-versus-GPU RNG seeding are implemented. Optional packages remain lazily imported and are not required dependencies; mocked hook/probe/conversion contracts pass. On 2026-10-23, the available RTX 3050 was also checked with the actual GPU child environment, Torch allocation/fill/synchronization, and the `nvidia-smi` sampler; this verifies the live backend and sampler path only, not training or physical pressure response. |
| 13. Source conflicts and snapshot caveats | ledger | **Distribution scope.** The upstream source archive is excluded. Historical source references are retained as provenance; production modules do not depend on that archive. |
| 14. Exact central-controller cycle and transitions | `runner/scheduler.py`, `runner/admission.py`, `runner/state.py` | **Partial.** The controller has a fixed ordering for sample, pre-gate status/log context, gate update, suspended resume, child reap, pressure victim, and queue admission. Mocked fixtures assert that the resource snapshot precedes gate mutation, external host-PID disappearance reopens the host gate and permits a paused CPU route to resume and complete in one controller run, a host-domain reopen still requires the paused GPU route to meet the unconditional GPU resume threshold, a full simultaneous-pressure pass terminates the original GPU process and restarts on the same lane with a new PID after resources recover, resume precedes reap, reap precedes pressure, pressure precedes new admission, one successful queue launch occurs per pass, positive post-launch delays sleep after launch while a zero delay skips that sleep call, failed launches retry to the queue tail inside the fill loop, and `test_queue_fill_rereads_concurrency_after_launch_failure` verifies a concurrency-file reduction between attempts changes the next queue budget before another route can launch. `test_launch_context_records_scaled_and_resource_capped_budget` verifies child context records the effective concurrency-times-backoff budget capped by resource capacity, alongside the separate resource cap, live counts, host sample, and pressure thresholds. An already-active failed child is reaped and appended behind the existing FIFO queue before the queued routes and retried route complete in order, and a disappeared recovered child is requeued as unknown-exit behind existing FIFO work, then relaunches and completes without being counted as a failed attempt. A separate full pass proves that all ready paused jobs resume before queue admission even when they exceed concurrency/active caps and both admission gates and lane holds are closed. A new full-pass launch-failure interaction injects an error while relaunching a pressure-paused child, then verifies it stays paused with its original cause while the queued route is admitted and later signal shutdown persists both states. Simultaneous host/GPU pressure labels both causes but selects one GPU victim by GPU footprint. Normal drain zeros logging lane counts before final status and writes reports afterward. A successful hard-pressure pause clears the transient backoff flag, reason, and timestamp. Pressure-pause and live-recovery state now retain source `ended_at: null` semantics, and pressure pause explicitly clears `last_exit_code`; a failed child attempt retains its end timestamp while queued for retry, until the next launch resets it. Resume now terminates any leftover live process before relaunch, clears the pause reason to an empty string at launch, and preserves the prior cause if relaunch fails. CUDA-OOM peer pausing and GPU memory-baseline updates are restricted to failed GPU-lane jobs, and a missing GPU sample leaves the prior baseline unchanged. A full pass combines a ready paused GPU route, OOM retry-to-tail and depth/width peer selection, then a second hard-pressure victim pause; resume occurs before failure classification, the OOM scale remains reduced while the hard-pressure path clears its backoff reason, and hard pressure blocks queued admission before shutdown. Final-report tests now assert the full summary key set and every task-summary CSV column/value, in addition to telemetry totals and Markdown timestamps. A new 64-case compatibility-state truth table crosses eight status values (including null, empty, and unknown) with eight exit-code values (including malformed, fractional, boolean, and conflicting current versus last exit code); it verifies normalized counts, success selection, latest-success job-ID ordering, first-state best-directory selection, and replay ordering. The later report/state schema crosswalk closes the field-inventory gap; controller interleavings remain partial. The combined controller branch coverage is now recorded below; it remains synthetic and is not an exhaustive source-versus-generic execution comparison. A new 16-case pressure-victim matrix crosses all four host/GPU pressure combinations with CPU-only, GPU-only, both-lane, and no-live-child sets; it asserts one-victim maximum, GPU-only eligibility under GPU pressure, host-lane eligibility under host-only pressure, and GPU selection under combined pressure. Invocation initialization is tested to reset persisted transient gates, holds, backoff and pressure/external samples to baseline while retaining unknown persisted fields. A full controller pass also exercises GPU lane-cap fallback to CPU followed by both post-launch holds blocking a third route despite free total capacity. A new full-pass edge contract verifies an active runtime with a missing process handle is removed without inventing an exit status, matching the source loop's `proc is None` branch. The full-pass fixture caught and fixed incompatible keyword arguments in `_resume_ready()` before the acceptance path could be exercised. The device-selector branch matrix and resume bypass of capacity, active caps, gates, and holds are covered. `test_full_pass_exit_pressure_and_peer_lane_cross_product` exercises 16,128 synthetic full-controller passes across seven exit kinds, four pressure combinations, four peer-lane sets, three total-cap values, three GPU-cap values, four gate states, and four hold states. The focused matrix passed (16,128 passed in 71.77 sec). Assertions cover exit/retry state, peer and hard-pressure victim interactions, persisted pause events, and OOM backoff. This closes the prior gap in combined synthetic branch coverage; it does not compare every pass to an executing copy of the source controller or exercise physical pressure hardware. |
| 15. Fine-grained runtime tuning | `runner/runtime_tuning.py` | **Linux source contract covered; live host effects unverified.** See section 10. The coordinator requests partitioned mode. Linux contracts pin allowed-CPU discovery, frequency-file precedence and CPU-ID tie-breaking, online-ID fallback after invalid syntax or rejected configured affinity, best-effort `nice(-20)`, `ionice -c2 -n0 -p <pid>`, `SCHED_BATCH` priority 0, and `mlockall(3)` while verifying failures are swallowed independently. Mocked systemd contracts assert the complete `systemd-run` command, each scope property including the systemd-253 `MemoryZSwapMax=0` gate, CPU affinity, opt-out/already-scoped/interactive/missing-executable branches, and exec-failure fallback; they do not prove host-level policy effects. Windows is outside acceptance scope.
| 16. Fine-grained source coverage ledger | [Source/function map](COORDINATED_PRESSURE_AWARE_SOURCE_FUNCTION_MAP.md), this table, and named modules | **Implemented for the in-scope generic runner.** The map links all source-index concern groups to target generic modules/functions and tests, distinguishes adapted/excluded/unverified behavior, and includes a regenerable AST inventory of all 310 function/method/nested-callback definitions across 26 runner modules. It is regenerated and checked with `scripts/generate_runner_function_map.py`; copied-source task/model/dispatcher internals remain excluded at the ledger boundary. |
| 17. Source index | ledger | **Implemented as reference.** Detailed copied-file provenance remains in the canonical ledger. |
| 18. Helper and record contracts | `runner/config.py`, `runner/plan.py`, `runner/resources.py`, `runner/state.py`, `runner/compat.py` | **Partial.** Public runner records are typed and serializable. Candidate, method, task, and replay view schemas now capture generic route identity, paths, attempts, device/status, `device_capable` metadata, pause counters, the always-false compatibility CPU-fallback view, aggregate counts, latest successful completion, first-success candidate, and replay candidates. Compatibility success selection and replay exit values use the source-authoritative `last_exit_code` field; a contradictory `exit_code`/`last_exit_code` regression contract passes. Runtime recovery restores host/GPU pause counters, resets the compatibility flag, and `RecoveredProcessHandle.poll()` uses synthetic exit 1 after PID disappearance. Controller defaults/reset and sampled resource state now persist `last_child_job_id` alongside latest-child memory values; a mixed-device contract verifies it points to the general latest active child even when the source's separate latest-GPU measurement is unavailable. Job-state and candidate/method/task/replay `updated_at` values now use local `%Y%m%d_%H%M%S` timestamps, matching the source helper; candidate summaries also persist schema 1, exit code, reason, attempt, device, and update time. A deterministic contract pins these fields and formats across all generated records. Tests assert device capability round-tripping/default compatibility, runtime status priority, paused/live PID recovery, recovery metadata persistence, invocation reset of persisted transient backoff/gate/pressure fields with preservation of unknown fields, event CSV schema/header/append/snapshot/local-time formatting/best-effort `fsync` behavior, the recorded unused parsing/signature/formatting helper contracts, and sorted/indented strict JSON serialization without a trailing newline through atomic replace. `JobRuntime.derived_status` now follows the source priority using `last_exit_code` as authoritative over current-attempt `exit_code`: ended zero-exit, live process, paused, missing exit, then failed/nonzero or completed/zero (including the empty-ended-at fallback). Contradictory current/last exit fixtures pin these branches. Invalid UTF-8 tests now pin broad source-compatible fallback behavior for controller state, job state, nested compatibility aggregation, result JSON, and persisted status JSON; malformed stage sidecar bytes are converted to the required exact-resume failure. Exact key/value contracts now also pin the complete default `ControllerMemory` record and all durable `JobRuntime.to_record()` fields, including optional recovery metadata and omission of live process/log handles. Launcher truth parsing now covers every token and fallback; parent CPU admission tests exercise every documented backend and truthy environment key; `load_json` fallback covers invalid UTF-8. Atomic JSON replacement failure is pinned to propagate while retaining the previous target and temporary file; log-close contracts prove close is attempted after flush failure and close errors are swallowed; emergency cleanup tests pin descendant/root order, PID deduplication, documented psutil race swallowing, and propagation of unexpected iterator/tree/kill errors. |
| 19. Static contract tests | `tests/test_runner_contracts.py`, `tests/test_runner_models.py`, `tests/test_runner_bundles.py`, `tests/test_runner_pressure.py`, `tests/test_runner_tuning.py`, `tests/test_runner_resources.py`, `tests/test_runner_child_telemetry.py`, `tests/test_runner_state_views.py`, `tests/test_runner_launcher.py`, `tests/test_runner_shell_launcher.py`, `tests/test_runner_compat.py`, `tests/test_runner_processes.py`, `tests/test_runner_logging_context.py`, `tests/test_runner_function_map.py`, existing runner tests | **Partial.** A subset of contracts are exercised; no claim of complete source contract coverage. The latest full Linux suite passed 18,969 tests on 2026-10-23 in 204.59 sec after adding an executing source differential for nested outer-watchdog traversal, including the real Linux child-process-group STOP/CONT integration under deterministic sensor samples and the Linux watchdog master-liveness contract. Existing coverage includes the 37-case differential validation contract against copied scheduler boundaries and permissive holes, plus source-grid default checks for all ADP and vanilla route coordinates. The shared estimator and model registry neuron-cap default is pinned to the documented 1,000,000,000 value. It includes a stubbed Linux shell-entrypoint contract for caller thread-setting preservation, thread defaults, priority command arguments, and forwarded launcher arguments. It includes a process-launch contract for the source's inherited shared-CPU OR behavior: an inherited truthy flag selects shared thread/affinity settings despite a `shared_cpu=False` call, while launch code later rewrites the flag and concurrency metadata. The expanded suite verifies all 46 copied reference files against their manifest byte counts and SHA-256 values, rejects unlisted snapshot files, exercises bundle path derivation for resolved in-repository and anchorless external results paths, plus archive extraction rejection for absolute paths, unexpected roots, symlinks, and hardlinks, rejects telemetry shard paths escaping their manifest directory, and pins partial rehydrate-temp residue on missing-shard/path failures; it also includes invalid-UTF-8 fallback for controller/job state, backward current-attempt log scanning beyond the 200 KB fallback window across multiple 64 KB chunks, compatibility aggregation including malformed unhashable route fields, worker result and persisted status, malformed-stage-sidecar exact-resume errors, and unreadable existing-manifest mismatch persistence; outer restore/export ordering, launcher child/watchdog startup-wait-timeout cleanup, configuration precedence, detach PID-write failure behavior, resume plan reuse, report-only generation with exact persisted-summary fields/status counts/telemetry fallback/CSV rows, POSIX watchdog timing/cleanup order, bounded and uncapped launch-error retries (including immediate queue-fill retry without poll sleep) and same-pass retry-to-tail ordering including disappeared recovered-child unknown-exit handling, descendant/root termination and survivor kill ordering including no-psutil fallback, plus a real Linux parent/grandchild tree termination check, exact emergency entrypoint ownership without substring or `py_compile` source-argument false positives, external-PID gate reopening and paused CPU/GPU resume/reap in full controller passes, including the case where a gate reopens but the child remains paused below its lane threshold, failed pressure-resume launch with retained pause cause followed by queued admission and signal shutdown, CPU-completion gate retention with CPU fallback, combined-pressure GPU-victim priority with no eligible GPU process, a 16-case pressure-victim matrix, a 12-case full-controller pressure/live-lane matrix, and a 12-case full-pass success/retry/OOM by host/GPU pressure matrix and a 16,128-case full-pass cross product of seven exit/device combinations, all four pressure states, all four CPU/GPU peer-lane sets, both active caps at 0/1/2, all four gate combinations, and all four lane-hold combinations; it also checks FIFO queue admission, GPU-cap CPU fallback, and OOM backoff reset after a successful launch, host-only pressure pausing the CPU worker, persisted transient-state reset, GPU-cap fallback and lane-hold interaction, a 16-case full-pass cross product of host/GPU gates and lane holds verifying GPU preference, CPU fallback and queue blocking, a 1,152-case full-pass cross product of CPU/GPU live peers, four hard-pressure states, all gate/hold combinations and disabled/one/two total and GPU active caps, plus a 9-case live-child full-pass cross product of total/GPU active caps verifying cap rejection and CPU fallback, a 36-case full-pass cross product of total/GPU caps and all four pressure states verifying pressure pause and CPU fallback, same-lane pressure pause/relaunch, the two-job pressure-resume admission-bypass contract across total/GPU caps, closed gates, active holds, and unscaled concurrency launch-budget forwarding, resource-capped child launch budgets, retry/OOM backoff reduction, floor no-op, persisted OOM backoff, and non-OOM timestamp clearing, CPU/GPU retry gate-release contracts, and a combined paused-resume/OOM-peer/retry/hard-pressure-victim/queue-block/shutdown pass, runtime-state recovery including trust of a live saved PID, the paused-record/exact-argv recovery quirk, and earliest exact-argv match selection without cwd comparison, candidate/method/task/replay serialization and aggregation including permissive dotted numeric timestamp classification and the 64-case status/exit-code compatibility matrix, event-CSV existing-empty-file header omission and falsey-snapshot fallback, emergency cleanup descendant/root order, PID deduplication, documented race swallowing and unexpected-error propagation, checkpoint durability/RNG behavior, child telemetry sidecar resume plus newest/malformed sidecar and existing-log fallback precedence, file-like stage-state reads/writes and raw-text wrapping, report generation, admission and resource branch matrices including top-level process-iterator failure propagation versus per-process skip behavior, GPU memory query exceptions across main/child/external samplers, unclamped over-total GPU percentage with zero-clamped free memory, latest-child remaining-memory clamp and per-sample child-liveness recheck behavior, logging handler preservation and GPU sensor-failure context, exact standalone GPU query/truncation/clamping and signal-versus-KeyboardInterrupt cleanup, Linux runtime-tuning contracts, exhaustive bounded ADP growth/capacity plus fixed-grid contracts, the 3,456-case nested-controller oracle with later-outer-round resume equivalence, 16 same-stage resume interruption boundaries, controller-level optimizer/scheduler reset across two architecture stages, all registered model/task one-epoch smoke routes including both generic profiles, additive screening route/report behavior, and progressive-depth checkpoint equivalence across all catch-up and completed-candidate stage boundaries for initial widths two and three. |
| 20. Child stage checkpoint and exact-resume contracts | `checkpoint.py`, `training.py`, `search.py`, `dedicated.py`, `generic_search.py`, `estimator.py` | **Implemented for registered estimator routes; nested controller has direct stage/controller resume.** The checkpoint stores current/best architectures and snapshots, controller counters/history/context, optimizer, scheduler, Python/NumPy/Torch RNG, sampler permutation/cursor/generator, epoch/batch position, preprocessing/task state, and a complete data/configuration signature. The model configuration portion signs every mechanism-affecting `ModelConfig` field as well as its public ID, so remapping an ID to another controller/growth policy fails exact-resume validation; descriptive text does not affect compatibility. The payload also records the resolved mechanism fields for inspection. It fsyncs the checkpoint file and containing directory after atomic replacement; CUDA RNG is captured only when CUDA is already initialized. `TabularADP.fit` loads only a matching signature. Dedicated width/depth/fixed-grid and generic candidate search reconstruct the interrupted stage, finish it, apply its acceptance rule once, then continue at the next candidate. Progressive depth catch-up is covered. Width transitions are exhaustively checked over small architecture states; depth eligibility and both insertion primitives now also have a finite exhaustive matrix over depth 1–3, widths 1–3, depth caps 1–4, neuron caps 1–12, and both seed widths. Reconstructing saved controller-best snapshots preserves Torch RNG both during search and resume, so bookkeeping constructors do not shift fresh parameters for later growth candidates. The guide's nested outer-width/inner-depth controller remains an explicit `SearchController.run_nested_width_depth` method outside the 14-ID registry; its stage/controller context is bound to settings, task metadata, initial architecture, training/validation arrays, width-growth strategy, and RR cursor, and rejects incompatible resume before restoring the active candidate. Contracts verify all-layer and round-robin outer width strategies, nonuniform-stack depth eligibility, changed-strategy rejection, and interrupted/uninterrupted round-robin equivalence from a nonzero cursor. `test_nested_controller_resumes_interrupted_inner_depth_candidate` also verifies settings mismatch rejection. |
| 21. Child telemetry and stage sidecar integration | `runner/telemetry.py`, `training.py`, `search.py`, `estimator.py`, `runner/models.py`, `runner/worker.py` | **Partial.** Generic progress callbacks now flow from the worker adapter through `TabularADP.fit`, `SearchController`, and `StageTrainer` into CSV/progress sidecars. Batch and completed-epoch counting, final state records, progress-JSON-first counter restore, and CSV fallback are covered. The child-writer fallback now matches source `TrainingTelemetry._load_progress_counts`: arbitrary progress-file read/parse/conversion failures fall through to CSV, valid persisted counts are returned without nonnegative clamping, and CSV `record_type` takes precedence over `kind`; offline suite summaries retain their separate `kind`-first precedence. Focused tests pin conflicting CSV fields, invalid UTF-8 progress JSON, and negative persisted counts. A stage-start callback records a new architecture before its first batch; generic `stage_state.json` plus uniquely named RNG companion checkpoints record controller phase, architecture/cursor, epoch/batch/patience/best-loss, and estimator checkpoint path. The telemetry Torch helper uses a fixed sibling `.tmp` path and atomic replacement without fsync, matching the ledger; the authoritative estimator `CheckpointStore` retains its separate unique-temp, file-fsync, replace, directory-fsync protocol. On signature-validated resume, malformed or mismatched identity/path/phase/cursor state fails closed; the active estimator checkpoint owns RNG. General same-stage monotonic width catch-up requires a stable stage key and matching current/best tensor layouts. Depth-stage catch-up additionally requires identical explicit `{stage_index, start_width, target_width}` metadata in the sidecar and estimator checkpoint; `stage_index` must be nonnegative and equal architecture length minus one, `start_width` must be positive, `target_width >= start_width`, and every saved and recovered layer width must stay inside the inclusive band. Monotonic per-layer catch-up and identical best/current tensor key/shape sets are also required. Accepted catch-up atomically updates sidecar widths/cursor. Regression contracts cover a non-indexed layer outside the band and architecture length inconsistent with the stage index. Integration coverage interrupts training after the first batch and resumes through `TabularADP.fit`. Per-batch exact resume remains in the atomic estimator checkpoint. A generic file-like stage-state surface provides `exists`, `read_text`, and `write_text`, wraps invalid text as a `raw_text` state, and persists the JSON sidecar, RNG companion, CSV row, and progress state. Reads defer RNG restoration when architecture rebuild or a complete active estimator checkpoint owns continuation; legacy sidecars without architecture/checkpoint metadata restore their RNG companion. Missing, malformed, or unsafe companion state fails with the source-compatible exact-resume error. If JSON is absent, stage lookup falls back to the latest job/model-matching `stage_state` CSV row, converts only recognized `completed` strings to boolean, and leaves other CSV values as strings before applying the same RNG rules. The default CSV schema intentionally omits the RNG companion name, active-stage flag, architecture vector, exact batch cursor, and stage checkpoint path, matching the copied source schema; JSON-sidecar loss therefore leaves an observation row that normally fails exact stage recovery because its RNG companion is unavailable. Appends preserve a legacy CSV header without shifting old columns; an old header only restores RNG when the row actually identifies a safe, existing companion. Task-specific telemetry remains out of scope. |
| 22. Windows supervisor and wrappers | Windows-only scripts and branches | **Out of scope per user direction.** Linux is the target platform. Windows runtime, PowerShell parser, power-scheme, and thermal-wrapper verification are not completion gates. Existing cross-platform files and static tests are retained but are not used to claim Linux fidelity. |
| 23. File-level audit boundaries | ledger | **Implemented as scope boundary.** The new package does not load copied OPF runner modules. OPF task-data acquisition and model training implementations are excluded. |
| 24. Backend admission and logging side effects | `runner/backend.py`, `runner/logging_context.py`, `runner/launcher.py`, `runner/controller.py` | **Partial.** Generic Torch backend admission and logging record-factory resource context are implemented, including default host/GPU sampling before a custom overlay. Logger configuration preserves existing stream handlers, supports named loggers with propagation disabled, and retains a GPU process count when CUDA is available but memory sampling fails. The OPF optional-backend counters and exact message format are replaced with simple-adp resource fields. Torch-to-CuPy fallback has a parameterized Linux contract for every documented Torch probe failure class at each probe step. Admission and device resolution read only the worker process environment, as in the ledger; explicit environment-map injection was removed from these runtime APIs and `test_backend_admission_and_device_resolution_use_process_environment_only` locks that boundary. |
| 25. Dispatcher import/argument/nested-child edge cases | `runner/launcher.py`, `runner/worker.py`, `runner/plugins.py` | **Adapted to the planned generic worker interface.** Explicit entry-point registration replaces OPF's model-name scan and aliases; typed serialized `RunSpec` values replace unknown-argument coercion; one supervised worker per route replaces the nested native/ported dispatcher and its stream tee. Worker initialization, plugin route planning/execution, launch logging, POSIX process groups, and Linux signal/watchdog cleanup are covered by `test_entry_point_plugins_extend_dataset_and_model_routes_without_coordinator_edits`, `test_coordinator_worker_process_runs_plugin_dataset_through_builtin_adp_routes`, `tests/test_runner_processes.py`, and `test_linux_watchdog_signal_cleans_real_controller_process_tree`. OPF import-time dtype mutation and model-specific aliases remain outside the generic adapter contract. |

## Exact defaults retained by the runner

The source defaults remain enumerated in `RunnerSettings` and
`StandalonePressureSettings`: concurrency `0` maps to `10**9`; active caps
`0` disable caps; repeat/retry defaults `5/1`; host available-memory stop/resume
`1024/1280 MiB`; GPU free-memory stop/resume `100/500 MiB` on device 0; reserve
`0/0 MiB`; child buffers `500/100 MiB`; external drop `500 MiB`; poll `0.5 sec`;
launch hold `30 sec`; OOM batch scale `0.5`; CPU-only false; swap `70/70%`
telemetry-only; width/depth/neuron caps `512/5/1,000,000,000`; epoch zero means
unlimited; base seed `42`; CUDA allocator
`expandable_segments:True,max_split_size_mb:128`; cuBLAS workspace `:4096:8`;
vanilla depths `[1,2,3,4,5]`; widths `[1,2,4,8,16,32,64,128,256,512]`;
the parsed compatibility grid `width_to_depth_widths=[1]` (combined-growth
job generation uses the separate scalar seed settings `width_to_depth_width=1`
and `width_to_depth_depth=1`);
standalone wrapper pressure/resume `90/85%` for host and GPU, external drop
`500 MiB`, poll `0.5 sec`.

The exact formulas, state ordering, and edge cases for these values remain
specified in the ledger. This companion does not replace those descriptions.

## Verification performed in this checkout

- The attached `pasted-text-1.txt` is 121,713 bytes and 582 lines. The
  repository ledger preserves its content with two expanded source-audit notes:
  section 4 records the distinct per-process versus iterator-level exception
  boundaries in external-memory sampling; section 9 records the live-child
  `poll()`/`NoSuchProcess` branch in the standalone pressure wrapper. The files
  are not byte-identical; the repository additions document source details used
  by the implementation audit.
- The attached 74-line plan is the saved implementation plan, wrapped with
  `<proposed_plan>` and the citation block in the repository copy.
- The copied OPF runner snapshot passes its 46-entry manifest byte-count and
  SHA-256 integrity check.
- `.venv/bin/python -m pytest -q`: **18,426 passed in 100.51s** (Linux, refreshed 2026-10-22, after the pressure-resume handoff, telemetry checkpoint-writer, and function-map updates). This includes a 37-case differential validation contract against the copied scheduler's shared argument boundaries and permissive holes, and regression contracts pinning source ADP/vanilla grid defaults including the compatibility-only width-to-depth list and the 1,000,000,000 model/search neuron-cap default. Coverage includes model/provider planning (including entry-point plugin registration),
  preprocessing and estimators, checkpoint resume, launcher/watchdog behavior,
  resource and admission rules, retry/recovery, pressure pause/resume, report-only
  mode, and synthetic full controller passes.
- A Linux restore contract confirms backslash-separated archive paths are
  normalized before traversal checks and cannot extract outside the results root.
- A four-case full-pass matrix samples host and GPU external-memory drops
  simultaneously while resuming paused CPU and GPU routes under active caps;
  it verifies each gate reopens only when its own tracked external PID disappears.
- A 16-case full controller-pass matrix crosses both admission gates and both
  lane holds, confirming GPU preference, eligible CPU fallback, and blocked
  queue behavior after the ledger-ordered sample/gate/admission path.
- A 9-case full controller-pass matrix keeps one GPU child live while crossing
  total active caps of disabled/one/two with GPU caps of disabled/one/two; it
  verifies total-cap rejection, GPU-cap CPU fallback, and eligible GPU launch.
- Generic ADP checkpoint coverage now interrupts and resumes the first
  candidate stage for all eight mixed-order configurations in both faithful and
  corrected profiles, and compares search history, forward architecture, and
  predictions against uninterrupted runs.
- The focused worker contract sets ordinary training to nine epochs and batch
  size eight, then verifies smoke mode passes exactly one epoch and a batch size
  equal to the two training indices in the persisted split to the model adapter.
  This proves the smoke override at the coordinator-to-model boundary without
  launching the OpenML suite.
- `test_runner_end_to_end.py` launches the real coordinator and three real
  worker subprocesses for `LGND-W-Greedy`, `ADP-W-RR`, and `ADP-W-All` using a
  dynamically registered dataset provider. The integration verifies completed
  result records, one-epoch smoke mode, and the split hash at the worker output.
  Split planning rejects a provider-supplied hash that differs from the saved
  train/validation/test indices before a worker is launched.
- A Linux GPU-environment integration contract verifies the ledger's physical
  device mismatch: an inherited single-device mask of `2` remains on the child
  with local GPU index `0`, while coordinator host/child/external GPU samplers
  continue to query configured physical index `0`.
- A model-adapter integration contract runs one-epoch classification and
  regression jobs from NumPy-backed provider data, confirming custom dataset
  providers do not need pandas or the optional scikit-learn dependency for core
  estimator execution.
- Estimator checkpoint loading now tags unreadable/corrupt files, malformed
  envelopes or payloads, unsupported versions, and signature mismatches as
  `exact-resume failure`; a controller contract proves a corrupt-checkpoint
  error is classified as terminal `inexact_checkpoint_resume`, matching the
  ledger's no-retry rule for unrecoverable resume state.
- Stage restore wraps model/optimizer/scheduler/RNG restoration failures with
  the same exact-resume marker. Controller resume validation checks the selected
  controller and phase, stage/current architecture agreement, history/counter
  types, and required global-best snapshots before continuing; regression tests
  cover malformed RNG and a missing dedicated-controller best snapshot.
- Focused report contracts verify the complete runtime summary field set and
  telemetry counts, matching final/suite JSON payloads, Markdown totals and
  timestamp difference, and the full task-summary CSV schema and rows in both
  live and persisted-state modes. Persisted-state reporting covers all status
  categories, launching/unknown/missing-state normalization, coercion of retry
  counts, saved resources, sidecar and state-field telemetry fallback, and its
  source-compatible telemetry directory count. Candidate-state and candidate-
  summary schemas, all generic candidate-state fields, architecture and task
  identity serialization, and the source-compatible always-false candidate OOM
  lane flag are asserted. Method, task, and replay compatibility documents now
  have full key-set contracts plus path, family, count, latest-success, first-
  successful-result, and exact candidate-entry checks, including no-success
  behavior where best-result and latest-success fields are null. CPU/GPU
  completion contracts verify device-based gate release, device-specific lane
  enablement (GPU completion enables both; CPU completion enables CPU only),
  preservation of a still-closed GPU gate, and transient backoff clearing. A
  complete synthetic controller pass verifies CPU completion preserves a
  closed GPU gate and admits the queued route through CPU fallback.
- Synthetic controller passes compare host-only pressure with combined pressure
  when only a CPU worker is live: host-only pressure pauses it, while GPU-first
  combined pressure finds no eligible GPU victim and leaves it active for that
  pass; both conditions keep queue admission blocked.
- A complete two-pass controller fixture first admits CPU because the GPU sensor
  is unavailable, then admits GPU while the CPU post-launch hold is still active.
  This verifies the per-lane hold rule through queue admission, alongside the
  existing fixture where a GPU hold permits CPU fallback and both active holds
  block a later queued route.
- A GPU external-PID drop of exactly `600 MiB` with PID disappearance reopens
  the GPU gate; a paused GPU route resumes on GPU and completes in that pass.
- The same-device pressure-pause fixture asserts the resumed worker receives
  the canonical signed estimator checkpoint path plus the source-compatible
  `checkpoint_last`, `checkpoint_best`, and `checkpoint_best_batch` paths under
  its job directory. The coordinator records these handoff paths without
  validating or restoring them; the built-in estimator continues exact resume
  through its unified signed checkpoint.
- A separate complete controller pass confirms that an external GPU drop can
  reopen admission without resuming a paused GPU route until the lane's
  `500 MiB` free-VRAM threshold is also met.
- Launcher-level `provider:suite` selectors are covered for custom-provider task
  listing, plan construction, and persisted-plan resume; bare names still resolve to OpenML. The
  launcher now reads the dataset registry's actual `.providers` mapping for
  plan creation and resume validation.
- Standalone wrapper contracts now test strict `>` pressure boundaries, swap as
  telemetry-only, and the both-domain resume rule with the exact external-drop
  `>=500 MiB` alternate in each domain. A Linux integration contract launches a real
  child in its own process group, requests SIGSTOP with synthetic pressure samples,
  then resumes it with SIGCONT when both gates clear. Interrupt cleanup while paused
  also verifies continue is attempted before process-tree termination.
- `bash -n scripts/simple-adp-suite.sh`: passed.
- `scripts/simple-adp-suite.sh --help`: passed without starting the suite.
- `.venv/bin/python -m compileall -q src/simple_adp scripts tests`: passed.
- `.venv/bin/python scripts/generate_runner_function_map.py --check`: passed; all 310 runner function/method/nested-callback definitions match the current AST across 26 modules.
- ADP guide fidelity is tracked separately in [ADP algorithm implementation audit](ADP_ALGORITHM_IMPLEMENTATION_AUDIT.md); the latest suite covers all eight mixed-order route mutations, a 3,456-case nested-controller oracle and nested resume equivalence through a later outer-width round, an explicit inclusive outer-width `delta_width` equality boundary, an 864-case progressive/matched-depth controller oracle, interrupted-candidate resume equivalence for both dedicated depth variants, width-to-depth generic phase-transition resume equivalence under both profiles, a new 16-case full two-phase trajectory matrix across all eight mixed-order IDs and both profiles, exhaustive small-domain width transition/capacity and depth transition matrices, a 12-case dedicated round-robin controller oracle crossing patience, delta, and growth cursor, direct dedicated Adam/cosine-restart and generic AdamW/weight-decay/clipping/scheduler-profile assertions, generic primitive nonpositive-size clamping, plus all registered model/task one-epoch full-training-batch smoke routes, both generic profiles, dedicated growth, patience, rollback, optimizer, generic-quirk, and fixed-grid traversal contracts.
- Backend contracts parameterize all four documented Torch probe failure classes across availability, allocation, fill, and synchronization, and verify fallback to a usable CuPy probe.
- Windows runtime and PowerShell checks are outside the target platform and are not used as acceptance evidence.
- No dataset suite or long-running training campaign was launched.

## Remaining work before claiming ledger fidelity

### Bounded source recheck: child checkpoint sidecar schema (2026-10-22)

Compared ledger sections 20–21's stage-state checkpoint metadata with
`TrainingTelemetry.save_stage_state` and `ChildTelemetryWriter`. The reference
writes stage sidecars with schema `2`, while progress sidecars separately use
schema `1`. The generic writer had incorrectly used schema `1` for both and
validated that value during checkpoint reconciliation. Updated the generic
stage writer, reconciliation guard, and CSV recovery view to use stage schema
`2`; progress schema remains `1`. Focused Linux verification:
`.venv/bin/python -m pytest -q tests/test_runner_child_telemetry.py` —
**28 passed in 1.94 sec**. This closes the version-field mismatch only; full
checkpoint and telemetry source reconciliation remains open.

After this correction, `.venv/bin/python -m pytest -q` passed on Linux:
**18,427 passed in 100.70 sec**. This remains test-suite evidence, not a live
pressure run or complete source-line reconciliation.

### Bounded source recheck: Linux runtime tuning (2026-10-22)

Re-read ledger sections 10 and 15 against the copied `utils/runtime_tuning.py`
and `src/simple_adp/runner/runtime_tuning.py`. The compared mechanics were CPU
discovery/frequency ordering, affinity fallback, slot partition and SHA-1
assignment, positive concurrency-hint precedence, thread/worker budgets,
shared-mode environment behavior, best-effort POSIX priority and `mlockall`,
systemd re-exec skip/fallback rules, version-gated zswap property, and bootstrap
ordering. No mismatch was found in this bounded comparison. Existing focused
contracts in `tests/test_runner_tuning.py` cover these branches; they verify
requested calls and environment values, not host-level priority, locking, or
systemd effects. This closes only the runtime-tuning slice; overall ledger
reconciliation remains incomplete.

### Bounded source recheck: launcher imports and runtime bootstrap (2026-10-22)

Compared the reference `opf_suite_launcher.py` import list and `main()` order
with `runner/launcher.py`, its plugin/model import chain, and the Bash entrypoint.
The apparent import-before-bootstrap ordering is source-compatible: the
reference outer launcher imports `opf_dataset_cache` at module load, which
imports Torch and PyG before `main()` calls `bootstrap_runtime`; the generic
launcher imports its model adapter chain before its matching bootstrap call.
The shell wrappers both establish thread/priority environment before Python,
and both leave systemd scope handling to the Python runtime bootstrap. No
ordering change is warranted from this comparison. This check closes only
launcher import/bootstrap ordering; controller branch combinations and the
remaining ledger sections are still under review.

### Bounded source recheck: central scheduling pass order (2026-10-22)

Compared ledger section 14's pass sequence with the reference loop in
`opf_massive_suite_runner.py`, `SuiteCoordinator._sample_snapshot`, and
`SuiteCoordinator.run`. The shared behavioral order is present: status and
concurrency resolution; host/GPU and child/external sampling; capacity and
strict pressure; snapshot and gate transitions; suspended resumes; active
reaping/failure handling; at most one pressure pause; live recount; then FIFO
admission. Both paths recalculate the effective scaled budget at admission
after resume/reap/pause effects, retry a launch failure at the queue tail
within the same pass, and stop after one successful launch. The source's
pre-sample `launch_budget` raise is superseded by its later admission-time
recalculation; the generic coordinator likewise calculates the effective
budget at admission from the current concurrency and saved resource capacity.
Existing full-pass contracts include a 16,128-case exit/pressure/lane/cap/gate/
hold product, plus separate resume/external-PID and OOM/peer-pause interactions.
No mismatch was found in this bounded order comparison. The finite matrices do
not exhaust every possible combination of custom providers, process races, or
numeric settings; the remaining source review stays open.

The same source pass exposed one missing pressure-resume handoff: the generic
launch context carried its unified estimator checkpoint but omitted the three
source paths. `_resume` now also records `checkpoint_last`, `checkpoint_best`,
and `checkpoint_best_batch` under the route's coordinator job directory while
retaining `checkpoint_path` for the built-in exact-resume worker. The focused
same-GPU pause/relaunch contract verifies all four values; it passed on Linux
(1 test, 1.63 sec). The updated source function map check passed, and the full
suite then passed **18,425 tests in 99.21 sec**. These three handoff files are
not created or validated by the coordinator, matching the ledger's
responsibility boundary.

The ledger's telemetry writer also exposes `checkpoint_last.pt`,
`checkpoint_best.pt`, and `checkpoint_best_batch.pt`, with save methods that
optionally mirror the same Torch payload to a stage path. `ChildTelemetryWriter`
now exposes those paths and methods using its existing fixed-suffix temporary
file plus replace behavior (no fsync), and the new contract verifies exact
byte equality between each local file and its same-basename mirror. The full
Linux suite passed **18,426 tests in 100.51 sec** after this addition.

The copied Python snapshot contains no production training-engine call sites
for these three methods. Its only call sites are in the copied contract test,
which supplies the same fixture payload independently to each method and
checks byte-identical stage mirrors. The generic package provides and tests
that writer surface, while the estimator uses the signed `CheckpointStore` as
the authoritative exact-resume record. The coordinator records the three
source-compatible paths in pressure-resume launch context but neither creates
nor validates them. No selection rule is inferred for `checkpoint_best_batch`;
the source snapshot does not establish one.

### Bounded source recheck: Linux child CPU and device environment (2026-10-22)

Compared ledger sections 6 and 15 with the reference `launcher_child_env` and
`make_child_env`, plus the generic `launcher_child_env`, `launch_job`, and
`choose_device` paths. The generic implementation retains the recorded
frequency-ordered CPU set, positive concurrency hint, hash/explicit slot
partition, inherited-affinity fallback, shared-mode OR behavior, thread
variables, zero worker count, CPU accelerator masks, inherited single-GPU mask,
multi-GPU index membership check, and GPU-first fallback constraints. The
central launcher rewrites the unbounded sentinel metadata to `1` after the
thread helper has consumed the `10**9` hint, as recorded in the ledger. No new
mismatch was found in this bounded comparison. Existing Linux contracts cover
these formulas and child environment branches; they do not prove that
best-effort host priority or affinity changes succeeded on a particular host.
This closes only the reviewed child environment slice.

### Bounded source recheck: retry, OOM, and shutdown (2026-10-22)

Compared ledger section 8 with `scheduler.py` and `processes.py` for retry
limits, latest-attempt log scanning and classifier windows, exact-resume
terminal handling, GPU peer selection, OOM backoff persistence/floor behavior,
launch-error retry-to-tail behavior, and signal/exception shutdown paths. The
review found no new mismatch in this slice. Focused source contracts cover the
documented paths, including intentional behaviors such as uncapped launch-error
spin, recovered unknown exit status, and arbitrary controller exceptions
bypassing shutdown cleanup. This does not replace the open review of combined
controller branches and other ledger sections.

### Bounded source recheck: result bundles and telemetry views (2026-10-22)

Compared ledger section 11 with the copied `utils/opf_resume_bundle.py` and
the generic bundle, telemetry, launcher, and report paths. The reviewed order
and edge behavior matches: manifest-first export, level-1 gzip and 64 MiB
splitting, telemetry CSV exclusion with size still counted, nonempty-target
restore refusal, multipart reassembly residue on extraction failure, normalized
member traversal/link rejection, repository-contained shard targets, checksum
validation, existing-target/malformed-manifest skips, and progress-view
precedence. No new mismatch was found in this bounded comparison. The generic
path and identity schema are intentional adaptations; Linux behavior is covered
by bundle, launcher, telemetry, and report contracts. The remaining global
runner audit and live external bundle restore/export remain distinct checks.

### Bounded source recheck: launch capacity, gates, and holds (2026-10-22)

Compared ledger section 5, the copied `resolve_launch_capacity`, and the
generic admission implementation and controller contracts. This uncovered a
real reserve-boundary mismatch: source computes `int(max(0, resource) //
reserve)`, while the generic code had used `math.floor(resource / reserve)`.
These differ for some binary floats (for example, `1.0 // 0.1` is `9`, while
`floor(1.0 / 0.1)` is `10`). `resolve_launch_capacity` now uses the source's
floor-division operation for both host and GPU reserves, with a regression
contract pinning the 9-slot outcome. The remaining gate/hold logic reviewed in
this slice matched the ledger; its tests include the full selector truth table,
gate-update matrix, resume bypass behavior, and controller pass cross-products.

### Bounded source recheck: child environment and process launch (2026-10-22)

Compared ledger section 6 with the copied child-environment/launch functions
and `runtime_tuning.py`. The reviewed sequence matches: calculate thread and
affinity settings using the launch budget and stable job identity, apply
device-specific masks and backend flags, persist command/metadata/context and
`launching` state, append the attempt marker to the job log, spawn the exact
argument vector from the repository root with merged stdout/stderr and a new
POSIX session, then persist PID and `running` state. A failed `Popen` closes
the opened log handle and returns to scheduler retry handling. Generic
environment variable names and the caller-supplied repository root replace
OPF-specific values by design; dispatcher/import behavior is separately
excluded as OPF-specific. Focused process and launcher integration contracts
cover metadata, checkpoint handoff, inherited environment precedence, shared
CPU budgeting quirks, device-mask behavior, process-group setup, and launch
failure cleanup.

The combined `test_runner_launcher.py` + `test_runner_end_to_end.py` run exposed
a test-isolation issue rather than a runner defect: the in-process launcher
contract intentionally applies `custom_alloc` to `os.environ`, and its test
fixture originally left that sentinel in the parent process. Subsequent worker
children inherited the invalid allocator value and exited during Torch startup.
The test now restores the caller's three runtime variables after its assertions.
The combined focused run then passed: **41 passed in 7.15s**. This is evidence
for the environment handoff in this test sequence, not a blanket guarantee that
arbitrary inherited allocator values are valid for Torch.

### Bounded source recheck: backend admission, seed, and logging context (2026-10-22)

Compared ledger sections 12 and 24 with the copied `utils/ml_backends.py`,
`utils/logging_utils.py`, and their generic ports. Torch/CuPy allocation and
synchronization probes, CPU admission checks, optional-hook import-order gates,
context memoization, accuracy conversion, host/GPU log sampling, record-factory
overlay order, and named/root handler behavior matched in the reviewed paths.
The comparison found one seed-precedence difference: the generic helper
consulted `SIMPLE_ADP_SEED` after `OPF_ADP_GLOBAL_SEED` was absent, while the
ledger specifies explicit integer, `OPF_ADP_GLOBAL_SEED`, then `42`. Removed
that unrecorded fallback; the runner's planned seed is passed explicitly.
`tests/test_runner_backend.py` passed (**26 passed**) after the correction,
and the regenerated source function map check passed with that backend test
(**27 passed**). The subsequent full suite passed **18,425 tests in 97.80s**.
CPU/backend environment aliases remain explicit generic integration
extensions and do not change source variable precedence when canonical names
are used.

### Bounded source recheck: helper and persisted-record contracts (2026-10-22)

Compared ledger sections 18–19 with `runner/compat.py`, `runner/admission.py`,
`runner/state.py`, `runner/scheduler.py`, `runner/resources.py`, route records,
and the copied helper contracts. Checked absent-versus-invalid environment
fallbacks, safe numeric conversion, dropped empty CSV tokens, exact-two-token
range parsing, primitive/sorted-JSON formatting, canonical JSON hashing,
local-time second-resolution timestamps, non-fsync atomic text replacement,
append-only CSV behavior, invocation reset of transient pressure state,
preservation of unknown persisted fields, derived job-status priority,
recovered PID liveness, and compatibility-view scan/tie ordering. The generic
helpers preserve these shared behaviors; field names are adapted to
`simple-adp`, and OPF case/method/task directory naming is not copied. The
reference snapshot integrity test also passed. Focused Linux verification:
`.venv/bin/python -m pytest -q tests/test_runner_compat.py
tests/test_runner_state_views.py tests/test_runner_reference_snapshot.py` —
**28 passed in 0.98 sec**. This closes only the helper/record slice; source
reconciliation and other partial ledger sections remain open. The broader
persisted-state/gate/resume selection
`.venv/bin/python -m pytest -q tests/test_runner_contracts.py -k
'batch_backoff_state or resume or gate or hold'` passed **5,865 cases**
(11,710 deselected) in 30.18 sec. The generated source function inventory
passed `scripts/generate_runner_function_map.py --check` immediately before
that run.

The telemetry and exact-resume integration selection
`.venv/bin/python -m pytest -q tests/test_runner_child_telemetry.py
tests/test_search_training.py -k 'telemetry or resume or checkpoint or stage'`
passed **61 tests** (484 deselected) in 2.68 sec. This covers current callback,
writer/mirror, and same-stage estimator checkpoint behavior; it does not
establish a best-batch selection policy absent from the source.

1. Expand contract fixtures across all controller branch combinations and
   per-lane capacity/gate/hold interactions. OOM/generic markers and malformed,
   missing, and checksum-failing telemetry shards have focused contracts; full
   cross-branch controller behavior and remaining unusual compatibility-view
   edge combinations still need ledger-by-ledger review. A focused mixed-lane
   contract pins the source quirk where the latest CPU child's zero GPU
   footprint is used for the GPU soft-gate estimate while the gate is labeled
   with the separately selected latest GPU job. A new 12-case
   end-to-end controller matrix drives all four host/GPU pressure combinations
   through CPU-only, GPU-only, and mixed live-lane sets. It checks eligible
   victim selection, GPU priority under combined pressure, reason strings, and
   shutdown persistence. This complements the existing lower-level 16-case
   victim-selection matrix. A 16,128-case full-controller cross product now combines seven exit/device
   combinations, all four pressure states and CPU/GPU peer-lane sets, total and
   GPU active caps of 0/1/2, every gate combination, and every lane-hold
   combination. It checks retry/terminal outcomes, OOM
   backoff and peer pause, pressure-victim eligibility, empty-peer behavior,
   post-exit FIFO admission, GPU-cap CPU fallback, and the successful-launch
   reset of OOM backoff.
   A 36-case full-pass matrix also crosses disabled/one/two total and GPU
   caps with all four pressure-domain combinations, proving same-pass GPU
   victim pause, eligible CPU fallback, total-cap rejection, and host-pressure
   blocking.
   A 1,152-case full-pass cross product now also combines CPU/GPU live peers,
   all four hard-pressure states, every host/GPU gate and hold combination,
   and disabled, one, or two total and GPU active caps. It checks victim pause before queue
   admission, the total-cap budget after a pause, GPU-to-CPU fallback, and
   post-signal persistence of launched and paused routes.
2. Exercise complete controller passes against synthetic resource and process
   providers across pressure, exit, gate, hold, resume, retry, and queue
   interactions without launching the real OpenML suite. The current tests now
   cross a paused resume, OOM failure/retry, OOM peer pause, hard-pressure
   victim pause, blocked FIFO admission, and shutdown in one pass; separate
   complete passes pin external-drop gate reopening without lane resume,
   uncapped immediate launch-error retries, and the arbitrary-exception path
   that bypasses shutdown cleanup. A new four-case CPU/GPU-by-PID-presence full-pass fixture
   combines exact 500 MiB external drop, tracked-PID disappearance or
   retention, exact matching-lane resume threshold, gate state, same-lane
   relaunch, and successful reap. Retained PIDs prevent external gate reopen
   even though a route meeting its lane threshold can independently resume.
   An eight-case CPU/GPU × tracked-PID-disappearance/retention × success/retry
   full-pass matrix combines external-drop detection, paused same-lane resume,
   holds, queued admission, exit classification, and shutdown. It captures gate
   state at resume launch: a disappeared PID opens the gate, while a retained PID
   leaves it closed even though matching-lane readiness independently permits
   the paused route to resume. Completion or retry may release the gate later in
   the same pass, so the contract observes the pre-exit launch boundary. Success
   completes the resumed job; retryable failure appends it after the existing
   FIFO route. Both outcomes keep queued launches blocked until signal shutdown.
   A separate four-case pass starts with both host and GPU gates closed and
   both CPU- and GPU-paused jobs ready to resume. It varies host and GPU PID
   disappearance independently and checks each gate at both resume launches;
   this confirms the two external-drop decisions remain domain-specific when
   both are evaluated in the same sample.
   Full-pass matrices now cover pressure-by-lane and this external-drop/hold/
   resume/exit interaction, but do not combine every remaining controller branch
   into a single exhaustive product.
   Compatibility-view contracts pin equal-timestamp job-ID tie-breaking against
   first-sorted-success `best_candidate_dir` selection, plus aggregation across
   split/seed coordinates, separation by profile/task/repeat, unknown-status
   fallback to queued, and malformed unhashable route identity fields. Remaining
   source-to-code review and controller branch combinations are still open;
   overall runner fidelity remains partial.

### Verification refresh: compatibility-state views (2026-10-22)

Re-read the complete attached ledger and proposed implementation plan before
continuing. Compared the source snapshot's `_summarize_group_states` and
`write_compat_state_views` behavior (`references/opf_adp_runner_suite/utils/opf_massive_suite_runner.py`)
with `runner/state.py` and `tests/test_runner_state_views.py`. The generic
implementation preserves the shared mechanics: sorted job-state discovery;
method/task aggregation boundaries; `launching` as running and unknown status
as queued; completed plus safe zero `last_exit_code` success; numeric-like
string ordering with the source's permissive single-decimal-point test and
job-ID tie break; first sorted successful state for `best_candidate_dir`; and
replay inclusion of persisted states outside the current plan. The current
contracts exercise malformed route fields, status/exit-code combinations,
ties, numeric-like strings, split/profile/repeat grouping, unplanned persisted
states, and no-success output. OPF case/head path fields remain adapted to the
generic provider/task/route identity. This bounded compatibility-view check
found no new mismatch and does not close the other open source-to-code review
items.

Fresh Linux verification on 2026-10-22: `.venv/bin/python -m pytest -q` —
**18,426 passed in 100.31 sec**. This is unit/integration suite evidence; it is
not a live OpenML run, host/GPU behavior certification, or proof of exhaustive
ledger/algorithm guide reconciliation. The Linux-only scope remains in force.

### Bounded source recheck: Linux orphan watchdog escalation (2026-10-22)

Compared `_orphan_watchdog.sh`'s recursive TERM/grace/KILL sequence with
`runner/watchdog.py`. The first pass terminates descendants before the launcher
root and waits the configured 15-second grace period. The source performs a
fresh recursive process-tree walk before KILL; the generic implementation had
been using the original descendant snapshot, which could miss a child spawned
while the parent handled TERM. Updated the Linux path to enumerate descendants
again after the grace interval and kill that current tree in reverse order,
then kill the root if it remains live. Added
`test_linux_watchdog_rescans_descendants_after_grace_before_kill` to pin the
late-child case and ordering. Focused verification:
`.venv/bin/python -m pytest -q tests/test_runner_watchdog.py
 tests/test_runner_launcher.py` — **41 passed in 0.90 sec**. This closes only
that bounded watchdog edge; the remaining ledger and ADP guide reconciliation
is still open.

Post-change verification: the first full run exposed the existing watchdog order
contract's now-stale expected sequence and the generated function map. Updated
the expected second tree enumeration and regenerated
`COORDINATED_PRESSURE_AWARE_SOURCE_FUNCTION_MAP.md`. Focused checks passed
(**3 passed in 1.70 sec**). The subsequent Linux full suite passed:
`.venv/bin/python -m pytest -q` — **18,427 passed in 100.86 sec**.

### Bounded source recheck: run planning, identities, and Linux runtime tuning (2026-10-22)

Compared `SuitePlan.build`, `derive_run_seed`, `manifest_compatibility_digest`,
and `SuitePlan.write_manifest` with ledger sections 2–3 and the copied source's
`build_suite_jobs`/manifest behavior. The generic planner retains deterministic
route enumeration, the asymmetric adaptive-versus-fixed repeat counts, the
ledger seed digest width/modulus/nonzero fallback, job slicing after ordered
route creation, signed manifest mismatch rejection, and the narrow interpreter
alias exception. Dataset/task/split and generic profile fields extend the job
identity for this repository's registered providers and routes; they do not
replace the shared scheduler mechanics. Focused contracts passed: **4 passed**
for route enumeration, repeat behavior, seed derivation, and manifest alias
compatibility.

Also compared the Linux runtime-tuning path with ledger sections 10 and 15 and
the copied `runtime_tuning.py`: affinity/frequency ordering, malformed and
rejected-affinity fallbacks, CPU partition and slot hashing, thread settings,
zero DataLoader workers, best-effort priority and `mlockall`, systemd scope
properties/version gate, and scope skip/fallback behavior are covered by Linux
mock contracts. Runtime tuning plus immutable-reference integrity checks passed
**38 tests in 0.76 sec**. These checks found no mismatch in these bounded
slices. They do not establish live host policy effects or close the broader
controller and ADP guide line-by-line audits.

### Verification refresh: generic search artifacts and full Linux suite (2026-10-22)

The guide's generic-search artifact path was checked against
`GenericSearchArtifactStore` and `GenericCandidateSearch`. Focused artifact
contracts passed (**3 passed in 1.74 sec**), including signed-signature
rejection, completed-result recovery, and continuation from a durable completed
candidate. The full Linux suite passed with captured output and exit status 0:
`.venv/bin/python -m pytest -q` — **18,430 passed in 100.44 sec**. The runner
function inventory check also passed. This updates test evidence only; ledger
source reconciliation and complete ADP guide line-by-line reconciliation
remain open, and no live OpenML campaign or host/GPU pressure certification
was run.

### Artifact recovery validation follow-up (2026-10-22)

The §8.11 candidate artifact recovery audit found that candidate checkpoint
paths lacked the task checkpoint's containment checks and the manifest schema
was not validated. Recovery now rejects paths escaping the task/candidate
directory (including symlink targets), mismatched candidate/context identity,
and stale artifact schema; the artifact format was bumped to schema 2. Focused
ADP/core verification passed (**548 passed in 9.69 sec**). The full Linux suite
passed with captured output and exit code 0: **18,433 passed in 99.78 sec**.
The generated runner function inventory check also passed. This is a bounded
checkpoint/recovery improvement; the remaining ledger-to-code review is still
open, and no live dataset campaign was run.

The later paused-PID recovery correction was checked in the complete Linux
suite: `.venv/bin/python -m pytest -q` — **18,434 passed in 100.13 sec**.
The first post-change run had 18,433 passes and only the expected stale
generated-function-map failure; regeneration, `--check`, and the map test
passed before the green rerun. No OpenML training campaign was run.

### Bounded source recheck: Linux bundles and child telemetry (2026-10-22)

Compared `runner/bundles.py` and `runner/telemetry.py` with ledger sections 11
and 21 and the copied bundle/telemetry sources. Verified the shared behavior
for direct manifest writing, level-1 gzip archives, 64 MiB chunk boundaries,
telemetry CSV filtering, nonempty restore refusal, safe member validation,
multipart reassembly residue on archive failure, sharded telemetry path and
optional checksum handling, progress-sidecar precedence/fallback, stage-state
RNG companions, and the CSV fallback limitation when the JSON stage sidecar is
absent. Field names and repository roots are genericized. The associated
focused Linux contracts passed: **48 passed in 1.76 sec**. This bounded review
found no additional discrepancy; it does not close full orchestration or
algorithm-guide reconciliation.

### Recovery correction: paused records cannot trust a saved PID (2026-10-22)

The ledger's restart sequence first normalizes a paused job by clearing its PID,
exit code, and end time and persisting that state; only afterward does runtime
reconstruction search for a live process with the exact saved command. The
generic `reconcile_runtime` path instead trusted any live saved PID before
checking paused status. That could attach an unrelated process after PID reuse.
Changed paused recovery to persist the normalized paused record first, clear the
saved PID, then attach only an exact-command match. Added a per-job-only state
write for this recovery path and launch-state normalization, avoiding premature
compatibility-view rebuilds; ordinary lifecycle writes still refresh those
views. The same recheck found that an accepted live saved PID on any other
nonterminal status must derive as running; generic recovery returned it in the
active list but retained a stale failed/queued status. It now sets running,
while completed success and terminal exact-resume failure remain earlier
special cases. Regression coverage checks paused PID reuse, exact-command
reattachment, state-only normalization, and live-PID status override. Focused
verification passed: **14 process tests**, **13 recovery/reconciliation
contracts**, and the direct selections (**8 passed**). After both corrections,
the regenerated function map passed and the final Linux full suite passed with
**18,436 tests in 101.22 sec**. The complete source-to-code and ADP guide audits
remain open.

### Bounded source recheck: pressure-resume handoff and signal shutdown (2026-10-22)

Compared the pressure-resume launch context and Linux signal path with ledger
section 8 and the exact controller code. The resume context omitted the
recorded `pause_reason` and `resume_device`; it now carries both alongside the
checkpoint paths. The signal handler previously only set a flag, delaying
process-tree termination and resumable-state writes until the next outer-loop
check. It now runs shutdown immediately and exits with status 130, matching the
source handler; `KeyboardInterrupt` continues through the same shutdown path.
Focused verification passed (**7 passed**) for pressure-resume handoff, all four
Linux shutdown signals (with a live child terminated and persisted immediately
inside the handler), `KeyboardInterrupt`, and the intentional no-cleanup
behavior for unrelated controller exceptions. The generated function map was
regenerated and its `--check` passed. The complete Linux suite has not yet been
rerun after these changes; broader ledger reconciliation and the ADP guide
line-by-line audit remain open.

The same shutdown recheck found a narrower persistence exception boundary:
per-job interruption-state write failures propagate after process termination,
but failure to save the controller/backoff record is swallowed. `_shutdown`
now follows that order and boundary. Focused failure-path verification passed
(**7 passed** across resume handoff, signal handling, and both shutdown write
failure cases); the function inventory remains current. Full-suite verification
is still pending.

### Linux full-suite verification after shutdown fidelity corrections (2026-10-22)

After adding the source-compatible resume context fields, immediate signal
shutdown, and best-effort controller-state persistence boundary, the complete
Linux suite passed: `.venv/bin/python -m pytest -q` — **18,437 passed in
101.42 sec**. `scripts/generate_runner_function_map.py --check` also passed.
This is regression evidence only; it does not complete the remaining
source-to-code audit or the separate ADP guide reconciliation.

### Bounded source recheck: terminal-failure gate release order (2026-10-22)

Section 8's non-retryable failure path finalizes and persists the failed job
before releasing its device-mode gates; retryable failures release gates before
writing retry state and appending the job to the queue. The generic controller
previously released gates before both branches. It now follows the distinct
ordering. A focused contract observes gate state during terminal finalization
and checks retry behavior remains unchanged. The full Linux suite then passed:
`.venv/bin/python -m pytest -q` — **18,438 passed in 100.87 sec**. The first
post-edit run reported 18,437 passing tests and only a stale generated function
map; after regeneration, the focused function-map and changed-path checks
passed, and this clean full run passed. The map `--check` is current. Broader
ledger reconciliation and the ADP guide line-by-line audit remain open.

### Bounded source recheck: standalone Linux pressure wrapper (2026-10-22)

Compared ledger section 9 and the copied `pressure_aware_wrapper.py` with
`runner/pressure_wrapper.py`. The Linux path retains strict `>` pressure
checks, SIGSTOP/SIGCONT over the child's process group, process-name-wide
Python memory subtraction without clamping, both-domain resume readiness by
threshold-or-drop, telemetry-only swap, the zero-pressure fallback when GPU
sampling fails, paused-child waiting without polling until gates reopen, and
replacement command launch if the stopped child exited while waiting. No
behavioral mismatch was identified in this bounded comparison. Focused Linux
contracts passed: `.venv/bin/python -m pytest -q tests/test_runner_pressure.py`
— **24 passed in 0.41 sec**. This does not cover every interruption timing or
host-level signal outcome and does not close the wider ledger audit.

### Bounded source recheck: detached Linux watchdog polling order (2026-10-22)

Compared ledger section 8's Bash watchdog with `runner/watchdog.py`. The source
waits 2 seconds after startup, then sleeps 3 seconds before each master-liveness
check. The generic watcher previously checked immediately after its 2-second
delay. Updated it to use the documented 2-second initial wait followed by the
3-second pre-check cadence. A contract now observes two checks (the first while
the master is alive) and pins the sleep/check ordering. Focused verification
passed: `.venv/bin/python -m pytest -q tests/test_runner_watchdog.py
tests/test_runner_launcher.py -k watchdog` — **2 passed**. Broader watchdog
termination and controller audits remain open.

### Bounded source recheck: startup process reconciliation (2026-10-22)

Compared ledger section 7's `_reconcile_stale_job_state`,
`load_existing_runtime`, and recovered-child exit handling with
`runner/processes.py::reconcile_runtime`, `find_exact_argv`, and
`SuiteCoordinator._build_runtimes`. The generic path preserves exact normalized
argv matching, earliest `(create_time, pid)` selection without cwd validation,
live saved-PID trust for nonpaused active records, paused-record PID clearing
before the second exact-command lookup, terminal exact-resume handling,
completed-zero-exit skipping, stale active-record queueing, and unknown-exit
tail requeue without incrementing retry count. The generic state calls the
terminal reason `last_error`; reconciliation also accepts the source field
`reason` so legacy records map to the same terminal behavior. Focused Linux
verification passed: `.venv/bin/python -m pytest -q
tests/test_runner_processes.py tests/test_runner_contracts.py -k
'reconcile or recovery or recovered or paused_record or exact_argv or saved_pid'`
— **21 passed, 17,573 deselected**. No discrepancy was found in this bounded
slice. This does not close the remaining ledger reconciliation or the separate
ADP guide line-by-line audit.

### Verification refresh after watchdog cadence and recovery audit (2026-10-22)

The watchdog cadence change exposed two stale event-order assertions: one
expected an immediate first liveness check, and the mocked Windows branch
expected one fewer poll. Updated those assertions to include the documented
pre-check polling sleep. The focused watchdog contracts passed (**2 passed,
17,578 deselected**), the runner function map check passed, and the full Linux
suite passed: `.venv/bin/python -m pytest -q` — **18,438 passed in 100.93 sec**.
No dataset benchmark or OpenML training campaign was run. This refresh confirms
the current test suite; it does not close the remaining source-to-code ledger
comparison or the separate ADP guide line-by-line audit.

### Recorded ledger/source conflict: emergency kill order (2026-10-22)

The ledger's §22 contract says the emergency utility kills descendants in
reverse enumeration order and then the owned root. The copied reference
`scripts/kill_all_runners.py` instead builds `descendants + [proc]` and reverses
that entire list, which makes the root the first kill target. The generic
`process_cleanup.kill_owned_processes` and its contract follow the user-provided
ledger order (descendants first, root last). I have retained that behavior
because the ledger and proposed plan define the implementation contract; the
copied reference remains unchanged. This is an explicit source-versus-ledger
divergence, not a verified source parity claim. The existing Linux cleanup
contracts pin reverse descendant order, PID deduplication, and root-last order;
these passed in the full suite (**3 cleanup-order/race contracts**, plus
ownership tests).

### Reference snapshot isolation check (2026-10-22)

Searched `src/`, `scripts/`, `pyproject.toml`, and `README.md` for imports or
runtime references to `opf_adp_runner_suite`, `opf_massive_suite_runner`, and
`opf_suite_launcher`; none were found. The reference-integrity contract
validated all 46 manifest-listed copied files and rejected unlisted snapshot
files. The snapshot-integrity and controller end-to-end contracts passed
(**2 passed in 7.22 sec**). This establishes that the coordinator uses its
generic package instead of depending on the copied scheduler at runtime; it
does not complete the behavior-by-behavior ledger audit.

### Linux launcher systemd re-exec correction (2026-10-22)

Running `bash scripts/simple-adp-suite.sh --help` revealed that the outer
launcher failed after entering the configured user systemd scope: Python's
`-m` invocation had become a source-file path in `sys.argv[0]`, and the helper
re-executed that path as a script, breaking package-relative imports. Updated
`maybe_reexec_systemd_scope` to preserve the original interpreter invocation
from `sys.orig_argv` when it is available, retaining flags such as `-u -m`;
explicit invocation arguments used by callers/tests keep their existing
behavior. Added a systemd re-exec regression contract. Focused verification
passed (**19 systemd-scope contracts**), `bash -n scripts/simple-adp-suite.sh`
passed, and the actual shell `--help` path now prints CLI help successfully.
The runner function map was regenerated and its `--check` passed. The full
Linux suite then passed after this source change: `.venv/bin/python -m pytest
-q` — **18,439 passed in 101.63 sec**.

### Full-suite verification refresh (2026-10-22)

Re-ran the complete Linux test suite from the current checkout after rereading
the attached ledger and implementation plan. `.venv/bin/python -m pytest -q`
passed: **18,439 passed in 101.42 sec**. This confirms the current regression
suite remains green; it does not close the line-by-line ledger reconciliation,
the full canonical ADP guide audit, or untested runtime/hardware behavior.

### Bounded recheck: child checkpoint mirror handoff (2026-10-22)

Compared ledger sections 7 and 20–21 with `launch_job`, the worker callback,
`TabularADP.persist`, `StageTrainer`, and `ChildTelemetryWriter`. The coordinator
already supplied last/best/best-batch paths on pressure resume, and the
telemetry writer exposed local/mirror methods, but the estimator callback did
not invoke them. `launch_job` now exports the three coordinator job paths to
the worker; after each signed estimator checkpoint save, the worker callback
mirrors the full per-batch last record locally and to `checkpoint_last.pt`.
At stage boundaries it also writes the best-validation snapshot locally and
to `checkpoint_best.pt`. At each fresh architecture stage it initializes
`checkpoint_best_batch.pt` from the stage's initial model snapshot with
`best_batch_loss=inf`. The signed estimator checkpoint remains authoritative
for exact resume. No later best-batch update is made because the canonical guide
and copied snapshot do not define its selection/update rule. Focused
verification passed for interruption/resume, env-path handoff, and
byte-identical local/coordinator last, best, and initial best-batch mirrors.
This implements the defined initial best-batch state and last/best mirror
behavior; the wider ledger and guide reconciliations remain open.

After the mirror implementation and source-map regeneration, the complete
Linux suite passed: `.venv/bin/python -m pytest -q` — **18,439 passed in
101.56 sec**; `scripts/generate_runner_function_map.py --check` passed. The
follow-up mirror-content assertions passed in the focused integration rerun
(**2 passed in 1.72 sec**). No dataset suite or OpenML training campaign was
run.

### Best-batch initialization fidelity follow-up (2026-10-22)

Section 20 specifies that a fresh stage initializes `best_batch_snap` from its
initial architecture snapshot and sets `best_batch_loss=inf`; it does not
specify a later replacement criterion. `ChildTelemetryWriter` now writes and
mirrors that initial best-batch record at each new architecture stage, while
leaving it unchanged afterward. The telemetry integration test verifies the
initial snapshot marker, infinite loss, local/coordinator byte identity, and
continuity across same-stage interruption/resume. Focused tests passed (**3
passed in 1.93 sec**, followed by **2 passed in 1.71 sec** after the content
assertions). The complete Linux suite then passed: `.venv/bin/python -m pytest
-q` — **18,439 passed in 101.91 sec**. The source function map check passed.
The unspecified post-initial best-batch update rule and the wider source/guide
audits remain open.

### Linux launcher interruption cleanup and controller-cycle recheck (2026-10-22)

Compared ledger §1's outer supervisor cleanup and §14's ordered scheduling
pass with `runner/launcher.py::_run_with_watchdog` and
`runner/scheduler.py::SuiteCoordinator.run` / `_sample_snapshot`. The launcher
now terminates the live controller process tree if watchdog startup fails or
the controller wait is interrupted, then always stops the watchdog it managed
to start. Focused tests cover normal controller completion, interrupted wait,
and watchdog startup failure. The bounded §14 source comparison confirmed the
implemented order: status and live concurrency, resource snapshot, suspended
resume, active child reap, pressure victim, then FIFO admission; successful
admission launches at most one child in that pass. The existing
`test_controller_pass_orders_resume_then_reap_then_pressure_then_new_admission`
checks the central transition ordering, while the broader §14 branch-by-branch
ledger audit remains open.

Verification on Linux: `.venv/bin/python -m pytest -q tests/test_runner_launcher.py
-k watchdog` — **3 passed**; the runner function map was regenerated and
`--check` passed; `.venv/bin/python -m pytest -q` — **18,441 passed in 100.83
sec**. No dataset benchmark or OpenML training campaign was run. This bounded
recheck does not close the full source-to-code ledger reconciliation or the
complete canonical ADP guide audit.

### Custom generic mutation resume coverage (2026-10-22)

The separate [ADP algorithm audit](ADP_ALGORITHM_IMPLEMENTATION_AUDIT.md) now
records bounded custom width- and depth-mutation resume contracts under both
generic profiles. The tests interrupt active candidates, resume their
checkpoints without reapplying the callbacks, and restore a custom MAE task
adapter. They compare mutation order, history, controller state, and predictions
with uninterrupted training. The
combined focused tests passed (**6 passed**) and the full Linux suite passed
(**18,447 passed in 101.79 sec**). The complete ADP guide audit and broader ledger audit
remain open.

### Linux runtime-tuning ledger recheck (2026-10-22)

Compared ledger §§6, 10, and 15 with `runner/runtime_tuning.py`,
`runner/processes.py::launch_job`, and the Linux child-launch path. The bounded
comparison covered affinity detection and frequency ordering, CPU-list/range
parsing, stable SHA-1 slot selection and partitioning, explicit/environment
concurrency precedence, thread-budget calculation, worker count zero, child
thread environment, CPU/GPU masking, POSIX `nice`/`ionice`/`SCHED_BATCH`,
best-effort Linux `mlockall(MCL_CURRENT | MCL_FUTURE)`, and the user systemd
scope properties including the systemd-253 `MemoryZSwapMax=0` gate. The
coordinator passes the large unbounded launch sentinel to the child helper,
then writes `TABULAR_CPU_JOB_CONCURRENCY=1` into the launched environment for
that sentinel while preserving the helper's one-thread budget. No discrepancy
was identified in this Linux slice; Windows mechanics are outside the active
scope.

Focused Linux verification: `.venv/bin/python -m pytest -q
tests/test_runner_tuning.py tests/test_runner_processes.py` — **52 passed in
0.94 sec**. This covers the listed runtime-tuning and child-launch contracts,
not actual permission to raise priority, lock resident memory, enter a systemd
scope, or bind hardware CPUs. Other ledger sections and the complete ADP guide
reconciliation remain open.

### Current full-suite refresh after paragraph-level ADP audit (2026-10-22)

After the bounded runtime-tuning and ADP controller/checkpoint audit slices,
the complete Linux test suite passed: `.venv/bin/python -m pytest -q` —
**18,447 passed in 102.36 sec**. This confirms current regression-suite health;
it does not complete the remaining source-to-ledger crosswalk, certify live
host/systemd/GPU behavior, or replace a dataset campaign.

The passing suite includes `tests/test_smoke_suite.py`, which runs every
registered canonical model/profile against both synthetic classification and
regression tasks for one epoch with `batch_size == len(training_partition)`;
the fixed model receives the full 50-architecture grid. It also checks
custom-task smoke routes. `tests/test_runner_end_to_end.py` exercises actual
child-process smoke execution through a registered plugin provider. These are
fixture-backed smoke contracts, not training on the configured OpenML pools;
the latter remains unlaunched by instruction.

### Bounded source recheck: process reconciliation and compatibility state (2026-10-22)

Compared ledger §§7 and 18 with `runner/processes.py::reconcile_runtime`,
`find_exact_argv`, `runner/state.py::JobRuntime.derived_status`,
`JobStateStore._summarize_compatibility_groups`, and
`seed_run_root_from_source`. Confirmed the recorded recovery quirks: ordinary
live saved PIDs are accepted without argv/cwd validation; stale PIDs fall back
to exact normalized argv and choose the earliest `(create_time, pid)` match;
paused records first clear and persist the saved PID, then may attach a still
live exact-argv process as running; disappeared recovered processes retain no
invented exit code and are requeued as unknown exits; exact-resume failures
remain terminal across invocation startup. Derived status keeps live-process
priority over paused state and preserves the documented retained-exit behavior.
Compatibility views scan sorted persisted job states, map `launching` to
running and unknown statuses to queued, require completed plus zero
`last_exit_code`, sort latest completions with the permissive one-decimal
numeric-like key and job-ID tie-break, and select the first sorted success for
`best_candidate_dir`. Source-run-root seeding copies the named root artifacts
first, then remaining files, skips existing destinations, and performs no
checksum or merge. These details match the ledger in this bounded slice; no
code correction was indicated.

Focused Linux verification:
`.venv/bin/python -m pytest -q tests/test_runner_processes.py
tests/test_runner_state_views.py tests/test_runner_contracts.py -k
'reconcil or recovery or exact_argv or compat or candidate_state or
paused_pid or saved_pid or source_run_root'` — **33 passed, 17,589
deselected in 1.72 sec**. This closes only this source slice; the broader
ledger-to-code reconciliation and final ADP guide crosswalk remain open.

### Bounded source recheck: bundle and telemetry-shard handoff (2026-10-22)

Compared ledger §11 with the immutable
`references/opf_adp_runner_suite/utils/opf_resume_bundle.py` and its generic
port in `runner/bundles.py`. The implementation retains the source ordering
and behavior: direct manifest write before deleting/replacing old archives;
level-1 gzip tar; telemetry CSV member filter; inclusive 64 MiB monolithic
boundary and numbered parts above it; nonempty-results restore guard; lexically
ordered part concatenation into `.restore`; normalized path, expected-root,
containment and link checks before extraction; successful restore cleanup
versus failed-extraction residue; and shard lookup/filter,
repository-contained target, manifest-directory-contained parts, optional
SHA-256 verification, and temporary-file retention on non-checksum write
errors. This focused comparison found no mismatch. The generic repository-root
parameter is the planned adaptation of the source module constant.

Focused Linux verification: `.venv/bin/python -m pytest -q
tests/test_runner_bundles.py tests/test_runner_launcher.py -k
'bundle or restore or rehydrat'` — **23 passed, 39 deselected in 0.90 sec**.
This does not close remaining launcher/controller, resource-accounting,
telemetry, or ADP algorithm crosswalk work.

### Bounded source recheck: defaults, validation holes, and job seed (2026-10-22)

Compared ledger §§2–3 with `runner/config.py::RunnerSettings`,
`parse_epoch_limit`, `resolve_concurrency`, and the plan's repeat/seed
construction. The generic runner preserves the listed Linux-targeted defaults:
unbounded concurrency sentinel `10**9`, disabled zero active caps, repeat and
retry defaults `5/1`, host stop/resume `1024/1280 MiB`, GPU free stop/resume
`100/500 MiB`, reserves `0/0 MiB`, child buffers `500/100 MiB`, external drop
`500 MiB`, poll/hold/backoff `0.5/30/0.5`, device `0`, CPU-only false,
telemetry-only swap `70/70`, architecture limits `512/5/1_000_000_000`,
unlimited epoch zero, seed `42`, and the fixed vanilla grids. The concurrency
file is reread per resolution; blank uses configured concurrency while unreadable,
malformed, and negative contents fail. Validation retains the ledger's named
holes for seed sign, pressure-stop signs, swap percentages, and job slicing.
The plan preserves one-or-more effective repeats and the ADP/vanilla repeat
asymmetry, while the SHA-256-derived seed uses the recorded byte width, modulus,
and zero fallback. No mismatch was found in this bounded settings slice.

Focused Linux verification: `.venv/bin/python -m pytest -q
tests/test_runner_contracts.py -k 'ledger_defaults or shared_defaults or
asymmetric_repeats or run_seed_uses_ledger or concurrency_control_file or
rereads_concurrency or lowering_live_concurrency or
missing_present_task_repeats'` — **9 passed, 17,577 deselected in 2.14 sec**.
Remaining work includes the source-level launcher and controller interactions,
resource accounting, telemetry/checkpoint handoff edges, and full ADP guide
crosswalk.

### Bounded source recheck: Linux outer-launcher handoff (2026-10-22)

Compared ledger §1 with `runner/launcher.py::main`, `_settings`, `_detach`,
and `_run_with_watchdog`. Configuration precedence is environment, settings
JSON, repeated `--set`, then explicit CLI values; explicit zero survives and
smoke/dry-run flags apply their documented overrides. The generic path creates
the nested run root before bundle restore, writes the launch manifest, handles
report/resume without scheduling new children, persists a new signed plan
before controller execution, and pairs restore/export behind the bundle
setting. Export runs in `finally` and its errors do not replace the controller
return code. Detach preserves the configured-run-root exception, fsynced PID
write, and surviving child if PID persistence fails. The controller starts
before the watchdog; interruption or watchdog startup failure terminates its
process tree, and the watcher is stopped after child completion. These Linux
handoff behaviors match the generic ledger contract in this bounded slice; the
OPF case preparation and Windows wrappers remain outside this Linux task
scope.

Focused Linux verification: `.venv/bin/python -m pytest -q
tests/test_runner_launcher.py -k 'preserves_nested_run_root_restore_guard or
settings_precedence or explicit_resume_reuses or report_mode or
detach_uses or watchdog_launcher or watchdog_start_failure'` — **8 passed,
34 deselected in 0.89 sec**. The controller's full pass order and resource
accounting still require their own source-level reconciliation.

### Bounded source recheck: central controller cycle (2026-10-22)

Compared ledger §14's invocation setup and per-pass sequence with
`SuiteCoordinator.run`, `_sample_snapshot`, `_resume_ready`, `_resume`,
`_handle_exit`, and `_suspend_pressure_victim`. The code order is manifest and
metadata, dry-run exit, transient controller reset, runtime reconciliation,
then status/concurrency read, resource sample and capacity, latest-child and
external-process measurements, pre-gate snapshot, gate update/reopen, all
ready resumes, child reaping, at most one hard-pressure victim, and FIFO queue
admission. Confirmed source quirks include capacity's separate live
concurrency rereads, active-RSS diagnostics without subtracting from OS
available RAM, the general-latest sample preceding the unreachable latest-GPU
fallback, same-sample resume bypass of admission limits, unknown-exit recovery
to the queue tail without a fabricated failure, OOM peer selection by
architecture priority, pressure victim selection by sampled footprint, and
the inner retry-on-launch-error path. No mismatch was identified in the
examined controller paths. The live-host hardware response remains unverified.

Focused Linux verification: four direct full-pass contracts covering ordering,
resume admission bypass, unknown-exit FIFO recovery, and combined OOM/pressure
handling passed (**4 passed in 1.29 sec**). The broader controller cross-branch
matrix and the runner's remaining subsystem audits are still open.

### Bounded source recheck: host, GPU, and child resource sampling (2026-10-22)

Compared ledger §4 with `runner/resources.py` and the coordinator sampling
path. Confirmed whole-MiB floor conversion for host/swap bytes; the exact
`nvidia-smi` argument vectors and no-timeout behavior; GPU free-memory clamp
with unclamped over-100% used percentage; unavailable main GPU sample versus
empty GPU-process sample on sensor errors; root-PID retention on process-tree
inspection failure; all-or-zero active-RSS sum on any lookup failure; best
effort per-process child RSS; GPU child aggregation only for GPU jobs; cached
process identity and Python-name/command filtering; owned-PID exclusion; and
the exception boundary where individual process inspection failures are
skipped but top-level or mid-iteration `process_iter` failures propagate.
External-drop reopening requires both sufficient adjacent-sample decrease and
disappearance of a previously tracked PID. These source details match this
bounded Linux slice; no change was indicated.

Focused Linux verification: `.venv/bin/python -m pytest -q
tests/test_runner_resources.py` — **27 passed in 0.76 sec**. This is synthetic
and mocked sampler evidence, not a live GPU/host-pressure certification.
Remaining source reconciliation includes the standalone pressure controller,
telemetry/checkpoint edge cases, and full ADP guide crosswalk.

### Bounded source recheck: retry classification and Linux shutdown (2026-10-22)

Compared ledger §8 with `SuiteCoordinator._latest_attempt_text`,
`_classify_failure`, `_handle_exit`, `_shutdown`, and the copied runner's
`_last_job_attempt_log`, OOM classifiers, retry predicate, and
`_shutdown_all_runtimes`. The attempt reader retains backward 64 KiB chunks,
latest-launch marker selection, 200,000-byte fallback, UTF-8 replacement, and
last-4,000-character OOM windows; exact-resume markers are searched over the
whole latest attempt. Classification order keeps exact-resume terminal and
GPU OOM distinct from memory OOM; retry zero remains uncapped, OOM reduces the
launch budget with a `1e-6` floor, non-OOM retry clears backoff, and retries
return to FIFO tail. Linux signal shutdown terminates owned trees, persists
active as queued and suspended as paused, retains paused causes, records the
interruption, and treats controller-state persistence as best-effort after
per-job state writes.

This comparison found and corrected one persisted-state mismatch: active
records now write `pause_reason: ""` on shutdown, matching the source instead
of writing JSON `null`. Added an assertion for both the in-memory value and
persisted record. Focused Linux verification across attempt scanning,
classification, shutdown failure boundaries, signal handling, and KeyboardInterrupt
passed (**14 passed in 1.53 sec**). Arbitrary controller exceptions still do
not invoke signal shutdown, matching the recorded source boundary. Broader
runner reconciliation and the final ADP guide crosswalk remain open.

### Bounded source recheck: admission and standalone pressure wrapper (2026-10-22)

Compared ledger §§5 and 9 with `runner/admission.py` and
`runner/pressure_wrapper.py`, including the POSIX paths in the copied source.
Admission preserves zero-as-unbounded concurrency, minimum-one reserve capacity,
strict `<` hard-pressure thresholds, strict `>` child-buffer gate closure,
matching external-drop PID disappearance, host-gate blocking of both lanes,
GPU-gate CPU fallback, GPU selection for all labels when both lanes are ready (with `pressure_aware` additionally requiring CPU-lane readiness),
and the separate matching-lane resume threshold after the broader reopen
predicate. The standalone wrapper remains a separate process-group SIGSTOP /
SIGCONT controller: it uses percentage pressure with strict `>`, treats swap
as telemetry, estimates external use by global Python process names, requires
both domains to resume by threshold or external drop, does not poll the child
while waiting for reopening, and relaunches if that stopped child exited.
The paused-path sensor fallback and pause-boundary process checks also match
the source. No Linux behavior discrepancy was identified in this slice.

Focused Linux verification: `.venv/bin/python -m pytest -q
tests/test_runner_pressure.py
tests/test_runner_contracts.py::test_admission_capacity_strict_pressure_gates_and_resume_predicates
tests/test_runner_contracts.py::test_gpu_admission_also_requires_host_resume_and_labels_match_source
tests/test_runner_contracts.py::test_reserve_capacity_preserves_python_float_floor_division_boundaries
tests/test_runner_contracts.py::test_soft_gate_updater_exhaustive_pressure_lane_sensor_and_buffer_product`
— **30 passed in 1.56 sec**. This does not certify live memory pressure or
real hardware behavior; the rest of the ledger and final ADP guide crosswalk
remain open.

### Verification refresh: stage-sidecar audit and full Linux suite (2026-10-22)

Re-read the user-provided coordinated-runner ledger and implementation plan, then
reconciled ledger §21's child telemetry/stage-sidecar rules with
`runner/telemetry.py`, coordinator checkpoint reconciliation, the copied
telemetry source, and focused contracts. The JSON sidecar remains authoritative;
CSV recovery remains observation-only and cannot establish exact active-stage
recovery because its current schema omits the required checkpoint/RNG/architecture
fields. Corrupt sidecars, mismatched identity/checkpoint paths, invalid RNG, and
unrelated architecture changes fail closed; the documented same-stage monotonic
width catch-up is bounded by controller stage and tensor-layout checks. No new
mismatch was found. Focused verification passed: `pytest -q
 tests/test_runner_child_telemetry.py tests/test_runner_bundles.py` — **48 passed
in 1.78 sec**.

### Bounded source recheck: exact GPU process-row parsing (2026-10-22)

Recompared ledger §4's GPU process-memory helpers with
`runner/resources.py`. This found two source-compatibility gaps: the generic
row parser required exactly three comma-separated cells instead of accepting
the source's first three cells when extra cells are present, and the external
GPU Python-name filter did not trim whitespace before applying the `python`
substring / `py` prefix checks. Updated `_gpu_process_rows` to use the source's
split-and-first-three behavior, skip malformed records, and contain query,
decode, and iteration errors at the sampler boundary; external GPU filtering
now shares the normalized predicate. Added regression cases for trailing CSV
fields, malformed decoding, and padded Python process names. Focused verification
passed: `.venv/bin/python -m pytest -q tests/test_runner_resources.py
tests/test_runner_contracts.py -k 'gpu_process or external_gpu or
memory_query_exception or external_memory_sampler'` — **8 passed, 17,606
deselected in 1.61 sec**. This closes those two parse/filter mismatches only;
resource sampling and live pressure remain broader audit/runtime boundaries.

### Audit note superseding an earlier remaining-work line (2026-10-22)

The statement above that standalone pressure, telemetry/checkpoint, and the
ADP crosswalk remained unaudited predates later sections in this file and the
generated ADP block crosswalk. The standalone Linux wrapper, telemetry and
checkpoint sidecar, and indexed ADP guide reviews are recorded in the later
rechecks; do not treat that earlier sentence as current status. The full
ledger-to-code reconciliation remains active.

The subsequent full Linux suite exposed only stale generated function-map output:
**18,446 passed, 1 failed** (`test_checked_in_runner_function_map_matches_current_ast`). Regenerated
`docs/COORDINATED_PRESSURE_AWARE_SOURCE_FUNCTION_MAP.md`; its `--check` and
focused test passed. A clean full rerun then passed: `pytest -q` — **18,449
passed in 101.25 sec**, including the generated ADP block crosswalk freshness
test. The focused provider-route and smoke contracts also passed (**50 passed
in 2.47 sec**). Those smoke contracts
exercise every registered model/profile against classification and regression
fixtures for one epoch with full-training-partition batches, and plan every
listed task in each OpenML suite using fixtures. No live OpenML download or
campaign was launched. This verifies the current unit/integration suite and
fixture-backed worker paths, not live GPU pressure behavior or dataset-specific
accuracy. Full ledger-to-code closure remains active; the ADP block index is
now generated and freshness-checked.

### Bounded source recheck: backend admission and logging context (2026-10-22)

Compared ledger §§12 and 24 with `runner/backend.py` and
`runner/logging_context.py`. Torch probing requires successful allocation,
fill, and synchronization before accepting CUDA; expected Torch probe failures
fall through to the independently validated CuPy probe. CPU/GPU admission
conflicts fail, accelerator state is bound to the worker environment, optional
cuDF/cuML hooks preserve their import-order checks, and array conversion
rejects accelerator values in CPU mode. Generic `TABULAR_*` and
`SIMPLE_ADP_*` environment aliases extend the OPF flags for this package.
Seed precedence and Python/NumPy/Torch/CuPy seeding follow the ledger while
using the generic suite's explicit seed handoff. Logging samples default host
and GPU context before applying the coordinator overlay, preserves existing
stream handlers, and disables propagation only for named loggers. Generic field
names and formatting replace OPF-prefixed names. No discrepancy was found in
this bounded Linux comparison. Focused tests passed:
`.venv/bin/python -m pytest -q tests/test_runner_backend.py
 tests/test_runner_logging_context.py` — **29 passed in 2.01 sec**. GPU probes
were mocked; no live CUDA device was exercised.


### Full Linux suite refresh after backend/logging review (2026-10-22)

After the backend and logging source comparison, `.venv/bin/python -m pytest
-q` completed with exit code 0: **18,449 passed in 107.29 sec**. This includes
all currently checked-in runner, ADP, provider, and fixture-backed smoke
contracts. It does not establish live CUDA, live pressure-control, or
real-dataset campaign behavior, and the remaining behavior-by-behavior ledger
reconciliation remains active.

### Bounded source differential: pressure-resume readiness (2026-10-23)

Extended the source differential to compare the reference and generic
`pressure_pause_reopened` predicates across **34,992** combinations of pause
reason, CPU/GPU device, host resume availability, GPU free memory (including
missing sensor), host/GPU used and prior peak values, and external memory
drops. The matrix includes strict-below and exact-equality boundaries for the
500 MiB external-drop threshold and each configured resume threshold. No
mismatch was found. The differential and source-snapshot integrity contracts
passed (**2 passed**), and the full Linux suite passed:
`.venv/bin/python -m pytest -q` — **18,451 passed in 106.89 sec**. This closes
the predicate comparison for these finite inputs; it does not establish every
controller interleaving or live resource-sensor behavior.

### Recovery persistence regression refresh (2026-10-22)

Extended the legacy recovered-lane regression to start from a `launching`
record with absent, null, or empty lane/pause fields and assert the normalized
`running` state is persisted with `device_mode="gpu"` and `pause_reason=""`.
The focused recovery tests passed (**2 passed**); the source function map and
ADP block crosswalk freshness checks passed. A full Linux rerun passed:
`.venv/bin/python -m pytest -q` — **18,450 passed in 105.99 sec**. This closes
the persisted-state assertion for that recovery edge; the overall runner audit
remains open as described above.

### Bounded source differential: failure classification (2026-10-23)

Added `tests/test_runner_source_fidelity.py`, which imports the immutable
reference runner in a bytecode-disabled subprocess and compares its
latest-attempt, OOM, and exact-resume classification helpers with the generic
coordinator. The 16 cases cover every documented marker family, overlapping
CUDA/memory markers, the backward scan's 64 KiB boundary, and an OOM marker in
an older attempt followed by a non-OOM latest attempt. The differential and
reference-snapshot integrity contracts passed (**2 passed**). The first
full-suite run exposed generated `__pycache__` files in the copied snapshot;
those generated files were removed, and the differential subprocess now uses
`python -B` so the immutable snapshot remains byte-for-byte clean. The clean
full Linux suite passed: `.venv/bin/python -m pytest -q` — **18,451 passed in
112.16 sec**. This closes only the failure-classifier source comparison; the
remaining ledger review and non-live hardware boundaries remain open.

### Bounded source differential: selector and CPU-only cap branch (2026-10-23)

Extended the source-fidelity contract to compare `select_launch_device` with
the generic `choose_device` across **256** combinations of CPU-only and parent
CPU masks, host-resume equality, total active cap, current active count, host
gate, lane enable, and CPU hold. This exposed a helper-level mismatch: the
source CPU-only/parent-mask branch returns after the CPU-admission and
host-resume checks, before evaluating the total active cap. The generic helper
had applied the total cap inside its shared CPU eligibility predicate. Updated
the helper to preserve the source early return; the coordinator's earlier
launch-budget check continues to enforce the total cap for actual queue
admission. The differential also compares 16 failure-classification cases.
Focused differential, selector-oracle, and admission edge tests passed (**3
passed**). Regenerated the line-number-sensitive function map and verified it
and the ADP crosswalk. The clean full Linux suite passed:
`.venv/bin/python -m pytest -q` — **18,451 passed in 105.69 sec**. This closes
the selector/helper comparison for these enumerated values; controller
cross-branch, arbitrary numeric, and live hardware behavior remain open.

The same source-differential contract now compares the launch-capacity helper
over **5,760** combinations of concurrency (including the unbounded sentinel),
host/GPU availability, absent GPU samples, host/GPU reserve values, and the
total active cap. Capacity matched in every case. The final full Linux rerun
passed: `.venv/bin/python -m pytest -q` — **18,451 passed in 104.60 sec**;
the source snapshot integrity contract passed in that suite. These finite
products do not cover arbitrary floats or every controller interleaving.

### Bounded source recheck: recovered lane and pause defaults (2026-10-22)

Compared runner startup recovery with the reference `JobRuntime` defaults,
`load_existing_runtime`, and stale-job reconciliation. Legacy records with a
missing or falsey `device_mode` must recover into the `gpu` lane, and a missing
or falsey `pause_reason` must normalize to the empty string. The generic
`reconcile_runtime` now applies these defaults during recovery and
`_persist_live_recovery` writes the empty pause marker. A regression test covers
absent, null, and empty fields. Ordinary queued runtime defaults remain
unchanged because lane assignment and recovery normalization are separate
stages in this coordinator.

Focused checks passed: 25 tests across recovery cases and state views. The
first complete run found the generated runner function map stale after source
changes; regenerated it and verified its `--check` mode. The clean full Linux
suite then passed: `.venv/bin/python -m pytest -q` — **18,450 passed in 106.01
sec**. This remains fixture/unit evidence; it does not exercise live GPU
pressure or a real dataset campaign. Full behavior-by-behavior ledger
reconciliation remains active.

### Bounded source differential: soft-gate close and external reopen (2026-10-23)

Extended the source differential to execute the copied runner's actual inline
gate-transition block (source lines 3075–3124) in a bytecode-disabled
subprocess. Compared its final gate flags, lock IDs, and lane-enable flags with
`update_soft_gates` plus `reopen_external_gates` across **82,944** combinations
of initial gate/lock state, lane flags, latest CPU/GPU jobs, host/GPU pressure,
GPU sample availability, child footprints, and external-drop events. The
product includes exact buffered equality and stale-lock retention/release
cases. No mismatch was found. Focused differential and snapshot-integrity
tests passed (**2 passed**); the complete Linux suite passed **18,451 tests in
108.96 sec**. This closes the enumerated gate-state comparison, not every
controller interleaving or live process/resource-sensor behavior.

### Bounded source differential: pressure-victim selection (2026-10-23)

Executed the copied runner's candidate-filtering and sorting statements
(source lines 3312–3349) and compared the selected victim with
`SuiteCoordinator._suspend_pressure_victim`. The differential exercises 120
permutations of candidate order under GPU-only, host-only, and simultaneous
pressure (360 cases), including GPU lane filtering, paused/dead exclusions,
RSS/VRAM ordering, descending architecture priority, and the job-ID tie break.
GPU pressure remains the selected pressure kind when both pressure flags are
true, while both reason labels are retained. Focused differential and snapshot
tests passed (**2 passed**); the full Linux suite passed **18,451 tests in
110.16 sec**. Live process suspension and hardware telemetry remain outside
this static fixture comparison.

### Bounded source differential: Linux CPU runtime-tuning helpers (2026-10-23)

Compared the copied `utils/runtime_tuning.py` helpers directly with the
generic Linux implementation in **1,635** bounded cases: CPU partitioning
across empty and 1–8 CPU tuples, part counts and slots from -2 through 10;
13 valid, empty, duplicate, reversed, and malformed CPU-list strings;
deterministic SHA-1 slot selection for four keys and part counts -2 through
10; and explicit/environment concurrency-hint precedence across seven
environment values and seven explicit values. Generic and source results or
exception types agreed in every case. Focused tuning, source-differential, and
snapshot-integrity tests passed (**40 passed**); the complete Linux suite
passed **18,451 tests in 112.61 sec** after the standalone-wrapper source differentials. This does not exercise live
affinity/priority effects or every possible string and integer input.

### Standalone wrapper resume-gate parity (2026-10-23)

Extracted the standalone wrapper's host/GPU resume decision into the reusable
`_external_drop_reopens_gates` helper and routed the pause loop through it,
preserving the source's independent readiness-or-drop condition for each
domain and its `threshold_mib > 0` switch. A direct differential against the
copied helper matched **600** combinations spanning ready flags, negative,
zero, below/equal/above-threshold drops, and disabled/nonpositive thresholds.
The standalone pressure and source-fidelity tests passed (**25 passed**).
This helper-level result does not certify the complete controller lifecycle
under live hardware pressure.

The same source differential compares the GPU percentage sampler and
process-name-filtered Python GPU usage sampler over **10** synthetic
`nvidia-smi` outputs each, and compares host Python RSS accounting across
four process-record sets plus iterator-construction and iterator-traversal
failures (**26 comparisons total**). The cases include fractional MiB,
zero/nonpositive totals, negative and over-total usage, malformed/empty rows,
multiple rows, query failure, Python/non-Python names, all documented
per-process exceptions, and propagation of unexpected record and iterator
errors. Generic results and exception types matched the source in every case.
These fixed-input checks supplement lifecycle tests; they do not measure a
physical GPU or host pressure event. The refreshed full Linux suite passed
**18,451 tests in 105.96 sec**.

### Bounded source differential: central child CPU environment (2026-10-23)

Compared the copied `launcher_child_env` with the generic child environment
builder across **288** cases crossing allowed-core counts (1, 2, 5), inherited
and explicit concurrency hints, optional affinity overrides, affinity slot
selection, and inherited shared-mode values. The comparison checks common
thread, worker, affinity, concurrency, and OpenMP/KMP environment keys; all
matched. This validates tuning before process-launch metadata overrides.
Device-specific backend/remapping fields and end-to-end Popen environment
assembly remain covered by their dedicated adapter/process tests, not this
helper-only comparison. The refreshed full Linux suite passed **18,451 tests in 105.73 sec**.

The device-specific differential compared source `make_child_env` with the
generic builder across **264** combinations of CPU/GPU mode, 11 inherited
CUDA masks, three requested indices, and four parent CPU-only controls. It
found that successful generic GPU environments did not explicitly set the
adapted `SIMPLE_ADP_DISABLE_GPU_ACCELERATORS` flag to `0`, while the source
sets its corresponding flag to `0`. The GPU builder now sets this value
explicitly. The comparison normalizes the intentionally renamed OPF/Simple-ADP
backend fields and checks acceptance/rejection plus device visibility and
allocator controls. Focused source-fidelity, runtime-tuning, and process tests
passed (**54 passed**), and the full Linux suite passed **18,451 tests in 105.73 sec**. This is a mocked
environment comparison; it does not launch children on physical GPUs.

### Bounded source audit: central child launch ordering (2026-10-23)

Read the complete source `launch_job` and generic `processes.launch_job`
implementations side by side. Both create job/result directories, increment the
attempt and set the selected device and start time, write command and metadata
files, optionally write launch context, persist `launching` state before
building the child environment, then open the append log and write the launch,
command, and optional context markers before spawning. On POSIX both start a
new session, inherit stdin, send stdout to the UTF-8 append log, and merge
stderr into stdout. After successful spawn both persist `running` state with
the PID. The generic implementation uses its injected repository root and
Simple-ADP identity/checkpoint fields in place of the source's fixed root and
OPF-specific fields; these are intentional adapter differences.

Added focused failure-path contracts. An environment-builder exception leaves
attempt 1 and `launching` state persisted, with metadata already present and no
job log yet opened. A `Popen` exception leaves `launching` state and the launch
marker on disk, leaves the runtime PID unset, and closes the opened log handle.
The process tests passed (**17 passed in 0.92 sec**). These are Linux unit
contracts with a mocked spawn; they do not verify scheduler recovery after an
actual operating-system launch failure.


### Full-pass controller cross-product refresh (2026-10-23)

Revalidated the existing `test_full_pass_exit_pressure_and_peer_lane_cross_product`
from the current checkout. It executes **16,128** synthetic `SuiteCoordinator.run`
passes over seven exit/device cases, four host/GPU pressure combinations, four
peer-lane sets, three total active caps, three GPU active caps, four gate
states, and four hold states. The test passed (**16,128 passed in 71.77 sec**).
Updated the controller coverage row and source/function map because their prior
claim that the full controller cross-product was absent had become stale. The
matrix checks generic controller transitions and selected persisted outcomes;
it does not execute the copied source controller, exercise real GPU/host
pressure, or exhaust every possible runner state/input. The overall
source-to-code reconciliation remains active.

### Source-executing process-launch differential (2026-10-23)

Added a subprocess-isolated test that invokes the copied runner's actual
`launch_job` and the generic `processes.launch_job` with mocked environment
builders and `Popen`. Both produced the sequence `launching state → environment
build → Popen → running state`; their Linux child commands, cwd, text mode,
merged stderr, append log, inherited stdin, and new-session semantics matched.
The generic call also supplies `creationflags=0`, which is accepted on Linux and
does not change the process-group behavior. The focused source differential
passed (**1 passed in 1.41 sec**). A second direct source differential,
`test_process_launch_failure_states_match_reference`, compares environment-
build and Popen failure behavior for persisted status, attempt count, log-open
timing, and handle cleanup; it passed (**1 passed in 1.21 sec**). These
comparisons do not execute child workers or compare every persisted metadata
field.


The complete Linux suite passed **18,455 tests in 107.35 sec** after adding
the direct source/generic launch success and failure differentials. The runner
function inventory and ADP guide block crosswalk both pass their generated
`--check` commands. This is updated repository test evidence; the broader
runner fidelity audit remains active.

### Central admission-budget source-order clarification (2026-10-23)

Re-read the source controller around its preliminary launch-budget adjustment
and later queue-fill calculation. At source line 2965, it raises the local
`launch_budget` to `max(launch_budget, min(concurrency, active_count + 1))`
when pressure, gates, and holds permit. That value is never observed: the queue
loop overwrites `launch_budget` at line 3409 from the current concurrency and
batch scale before checking or passing it to `launch_job` (lines 3410–3499).
The effective admission behavior therefore remains `min(ceil(concurrency *
batch_scale), earlier launch_capacity)`. The generic runner implements that
observable formula. This source dead write is recorded so the prose in ledger
step 3 is not mistaken for a separate admission boost; no code change is
needed to reproduce an unobservable assignment.


### Source-executing one-route controller comparison (2026-10-23)

Added `test_source_main_and_generic_coordinator_retry_trajectory_match`.
It runs the copied runner's real `main()` scheduling loop and the generic
`SuiteCoordinator.run()` on the same one-route CPU fixture, with only resource
sensors and child processes mocked. The child exits with code 7 once, is
requeued and relaunched, then exits 0. Both controllers report one completed
route, one failed attempt, one retry, attempt 2, retry count 1, final exit 0,
and exactly two launches. The focused comparison passed (**1 passed in 1.43
sec**). This adds direct evidence for the complete ordinary retry trajectory;
it does not compare pressure-driven multi-job interleavings or real OS/GPU
resource behavior.


The complete Linux suite passed **18,456 tests in 108.43 sec** after adding
the source-executing one-route controller retry comparison. Both generated
audit maps pass their `--check` commands. The source-level reconciliation is
active; multi-job pressure interleavings and live hardware remain outside this
synthetic controller comparison.


### Source-executing pressure pause and resume comparison (2026-10-23)

Added `test_source_main_and_generic_coordinator_external_gate_reopen_resume_match`.
It runs the copied `main()` and generic coordinator on two synthetic CPU jobs
with matching per-pass host samples: normal availability while both jobs
launch, one host-pressure sample that pauses the larger-RSS child, then a
600 MiB external-host drop from a tracked PID that disappears. The gate reopens
and the suspended route resumes on CPU; both runs complete. Both sides match on
completed count (2), failed attempts (0), successful launches (3), final
statuses, per-job attempts `[1, 2]`, and pause counts `[0, 1]`. The test passed
(**1 passed in 1.42 sec**) and asserts the source reopen log plus the generic
`external_host_reopened` snapshot. OS processes and sensors are mocked; GPU
pressure and multi-domain interleavings are not covered by this direct
execution comparison.


The refreshed complete Linux suite passed **18,457 tests in 109.93 sec** after
adding the source-executing host-pressure pause/resume comparison. Generated
runner and ADP maps are current.


### Source-executing GPU OOM peer lifecycle comparison (2026-10-23)

Added `test_source_main_and_generic_coordinator_gpu_oom_peer_resume_match`.
Both real controller loops run two synthetic GPU routes. The first child emits
`CUDA out of memory` after its peer has launched. The source and generic
controllers select and pause the peer, retry the failed route with reduced
batch scale, complete that retry, and resume the paused peer after a synthetic
VRAM recovery. Both match on completed routes (2), failed attempts (1),
retries (1), four total launches, per-route attempts `[2, 2]`, retry counts
`[1, 0]`, pause counts `[0, 1]`, GPU pause counts `[1, 1]`, GPU lane identity,
and final batch scale 1.0. The focused comparison passed (**1 passed in 1.22
sec**). This is a mocked GPU-sensor/child run; it does not test CUDA hardware,
actual allocator OOM, or arbitrary multi-job OOM interleavings.

The full Linux suite passed **18,458 tests in 110.14 sec** after adding the
source-executing GPU OOM peer lifecycle comparison. Both generated audit-map
checks passed. Direct source/generic controller execution now covers bounded
normal retry, host-pressure pause plus external-PID gate reopen, and GPU OOM
peer pause/retry/resume trajectories; it does not establish every possible
interleaving or live hardware behavior.

### Full Linux suite refresh and GPU pressure-victim parity (2026-10-23)

Added a source-executing comparison of the copied runner's `main()` and the
generic coordinator for GPU hard pressure. The fixture gives one active child
100 MiB VRAM use and the other 400 MiB, injects a strict low-free-VRAM sample,
and then restores free VRAM. Both controllers pause the larger-footprint GPU
child, relaunch it on the same GPU lane, and finish both jobs with matching
attempt, pause, GPU-pause, and device state. The focused case passed (**1
passed in 1.44 sec**). Resource samples, child processes, and process-tree
termination are mocked; this does not exercise physical GPU pressure.

The refreshed full Linux suite passed **18,460 tests in 111.96 sec** after
adding the real-child Linux STOP/CONT integration case. The focused ADP
acceptance selections passed **46 smoke tests in 2.55 sec** and **542 core,
growth, controller, and checkpoint tests in 10.18 sec**. The standalone
pressure-wrapper module passed **25 tests in 0.82 sec**. Both generated
freshness checks passed after the update:
`generate_runner_function_map.py --check` and
`generate_adp_block_crosswalk.py --check`. Physical pressure and GPU hardware
remain untested.

### Full Linux verification refresh (2026-10-23)

After removing an exploratory real-child coordinator comparison that allowed
the child to finish during the synthetic pressure window, `.venv/bin/pytest -q`
passed **18,460 tests in 112.39 sec**. The source fidelity module passed **7
tests in 11.00 sec** independently. This refresh does not close the documented
source/generic interleaving or hardware pressure gaps; all partial audit rows
remain partial.

### Real-child central pressure restart comparison (2026-10-23)

Added `test_source_main_and_generic_coordinator_restart_a_real_child_after_host_pressure`.
It runs the copied `main()` and `SuiteCoordinator.run()` with real Linux child
processes and injected host-memory samples. The first child stays live while
available memory is strictly below the stop threshold; the source and generic
controllers each terminate it, persist one host pause, then relaunch on CPU
when the available-memory resume threshold is met. The second child exits
successfully. The focused comparison passed (**1 passed in 2.10 sec**) with two
launches, one host pause, zero failed attempts, and completed final state for
each controller. This verifies that pressure decision, child termination, and
same-lane restart are connected end to end on Linux; pressure sensors remain
synthetic and this does not cover arbitrary multi-child interleavings.

The refreshed full Linux suite passed **18,461 tests in 114.64 sec** after the
real-child central pressure restart comparison. The source-fidelity module
passed **8 tests in 13.11 sec**. Generated inventory checks were rerun after
updating the source map; focused and full suites provide regression evidence,
not proof for arbitrary multi-child timing or physical pressure sensors.

### Direct resume-admission bypass comparison (2026-10-23)

Added `test_source_main_and_generic_coordinator_resume_all_paused_jobs_past_active_caps`.
It begins both controller loops with two paused CPU jobs, one ready host-memory
snapshot, and `concurrency=1` plus `max_active_jobs=1`. The source and generic
loops each relaunch both suspended jobs before queue admission, on their
original CPU lane, then reap both successful exits. The focused comparison
passed (**1 passed in 1.25 sec**) and matched two launches, attempt 2 for both
jobs, and completed states. This pins the ledger's resume-loop admission bypass
against the source implementation; it uses synthetic process handles and does
not exercise real concurrent resume processes.

### Uncapped launch-error retry comparison (2026-10-23)

Added `test_source_main_and_generic_coordinator_match_uncapped_launch_error_spin`.
Both the copied source `main()` and `SuiteCoordinator.run()` receive repeated
synthetic spawn failures with `max_retries=0`; each makes three failed attempts
and immediately retries within queue fill without invoking the configured poll
sleep. A fourth injected interruption stops each loop. The focused differential
contract passed (**1 passed in 1.24 sec**) and both record three failed attempts
and three retries before interruption. This reproduces the ledger's uncapped
retry edge; it intentionally tests the recorded tight retry behavior, not
operational recommendation.

The refreshed full Linux suite passed **18,463 tests in 116.46 sec** after the
same-pass resume-bypass and uncapped launch-error source comparisons. The
source-fidelity module passed **10 tests in 15.71 sec**. No physical pressure
sensor run was made; controller comparisons use deterministic injected
samples, with real Linux child processes for the single-child kill/relaunch
case.

### Report and compatibility-artifact schema crosswalk (2026-10-23)

Compared the source artifact writers with `JobStateStore`,
`SuiteCoordinator._write_final_reports`, and live/offline telemetry summaries.
The final-report process-outcome fields have a direct semantic mapping:
`total_jobs` → `manifest_jobs`, `completed_jobs` → `completed`,
`failed_jobs` → `terminal_failures`, and `retried_jobs` → `retries`; run/results
roots retain their meanings. A source-executing differential contract pins
those mappings and both complete report key sets. Generic-only fields are
`run_signature`, invocation-local `failed_attempts`,
`recovered_unknown_exit`, `resource_snapshot`, `hardness_screening`, and live
telemetry counters. They describe generic plan identity, retry attempts,
recovery, screening, and adapter-owned progress and do not change source
outcome counts. The generic persisted-state report remains a distinct mode: it
derives counts from persisted job state and sets invocation-local failure and
recovery counters to null because they cannot be reconstructed from those
files.

Compatibility views preserve aggregation mechanics while changing identity
schemas and paths. Source grouping uses case/repeat/phase; generic grouping
uses provider/suite/task/repeat and model/profile. Both count `launching` as
running, map unknown status to queued, choose latest completion by numeric-like
`ended_at` and job ID, and put the first successful state in sorted discovery
order into `best_candidate_dir` without scoring. Candidate, method, task, and
replay records use generic identity/route data in this package; they are
semantic compatibility artifacts, not byte-compatible source records.

CSV schemas are adapted too. Source `run_progress.csv` records
phase/family/case/repeat/depth/width; generic records
provider/suite/task/split/model/profile/route/seed/architecture. Both append
lifecycle rows and best-effort fsync. Source `task_summary.csv` contains
phase/family/case/repeat/depth/width/status/attempt/last-exit/results; generic
contains job ID/status/attempt/retry count/current and last exit/result. Both
rewrite one row per planned job after drain and best-effort fsync. Report file
names and atomic JSON/text replacement are preserved; Markdown wording is
adapted. The field-mapping contract passed on Linux (**1 passed**). Existing
state-view and progress tests cover generic aggregation, ordering, task rows,
fallback precedence, and counter semantics. These artifacts are not claimed
to have source-identical schemas or paths.

This closes the report/state-field inventory gap at the semantic-crosswalk
level. It does not close full ledger-to-code reconciliation, arbitrary
controller interleavings, or physical sensor/hardware behavior.

Verification after this crosswalk: `tests/test_runner_source_fidelity.py`
passed (**11 passed in 16.91 sec**); `scripts/generate_runner_function_map.py
--check`, `scripts/generate_adp_block_crosswalk.py --check`, and `git diff
--check` passed. The full Linux suite then passed (**18,464 passed in 117.60
sec**). This refresh is Linux test evidence, not live GPU or physical pressure
certification.

### Source differential: simultaneous host and GPU pressure (2026-10-23)

Extended the source-executing pressure-victim comparison so host and GPU
pressure are both strictly active in the same scheduler pass. The two active
GPU children have opposing footprints: host RSS would select route 0, while
GPU VRAM selects route 1. Both the copied source controller and generic
coordinator select route 1 because GPU pressure takes precedence, and both
record `gpu_memory_pressure+host_ram_pressure`. The selected route is
terminated/relaunched on GPU and completes; the other route completes without
a pause. The focused source-fidelity module passed (**11 passed in 17.02 sec**)
and the full Linux suite passed (**18,464 passed in 118.07 sec**). This closes
the direct source comparison for simultaneous-domain precedence and conflicting
victim footprints. Sensors were injected; this does not validate physical
pressure sampling or all multi-child pass orderings.

### Source differential: recovered child exits after restart (2026-10-23)

Added a direct transition comparison for the copied runner's
`_requeue_recovered_runtime_after_exit` and the generic coordinator's
`_handle_exit`. When a recovered child is no longer live, both paths remove it
from active work, clear its process/PID and exit status, retain attempt and
retry counts, mark it queued, and append it behind already queued work. Neither
path increments failed-attempt or retry counts. The source persists
`reason=recovered_child_exit_status_unknown`; the generic path maps that to
`last_error` and additionally increments its `recovered_unknown_exit` counter
and appends an event. These report-field differences are explicit and
intentional. The focused differential test passed (**1 passed**). This checks
the exit transition and FIFO order directly; full process discovery and
restart-loop interleavings remain separate audit coverage.

### Full Linux verification after recovered-child differential (2026-10-23)

The recovered-child differential test and report note are included in the
current checkout. `tests/test_runner_source_fidelity.py` plus the generic
unknown-exit integration contract passed (**13 passed in 19.17 sec**). The full
Linux test suite passed (**18,465 passed in 117.69 sec**). Both generated
inventories passed their `--check` commands and `git diff --check` was clean.
This verifies software behavior under the test fixtures; it does not establish
physical host/GPU pressure response or exhaustive real-process scheduler
interleavings.

### Source differential: mixed-device latest-child sampling (2026-10-23)

The ledger's latest-child quirk was previously tested only against the generic
coordinator. Added a source differential with a later-started CPU child and an
earlier GPU child. The copied source's `_latest_active_runtime` selects each
according to its general/GPU filters; its main-loop sampling order uses the
general latest runtime, so both implementations record the CPU child's host
RSS and zero GPU VRAM while associating the GPU gate with the independently
selected GPU runtime. The remaining-GPU estimate is therefore the full sampled
free VRAM. Focused differential: **1 passed**. This verifies the recorded
mixed-device selection quirk, including the intentionally undercounted GPU
child footprint; it does not establish physical sensor accuracy.

### Linux verification after mixed-device sampling differential (2026-10-23)

The focused source-fidelity module plus the generic mixed-device sampling
contract passed (**14 passed in 20.62 sec**). The complete Linux suite passed
(**18,466 passed in 120.20 sec**). Runner function inventory, ADP block
crosswalk, and `git diff --check` all pass. This is injected-resource test
evidence; full source/generic scheduler interleaving coverage and physical
pressure behavior remain open.

The copied reference snapshot was revalidated against all **46** entries in its
SHA-256 manifest after these audit additions; there were zero byte-count or
hash mismatches. The source snapshot remains unchanged.

### Source differential: strict hard-pressure boundaries (2026-10-23)

Added a direct comparison that executes the copied controller's `gpu_pressure`
and `host_pressure` assignment statements and compares them with generic
`hard_pressure`. Twelve cases cover host/GPU values just below, equal to, and
just above stop thresholds, plus a missing GPU sample. Equality is not
pressure; missing GPU telemetry is not GPU pressure. Focused test: **1 passed**.

### Linux verification after strict pressure-boundary source comparison (2026-10-23)

The first full-suite run caught a test-isolation regression: importing the
reference module from the pytest process wrote bytecode caches into the copied
snapshot. The boundary contract now reads and executes only the two source
predicate statements, without importing that module. Generated `__pycache__`
files were removed, the dedicated snapshot-integrity contract passed, and the
clean full Linux suite passed (**18,467 passed in 118.47 sec**). This preserves
the reference snapshot while adding the strict-threshold differential.

### Source differential: concurrency reread timing around capacity (2026-10-23)

Added a controlled-reread comparison for the capacity stage of a controller
pass. It varies the setting between the launch-capacity helper read and the
separate finite-concurrency `active_count + 1` check. The copied helper/caller
and generic `_sample_snapshot` both produce capacities 2, 5, and 1 for bounded,
bounded-to-unbounded, and one-slot cases, respectively, while consuming the
same number of control-file reads. Focused test: **1 passed**. This verifies
that intra-pass capacity reads match; it does not replace a full source/generic
run through arbitrary concurrency-file edits at every loop point.

### Full Linux verification after concurrency reread differential (2026-10-23)

The source-fidelity and immutable-snapshot tests passed (**16 passed in 20.54
sec**). The complete Linux suite passed (**18,468 passed in 119.62 sec**).
Both generated inventories and `git diff --check` passed. The 46-file copied
source snapshot remains the comparison authority and is checked separately
against its manifest.

### Source differential: live concurrency change across passes (2026-10-23)

Added a full-loop comparison using the copied `main()` and generic
`SuiteCoordinator.run()`. Both start with concurrency 1; the first successful
child launch changes the shared control file to 2. On the following pass, each
loop admits the next FIFO route and completes both children. This supplements
the generic-only dynamic-limit tests and the intra-pass capacity-read
comparison. Focused source-loop test: **1 passed**. The setting takes effect for
subsequent admission; it does not batch multiple successful launches into one
pass.

### Full Linux verification after live-concurrency source comparison (2026-10-23)

The source-fidelity module and selected concurrency integration cases passed
(**18 passed in 22.74 sec**). The complete Linux suite passed (**18,469 passed
in 123.01 sec**). Generated runner and ADP inventories pass their `--check`
commands and `git diff --check` is clean. The copied snapshot remains unchanged
under the dedicated manifest test.

### Source differential: retry-to-tail FIFO order (2026-10-23)

Added a two-route full-loop comparison with the first attempt of route A exiting
nonzero. Both the copied runner and generic coordinator launch routes in order
A, B, A: B was already queued, so A's retry returns behind it. Final states,
attempt counts, retry counts, completed count, failed-attempt count, and retry
count agree between the two loops. Focused source-loop test: **1 passed**.

### Full Linux verification after retry-to-tail comparison (2026-10-23)

The source-fidelity and immutable-reference tests passed (**18 passed in 22.81
sec**). The complete Linux suite passed (**18,470 passed in 122.52 sec**).
Generated runner and ADP inventories pass their `--check` commands, and
`git diff --check` is clean.

### Source differential: failed pressure resume with queued admission (2026-10-23)

Reread the supplied ledger and implementation plan attachments; their SHA-256
identities remain `02ce31f510c2c9a2b87b015c7e63ce5114719585aed44b928006254cc13e3856`
and `3dd3b35057cef2982a7bba9467f1bad0873280adbab14e984d2ccb1df8e68cd8`.
Added `test_source_main_and_generic_resume_failure_still_launch_queued_route`,
which runs the copied runner's `main()` and the generic coordinator through the
same failed paused-job relaunch, successful FIFO queued-job launch/completion,
and KeyboardInterrupt shutdown. It compares launch order, attempts, final job
statuses, retained pause cause, and interruption state. This complements the
function-level state comparison and generic full-pass queue-admission contract.
The focused three-contract selection passed (**3 passed in 3.64 sec**).

The complete Linux suite passed (**18,472 passed in 125.87 sec**). Runner-map
and ADP-crosswalk freshness checks, source compilation, `git diff --check`, and
reference snapshot verification passed. The copied snapshot has 46 manifest
entries with matching hashes; its README remains the documented unlisted note.
This closes only this resume/admission trajectory. Remaining partial ledger
areas, physical host/GPU response, and live dataset execution remain open.


### Live GPU backend and sampler check (2026-10-23)

The current Linux host exposes an NVIDIA GeForce RTX 3050 Laptop GPU (6 GiB)
and the project environment reports Torch CUDA available. Built the configured
GPU child environment, asserted its selected device/backend and CPU-admission
flags, ran the backend's real Torch allocation/fill/synchronize probe, allocated
and synchronized a one-element CUDA tensor, then sampled device 0 through the
runner's `nvidia-smi` resource sampler. The checks passed; the sample reported
6,144 MiB total and 5,996 MiB free. No model, worker job, suite, or memory
pressure workload was launched. This closes live backend visibility and sampler
availability on this host; physical host/GPU pressure transitions and training
performance remain unverified.

### Fidelity audit refresh (2026-10-23)

Refreshed the source-to-code audit evidence against the supplied runner ledger
and implementation plan. The runner function inventory and ADP guide block
crosswalk both pass their `--check` commands. The full Linux unit/integration
suite passed **18,472 tests in 126.13 sec**. The audit records source-level
comparisons for defaults, controller ordering, resource and pressure predicates,
admission, process/retry/recovery behavior, telemetry/checkpoint handoff,
Linux tuning, bundles, and reporting. It also includes differential full-loop
cases for pressure pause/restart, retry-to-tail order, failed resume with queued
admission, and changing concurrency.

This is strong bounded fidelity evidence, not proof over every possible process
interleaving or live hardware condition. The suite was not launched. Physical
host/GPU pressure transitions, long-running dataset campaigns, and unbounded
state-space behavior remain unverified; Windows behavior remains excluded by
the user's Linux-only scope. The ledger's detailed mechanism/value comparison
is recorded, but the traceability table retains `Partial` where these limits
or provider-owned behavior remain, rather than treating passing tests as
universal certification.

The attachment identities were rechecked before this audit pass: ledger SHA-256
`02ce31f510c2c9a2b87b015c7e63ce5114719585aed44b928006254cc13e3856` and plan
SHA-256 `3dd3b35057cef2982a7bba9467f1bad0873280adbab14e984d2ccb1df8e68cd8`.
The isolated source-differential and copied-snapshot integrity tests then passed
(**20 passed in 25.26 sec**), confirming the reference copy still matches its
manifest and the checked source/controller trajectories agree. This remains a
bounded set of comparisons, not an exhaustive proof of all runtime schedules.

### Individual vanilla-route smoke coverage (2026-10-23)

The suite planner already emitted 50 individual fixed-MLP routes, but the
runner adapter tests did not execute each one. Added a worker-adapter smoke
contract for all 50 architectures on both classification and regression toy
tasks. Each route trains for one epoch with a batch equal to the complete
four-row training partition and verifies the selected architecture, history,
and held-out test metrics. The targeted matrix passed **100 cases in 2.56 sec**.
The full Linux suite then passed **18,572 tests in 127.62 sec**. This closes
individual vanilla route smoke coverage on representative task adapters; it
does not claim a live run for every registered OpenML task.

### Canonical ADP route adapter smoke coverage (2026-10-23)

Extended the adapter matrix to cover all 13 non-fixed canonical ADP identities,
both faithful/corrected profiles for the generic identities, and both built-in
task kinds. Each is fitted for one epoch with a full training-partition batch,
with architecture caps preventing extra growth candidates. The targeted model,
smoke, and coordinator integration selection passed **194 tests in 10.17 sec**;
the complete Linux suite passed **18,614 tests in 125.82 sec**. Together with the
individual vanilla matrix above, all registered model-route variants now have
adapter-level smoke evidence on classification and regression fixtures. This
does not cover live OpenML task downloads or dataset-specific runtime behavior.

### External model-adapter child execution (2026-10-23)

Extended the subprocess integration fixture with a package-entry-point model
adapter that registers a custom route, delegates fitting to the built-in tabular
adapter, and runs through the ordinary suite plan and worker child. The
coordinator completed all three built-in routes plus the plugin route and wrote
completed result records for each. The focused plugin/end-to-end tests passed
(**2 passed in 9.38 sec**); the full Linux suite passed **18,614 tests in
130.84 sec**. This verifies external model-adapter discovery and execution at
the process boundary; it does not test arbitrary third-party model semantics.

### Aggregate fixed-grid runner-adapter smoke coverage (2026-10-23)

Added a contract that sends the planner's aggregate `MLP-Fixed` route, carrying
the complete 50-architecture vanilla grid, through `SimpleADPModelAdapter` for
classification and regression. Both use one epoch and a batch equal to the
four-row training partition. The test asserts 50 history entries, confirming
the adapter/controller traversed every supplied candidate; `max_candidates=1`
is set deliberately and does not truncate this dedicated fixed-grid traversal.
The focused aggregate-plus-individual vanilla matrix passed **102 tests in
2.97 sec**. The full Linux suite then passed **18,616 tests in 129.85 sec**.
Both generated inventories passed their `--check` commands, and
`.venv/bin/python -m compileall -q src tests` completed successfully.

### Source-run-root copy fidelity correction (2026-10-23)

A direct differential check found that `seed_run_root_from_source` copied files
under a reserved top-level artifact name when that source path was a directory,
where the reference skips the entire reserved subtree. It also created an empty
destination for missing, empty-directory, and regular-file sources, while the
reference creates destination parents only when it copies a file. The generic
helper now preserves the reserved-first-component exclusion and lazy parent
creation. Differential coverage compares copied/skipped counts, resulting
files, no-overwrite behavior, and destination-existence side effects with the
copied reference implementation. It also checks that a reserved-name source
directory with no existing destination raises the same `IsADirectoryError`
after creating the destination parent, without copying its descendants. The
focused differential test passed again (**1 passed**). The source-fidelity
group passed **20 tests
in 26.58 sec**, including the new regression. The complete Linux suite then
passed **18,617 tests in 130.41 sec**; the regenerated function inventory,
ADP block crosswalk, and Python compilation checks also passed.

### Full model-route planning across registered OpenML pools (2026-10-23)

The prior OpenML planning test combined each registered pool with its listed
tasks but restricted the model routes to width-only. Added a fixture-backed
Cartesian planning contract for `cc18`, `tabzilla`, and `ctr23`: two listed
task IDs per pool each receive the complete **146-route** adapter set, covering
all 14 canonical IDs, both generic profiles where applicable, the aggregate
and 50 individual vanilla routes, and 44 diagnostic screens. It verifies each
task's route records, suite identity, split hash, and persisted train/validation/
test indices. The focused OpenML provider-plan group passed **7 tests in
1.33 sec**. This validates plan construction without downloading or training
on OpenML data; the dataset campaign remains unlaunched.

After adding this contract, the complete Linux test suite passed **18,620
tests in 130.28 sec**. Both generated inventory `--check` commands passed,
and `.venv/bin/python -m compileall -q src tests` completed successfully.

### Source differential: result bundle round-trip (2026-10-23)

The existing bundle contracts checked the generic implementation's archive,
restore guard, and safety behavior, but did not execute the copied source and
generic bundle paths against the same fixture. Added a source-executing
differential that compares stable manifest fields, archived file names and
bytes, telemetry CSV filtering, and restored result trees. Both paths omit
root and nested `telemetry.csv` while retaining the telemetry manifest and
other CSV files. The source differential passed (**1 passed**); the generic
bundle module passed (**20 passed in 1.00 sec**). This verifies fixture-backed
export/restore parity; chunked archives above 64 MiB and external storage
failure conditions remain separate cases.

The refreshed full Linux suite passed **18,621 tests in 131.69 sec** after
adding the differential. Both generated inventory checks, Python compilation,
and `git diff --check` passed.

### Source differential: multipart bundle round-trip (2026-10-23)

Extended the bundle comparison to the multipart branch by reducing the source
and generic part-size constants to 128 bytes in an isolated subprocess. Both
implementations emitted sequential bounded parts and equivalent stable
manifest fields, restored the same payload, filtered telemetry CSV, and
removed the successful `.restore` file. A corrupt part also produced the same
exception type, created the result parent before reading, and retained the
partial `.restore` artifact in both implementations. The source differential
passed (**1 passed**); the generic bundle module passed (**20 passed in
0.77 sec**). The test exercises chunk ordering and residue behavior without
allocating a 64 MiB archive; it does not measure sustained large-file I/O.

The complete Linux suite then passed **18,622 tests in 131.63 sec**. Both
generated inventory checks, compilation, and `git diff --check` passed.

### Source differential: bundle manifest-write failure order (2026-10-23)

Added a filesystem-failure contract where the manifest destination is a
directory and an older archive plus `.part000` already exist. Both exporters
raise the same exception before removing either prior artifact, confirming
the ledger's manifest-first cleanup order through real filesystem operations.
The source differential passed (**1 passed**); the generic bundle module
passed (**20 passed in 0.78 sec**). This covers that concrete write failure;
permission-denied and device-full behavior are not asserted on this root-run
Linux environment.

The refreshed complete Linux suite passed **18,623 tests in 130.17 sec**
after adding the manifest-write failure contract; generated inventory,
compilation, and whitespace checks were refreshed separately.

### Source differential: telemetry shard rehydration (2026-10-23)

Added an isolated source-vs-generic contract for sharded telemetry. Both paths
restore concatenated shards with a valid optional digest, skip an already
existing target before inspecting its missing parts, leave `.rehydrate` after
a missing-shard or escaped-shard failure, and delete that temporary after a
checksum mismatch. The contract also compares the exception classes and
resulting bytes. Focused verification passed (**1 passed**). This exercises
the path and checksum branches with fixture files; large shard I/O and
concurrent manifest mutation are not exercised.

The complete Linux suite passed **18,624 tests in 130.83 sec** after adding
this comparison. Generated inventory checks, Python compilation, and
`git diff --check` passed.

### Startup retry-budget source differential (2026-10-23)

Added `test_source_and_generic_startup_retry_budget_and_terminal_resume_failure`.
Both actual controller loops load two persisted failures. An ordinary failure
with `attempt=2`, `retry_count=1`, and `max_retries=1` is queued, launched once,
and completed at `attempt=3` while the retry count remains one. An
`inexact_checkpoint_resume` failure stays terminal and is not launched. This
verifies both sides of the startup rule; the focused differential passed.
The current Linux suite then passed **18,969 tests in 204.59 sec**; immutable
snapshot verification, both generated inventories, and `git diff --check`
passed.

### Source differential: standalone wrapper relaunch after paused-child exit (2026-10-23)

Added an isolated source-vs-generic executable differential for the standalone
pressure wrapper. Both implementations receive identical host/swap/GPU and
process-memory fixtures. The first child is paused on high host pressure, exits
while the wrapper is waiting for resume gates, and is then replaced by a second
child once the host gate reopens. The comparison covers exit status, child
launch arguments/order, pause events, and emitted output. The focused differential
passed (**1 passed**); the combined source-fidelity and generic pressure-wrapper
groups passed (**50 passed in 28.40 sec**). This is deterministic lifecycle
coverage; it does not exercise real sensor hardware or nondeterministic OS
scheduling beyond the existing Linux signal tests.

The complete Linux suite passed **18,625 tests in 131.41 sec** after adding
the standalone-wrapper relaunch differential. Both generated inventory checks,
Python compilation, and `git diff --check` passed before this full run.

### Source differential: standalone wrapper natural child exit (2026-10-23)

Added an isolated source-vs-generic standalone-wrapper comparison for a child
that exits naturally with status 7 while no pressure gate is active, plus the
adjacent race where the process disappears before its PID can be inspected.
In both cases the implementations launch exactly once and propagate status 7;
the standalone wrapper does not apply the central suite's retry policy. The
source-fidelity module passed (**26 passed in 28.58 sec**). The test uses a
mocked child and synthetic host/swap readings; OS signal and real process
behavior remain covered by their separate Linux contracts.

The complete Linux suite passed **18,626 tests in 132.35 sec** after the
natural-exit differential. Both generated inventory checks, Python compilation,
and `git diff --check` passed before the run.

After expanding the standalone natural-exit comparison to include PID loss
before process inspection, the complete Linux suite passed **18,626 tests in
132.48 sec**. The function-map and ADP-crosswalk checks, Python compilation,
and `git diff --check` passed before the run.

### Source differential: standalone real process-group pause and resume (2026-10-23)

Added a Linux integration differential that runs the copied standalone wrapper
and the generic wrapper with a real isolated child process and identical
synthetic host/GPU readings. Each wrapper observes strict host pressure,
SIGSTOPs its process group, reopens both resume gates on the next sample,
SIGCONTs the same process, and returns its successful exit without relaunching.
The source-fidelity module passed (**27 passed in 30.90 sec**). This verifies
the real Linux signal lifecycle under injected pressure; it does not apply
physical host/GPU pressure or validate hardware sensor response.

The complete Linux suite passed **18,627 tests in 132.49 sec** after adding the
real process-group differential. Both generated inventory checks, Python
compilation, `git diff --check`, and the immutable reference-snapshot check
passed.

### Source differential: standalone interrupt while paused (2026-10-23)

Added a source-executing comparison for `KeyboardInterrupt` during paused
standalone-wrapper polling. Both implementations record the same order:
pause the child, attempt to resume it, terminate its process tree, and return
status 130. This checks cleanup ordering independently of the signal-handler
path. The fixture uses a mocked child and deterministic pressure samples; the
real Linux STOP/CONT lifecycle is covered separately.

The source-fidelity module passed **28 tests in 31.80 sec**, and the complete
Linux suite passed **18,628 tests in 136.29 sec** after adding the paused-
interrupt differential. Both generated audit-map checks, Python compilation,
`git diff --check`, and the immutable reference-snapshot check passed.

### Emergency cleanup marker fidelity correction (2026-10-23)

A source-vs-generic review found that the emergency cleanup predicate accepted
any recognized marker for any generic entrypoint. The copied source requires
entrypoint-specific markers. Updated generic ownership matching so the launcher
and suite scripts require launcher CLI flags, the controller requires `--plan`,
and the worker requires `--spec`; mismatched flags no longer qualify a process
for force-kill. A six-case differential compares the source's launcher,
scheduler, and dispatcher marker rules with their generic launcher, controller,
and worker equivalents. The source differential, ownership contract, and
cleanup ordering/race tests passed (**4 focused tests**). The generated runner
function map was regenerated and its freshness check passed.

After the emergency-cleanup marker correction, the complete Linux suite passed
**18,629 tests in 134.09 sec**. Both generated inventory checks, Python
compilation, `git diff --check`, and the immutable snapshot test passed.

### Bundle export failure-order differential (2026-10-23)

Added source-executing differentials for failure while opening the temporary
gzip tar archive and while opening the second multipart output part. For tar
open, both implementations have written the manifest, removed old outputs,
and propagate the error without leaving a temp archive or new part. For the
second part failure, both leave the first eight-byte part and temporary
archive, have no monolithic archive, and propagate the same injected error.
The focused manifest-write, tar-open, and part-write failure differentials
passed (**3 tests**). This covers those injected failure points; it does not
prove every possible filesystem failure mode.


The full Linux suite after the multipart part-write failure differential passed **18,631 tests in 136.28 sec**. Both generated inventories, Python compilation, `git diff --check`, and the immutable reference-snapshot test passed. The pressure-aware coordinator and ADP implementation fidelity audits remain in progress; this run is test evidence, not an exhaustive fidelity sign-off.

### Controller persistence before pressure resume (2026-10-23)

Extended the source-main-versus-generic-coordinator host-pressure/external-PID
reopen trajectory to inspect persisted backoff/resource state inside the resume
child's `Popen` boundary. Both loops have already persisted the current external
host memory (400 MiB) and empty tracked-PID set before launching the resumed
child. The source differential passed (**1 test**). This confirms the
sample/gate-save/resume ordering for that trajectory; other controller
interleavings remain partial as noted in the ledger crosswalk.

The full Linux suite after the controller persistence-order assertion passed **18,631 tests in 135.74 sec**. Both generated inventories, Python compilation, `git diff --check`, and the immutable reference-snapshot test passed. The suite remains a regression gate; the incomplete controller interleaving and line-by-line source reconciliation recorded above are still open.

### Same-pass pressure close and external-drop reopen (2026-10-23)

Strengthened the source-main-versus-generic host pressure trajectory so the
tracked external PID disappears in the same sample that reports hard host
pressure. Both implementations first close the host gate from the pressure /
child rule, then reopen it from the qualifying 600 MiB external-memory drop;
at the subsequent pressure-pause boundary the persisted state has the gate
open and CPU launches enabled. The generic differential captures the exact
`(closed, disabled) -> (open, enabled)` transition, and the copied source logs
both transitions. The focused test passed (**1 test**). This verifies this
specific simultaneous transition, not every possible scheduler interleaving.

### Finalization log marker fidelity correction (2026-10-23)

A source review found the generic finish marker logged only status and timestamp,
while ledger §11's copied helper writes job ID, exit code, and reason. Updated
`SuiteCoordinator._finalize` to emit the source marker fields (using the
generic run ID in the job-ID position), then close the handle with the same
best-effort behavior. A source-executing failure-finalization contract checks
the source marker and generic identity-mapped marker byte-for-byte, including
the absence of a leading blank line; it passed (**1 test**).

Added an exact source/generic process-launch log-prefix differential covering
the leading blank line before a launch, attempt/device marker, command line,
sorted launch-context JSON, and flush-before-`Popen` ordering. The generic ID
is mapped into the source job-ID position. The focused contract passed (**1
test**); source/generic retry markers also remain covered by the retry
trajectory contracts. The source-main/generic retry trajectory now asserts the
exact retry marker in each log and that it precedes the second-attempt launch
marker; it passed as part of the 33-test source-fidelity module.

The full Linux suite after the simultaneous pressure-close/external-reopen differential passed **18,631 tests in 133.69 sec**. Both generated inventories, Python compilation, `git diff --check`, and the immutable reference-snapshot test passed. Broader controller interleavings and full ledger-to-code reconciliation remain open.

After regenerating the AST function map for the finalization log-marker edit, the complete Linux suite passed **18,632 tests in 135.69 sec**. Both generated inventory checks, Python compilation, `git diff --check`, and the immutable reference-snapshot test passed. This closes the identified finish-marker mismatch; broader source reconciliation remains in progress.

After tightening the finish-marker contract to compare exact bytes and regenerating the source-function map, the full Linux suite passed **18,632 tests in 136.88 sec**. Both generated inventory checks, Python compilation, `git diff --check`, and the immutable reference-snapshot test passed. Broader ledger reconciliation remains open.

The full Linux suite including the exact launch-prefix differential passed **18,633 tests in 136.17 sec**. Both generated inventory checks, Python compilation, `git diff --check`, and the immutable reference-snapshot test passed. This adds contract evidence for process-launch logging; broader ledger reconciliation remains open.

The full Linux suite including the exact retry-marker/order assertion passed **18,633 tests in 137.21 sec**. Both generated inventory checks, Python compilation, `git diff --check`, and the immutable reference-snapshot test passed. This adds source-comparison coverage for retry log sequencing; broader ledger reconciliation remains open.

### Process-tree termination source differential (2026-10-23)

Added a direct copied-source-versus-generic comparison for central process-tree
termination with three survivor patterns: no survivors after the first wait,
survivors that exit after the first kill, and a root that survives the second
wait and is killed again. All traces match for descendant reverse-termination,
root termination, wait sets and timeouts, survivor kill order, and final Popen
wait behavior. The focused Linux differential passed (**1 test**).

The launch-failure audit found that generic cleanup explicitly flushed the
log handle after `Popen` failed, while the copied launch helper directly calls
`close()`. Updated the generic exception path to call `close()` directly
(swallowing close errors) and added a source differential that verifies both
paths flush the complete launch preamble before `Popen`, then perform only the
direct close on failure. The focused contract passed (**1 test**).

The full Linux suite including the process-tree termination differential passed **18,634 tests in 137.21 sec**. Both generated inventory checks, Python compilation, `git diff --check`, and the immutable reference-snapshot test passed. This verifies the tested termination traces against copied source; wider line-by-line reconciliation remains open.

After regenerating the source-function map for the launch-failure cleanup correction, the complete Linux suite passed **18,635 tests in 139.70 sec**. Both generated inventory checks, Python compilation, `git diff --check`, and the immutable reference-snapshot test passed. The launch-failure behavior is now source-compared; broader ledger reconciliation remains open.

### Retry log cleanup exception boundary (2026-10-23)

Rechecked the retry block against the copied runner and found its inline
write/flush/close sequence differs from the general log-close helper: all
three source calls share one `try`, so a write or flush error skips later
calls. Changed the generic retry path to preserve that boundary and added
fault-injection contracts proving write failure skips flush/close and flush
failure skips close. The focused retry contracts passed (**4 passed**) and the
copied-source fidelity module passed (**35 passed**). Regenerated and checked
the function map. The full Linux suite passed **18,637 tests in 138.66 sec**;
`git diff --check` passed. This closes the retry cleanup edge identified in
this bounded comparison; broader ledger reconciliation remains open.

### Standalone wrapper signal-handler lifecycle (2026-10-23)

Compared standalone-wrapper shutdown registration with the copied Linux
source. The generic wrapper restored previous handlers after `run()`, while
the source installs SIGINT/SIGTERM/SIGHUP/SIGQUIT handlers for the process
lifetime and clears its active child handle at exit. Updated the generic
wrapper to retain those four process-lifetime handlers and clear its child
reference in `finally`. A source-executing contract compares the registered
signals and verifies neither path queries/restores prior handlers. It passed
(**1 test**); the standalone pressure module passed (**25 passed**). This
closes that lifecycle difference for the Linux wrapper path; the broader
source and algorithm audits remain open.

The full Linux suite after the retry-cleanup and standalone-handler changes
passed **18,638 tests in 139.45 sec**. The generated runner function map,
458-block ADP crosswalk, Python compilation, and `git diff --check` passed.
This is regression evidence for the current checkout; full ledger-to-code and
all-settings ADP fidelity reconciliation remain open.

### Post-launch hold log parity (2026-10-23)

The controller-cycle comparison found that the generic scheduler persisted
the per-device hold and slept for the configured delay but omitted the source
`[HOLD]` log event. Added the exact source message and arguments immediately
after controller-state persistence and before the hold sleep. Extended the
lane-cap/dual-hold contract to assert each log record's format and values and
that each is followed by the configured hold sleep. The focused controller
contract passed (**1 test**); generated inventories and the full Linux suite
will be refreshed after this audit batch. Other source/controller details
remain under review.

The launch boundary comparison also found the generic controller omitted the
source `[RESOURCE]` record immediately before `launch_job`. Added that record
with the source values for queue/active/capacity, host and GPU memory, latest
child footprint, and remaining per-device holds. The existing controller
contract now checks the resource record, its position before `[HOLD]`, and the
hold log before the full configured sleep. The focused contract passed again
(**1 test**). This covers the pre-launch logging boundary and does not close
the remaining controller, bundle, recovery, or algorithm audit.

After the `[RESOURCE]` and `[HOLD]` logging additions, the full Linux suite
passed **18,638 tests in 147.34 sec**. The generated function map, ADP
crosswalk, Python compilation, and `git diff --check` passed before the run.

### Successful pressure-resume event reason (2026-10-23)

Compared the full paused-job resume trajectory's persisted events and found
the generic coordinator stored the original pause cause as the resume-event
reason. The copied controller records `gate_reopened`; it carries the old
pause cause separately in launch context. Updated the generic event to use
`gate_reopened` and extended the source/generic run comparison to assert both
resume rows. The focused differential passed (**1 test**). This fixes the
persisted event contract for successful pressure resumes; other event and log
comparisons remain open.

The same cycle review found source-shaped operational event records absent
for successful launches, retries, pressure pauses, and resumes. Added a shared
event path that emits the contextual event log before persisting the generic
event row; launch failures now log their exception at the retry boundary, and
resume-launch failures emit the source warning while keeping the event row
separate. Finalization remains a state/progress event without a new source
event log. The controller contract pins launch event fields and ordering
alongside `[RESOURCE]` and `[HOLD]`; the source/generic paused-job trajectory
checks both `gate_reopened` rows. Both focused checks passed (**2 tests**).

The full Linux suite after the successful-resume event correction and
source-shaped event logging passed **18,638 tests in 141.81 sec**. The runner
function map, ADP block crosswalk, compilation, and `git diff --check` passed
before the run. This is regression evidence only; gate-transition log parity
and the broader line-by-line runner and ADP audits remain open.

### Gate close/reopen transition logs (2026-10-23)

Compared the reference sampler’s gate block with `SuiteCoordinator._sample_snapshot`.
The generic implementation changed gate state without emitting the source
transition records. It now logs newly closed GPU and host gates with the source
fields, followed by external non-Python memory-drop reopen records in source
order. A focused same-sample contract verifies both gates close and reopen and
asserts the four log records occur as GPU close, host close, GPU reopen, host
reopen. The focused resource tests passed (**2 tests**, including the existing
latest-child gate contract). The full Linux suite then passed **18,639 tests in
140.05 sec**. The runner function map was regenerated; both generated
inventories, compilation, `git diff --check`, and the immutable reference
snapshot test passed. Broader source reconciliation remains open.

### Successful hard-pressure pause summary log (2026-10-23)

The reference emits a warning after it terminates and records a pressure-paused
job, including job ID, combined reason, active/suspended counts, host available
memory, active RSS, and GPU free/used memory. The generic pause path recorded
the state/event but omitted this warning. Added the source-shaped warning after
the pause event and before saving controller state. The focused contract checks
the exact formatted message and post-pause counts; it and the real-child
source/generic pressure-resume comparison passed (**2 tests**). Full-suite
verification initially exposed minimal synthetic snapshots without the new
resource fields. Log formatting now falls back to `-` for absent fields while
retaining exact source values for real `ResourceSnapshot`s. The pressure-victim
matrix passed (**19 tests**), and the full Linux suite passed **18,639 tests in
141.21 sec** after the correction. The runner function map, ADP crosswalk,
compilation, diff check, and immutable snapshot check passed.

The same exit-path comparison found the source's `[GATE] admission reopened`
record missing when completion, retry, or terminal failure releases gates.
Added one shared release helper that logs the exact device-derived lane list
after mutating gate state. CPU releases `host`; GPU releases `gpu,host`; unknown
device modes emit no release record. Focused retry contracts verify both lane
values and that the release record precedes the retry record; the success and
terminal paths share this helper. Focused release/pause checks passed (**4
tests**); the full Linux suite passed **18,639 tests in 143.11 sec** after this
addition. The runner function map was regenerated; both generated inventories,
compilation, diff check, and immutable snapshot check passed.

### Recovered-child requeue warning (2026-10-23)

The reference emits `[RECOVER] child disappeared after orchestrator restart;`
after requeuing a recovered process whose real exit status is unavailable.
The generic branch already persisted `recovered_child_exit_status_unknown`
and appended the job to the FIFO queue, but omitted this warning. Added the
source-formatted record after state/event persistence. The full-pass FIFO
requeue contract and source/generic recovered-child comparison passed (**2
tests**). The full Linux suite passed **18,639 tests in 142.23 sec**. The runner
function map was regenerated; both generated inventories, compilation, diff
check, and immutable snapshot check passed.

### Periodic status log parity (2026-10-23)

The reference emits a 30-second `[STATUS]` line containing queue/live counts,
completion/failure/retry counters, cumulative and live telemetry counts,
batch scale, lane/gate state, remaining holds, sampled host/GPU resources,
latest child footprints, and launch capacity. The generic coordinator wrote a
different JSON-formatted log alongside its `status.json`. Replaced that log
record with the reference format and fields while keeping `status.json` as the
generic status artifact. A focused contract checks the complete rendered line
and confirms a second call inside the 30-second window emits no duplicate.
That check passed (**1 test**). The full Linux suite passed **18,640 tests in
141.65 sec**. The runner function map was regenerated; both generated
inventories, compilation, diff check, and immutable snapshot check passed.

The scheduler logging comparison also found the source's `[INTERRUPT]`
shutdown warning absent from generic signal/KeyboardInterrupt cleanup. Added
the warning before process-tree termination and state persistence. The
controller-pass shutdown contract now asserts the exact signal-specific line;
the status and shutdown focused contracts passed (**2 tests**). The full Linux
suite passed **18,640 tests in 145.15 sec**. The runner function map was
regenerated; both generated inventories, compilation, diff check, and
immutable snapshot check passed.

The launcher comparison additionally found the source module's top-level
`KeyboardInterrupt` notice missing from the generic entry point. Added
`cli_main`, which prints the source-shaped resumability notice and returns
status 130; both `python -m simple_adp.runner.launcher` and the installed
`simple-adp-suite` entry point use it. A focused launcher contract and the
coordinator signal-restoration contract passed (**2 tests**). This closes that
specific observed difference; it does not certify every launcher behavior.

### Linux watchdog source comparison (2026-10-23)

Rechecked the generic detached watchdog against the ledger's Bash orphan
watcher. Matched its orphan log record (`[WATCHDOG] master ... gone; cleaning
python children`) and the launcher's normal-success cleanup, which removes the
watchdog log only after a zero child exit. Nonzero exits retain the log, as the
source's `set -e` path skips its cleanup statements. Added focused assertions
for both outcomes and a real Linux master-death test that verifies TERM/KILL
cleanup of an orphaned parent/child tree with shortened test timings. The
watchdog and launcher focused checks passed (**7 tests**). The test exercises
process cleanup on Linux; it does not simulate every race during descendant
creation or prove host-level behavior under forced PID reuse.

After these changes, the complete Linux suite passed (**18,644 passed in
144.89 sec**). Compilation, `git diff --check`, the runner function map check,
and the ADP guide block crosswalk check also passed. No OpenML campaign or
physical pressure workload was launched.

### Linux launcher environment-before-Torch ordering (2026-10-23)

A fresh-process import trace showed that `runner.launcher` imported
`runner.scheduler`, whose telemetry dependency imported Torch, before
`_apply_runtime_environment_defaults` ran. The top-level plugin import also
loaded the estimator/Torch early. That meant configured CUDA visibility,
allocator configuration, and cuBLAS workspace settings could arrive after
Torch initialization. Removed those import-time Torch dependencies from the
coordinator path: telemetry now imports Torch/checkpoint helpers only inside
the checkpoint operations that need them, and the launcher loads registries
after environment defaults and runtime bootstrap. A fresh-process contract
wraps Python's first Torch import and verifies all three configured values
have already been applied; it passed. Focused launcher and telemetry checks
passed (**74 passed in 11.35 sec**). This fixes import ordering for
the suite entry point; direct imports of estimator/model modules still load
Torch as expected.

After the import-order change, the complete Linux suite passed (**18,645
passed in 146.45 sec**); compilation, `git diff --check`, and both generated
inventory checks passed.

### Linux fidelity audit continuation (2026-10-23)

Re-read the standalone pressure-wrapper implementation against ledger §9 and
the copied Linux source path, including pressure sampling, Python process
accounting, pause-boundary failures, paused-child exit/relaunch, signal and
KeyboardInterrupt handling, and command exit propagation. No additional
mechanism or value discrepancy was found in that bounded slice. This review
does not certify physical pressure response or close the remaining controller
and recovery interleaving audit items.

The combined ADP algorithm, model-route, end-to-end, standalone pressure, and
source-fidelity tests passed (**754 passed in 55.30 sec**). The complete Linux
suite then passed (**18,645 passed in 147.05 sec**). Runner and ADP generated
inventories, Python compilation, and the reference-snapshot integrity test
passed. No OpenML campaign or hardware pressure workload was run. These are
regression results; the open fidelity boundaries above remain open.

### Registered-route worker smoke matrix (2026-10-23)

Added a worker-level smoke contract that serializes and reconstructs every
route from the built-in model adapter, including canonical identities, both
generic profiles, the aggregate fixed grid, all 50 individual vanilla
architectures, and all configured diagnostic screen seeds/architectures. Each
route is executed for both classification and regression fixtures through
`worker.run_spec`; the worker override is asserted to use one epoch and a batch
equal to the six-row training partition. The matrix passed **292 route/task
cases in 21.78 sec**. It verifies the worker and adapter path on synthetic
data, not every live task from the OpenML providers or physical GPU execution.

The refreshed complete Linux suite passed **18,937 tests in 166.73 sec** with
the matrix included. The expanded smoke coverage closes the prior worker-path
route-count gap; remaining controller/recovery interleavings and live provider
and hardware execution boundaries remain open.

### Startup recovery source differential (2026-10-23)

Added a direct source/generic comparison for startup reconciliation across 66
persisted-state cases: eight statuses (`running`, `launching`, `retrying`,
`paused`, `queued`, `failed`, `completed`, and unknown), saved PID present or
absent, saved PID live or stale, and exact-argv match or no match. The compared
outcome is the coordinator placement (`active`, `suspended`, `queued`,
`completed`, or terminal), including source behavior where paused recovery
clears the saved PID before exact-argv lookup and completed zero-exit state
completed, or terminal), including source behavior where paused recovery
clears the saved PID before exact-argv lookup and completed zero-exit state
prevents stale-process recovery from requeueing the route. Two extra cases
verify terminal exact-resume precedence over a live PID, translating the
source `reason` field to the generic state’s `last_error` field. The focused
differential passed (**1 test, 66 cases, 1.26 sec**). This closes that bounded
startup-classification slice; persisted-field edge values, process-table
races, arbitrary command collisions, and full source-line reconciliation
remain open. Before the two added terminal cases, the complete Linux suite
passed (**18,938 tests in 167.66 sec**) and focused process/recovery and
source-fidelity modules passed (**54 tests in 40.09 sec**). After adding the
terminal cases, those focused modules passed again (**54 tests in 39.96 sec**)
and the complete Linux suite passed (**18,938 tests in 166.64 sec**). Runner
function map, ADP guide block crosswalk, Python compilation, reference-snapshot
integrity, and `git diff --check` all pass on the current checkout. No dataset
campaign or hardware pressure run was started.

### Runtime-tuning systemd differential (2026-10-23)

Added direct comparison of the Linux systemd re-exec command built by the
generic runtime-tuning module and the copied runner source. Four mocked version
outputs cover systemd 252, systemd 253, malformed systemd version text, and a
non-systemd first line. The comparison checks the complete argv, exported scope
environment, CPU allow-list, and whether `MemoryZSwapMax=0` is included. All
four cases matched. Runtime-tuning and source-fidelity modules passed together
(**76 passed in 40.22 sec**) using the repository `.venv`; an initial attempt
with system Python could not import project dependencies and is not test
evidence. This closes only the systemd command differential; the broader
controller interleaving and full ledger-to-code reconciliation remain open.

### Combined-pressure victim-pool edge differential (2026-10-23)

Added an end-to-end source/generic controller comparison for simultaneous host
and GPU hard pressure when the only live child is CPU-mode. The ledger gives
GPU pressure precedence when choosing the pressure victim pool; with no GPU
candidate, the CPU child remains active, completes on the next healthy sample,
and is not pressure-paused. The copied controller and generic coordinator
matched in status, attempt, pause count, and event outcome (**1 focused test
passed**). This covers one previously unpaired controller branch; it does not
close all multi-child controller interleavings.

After the no-GPU-candidate comparison was added, the source-fidelity module
passed (**40 passed in 42.03 sec**) and the full Linux suite passed (**18,941
passed in 170.61 sec**). Compilation and both generated inventory checks also
passed. No live pressure workload or dataset campaign was run.

Added a second controller comparison for CPU completion while a separate GPU
child keeps the GPU soft gate closed. Both source and generic runners launch
the queued route on CPU. Reinspection found that the source only re-enables
`gpu_launches_enabled` for a successful GPU child; CPU completion leaves that
flag unchanged while always enabling CPU launches. The earlier comparison
captured the gate-close save rather than the CPU-completion save, so its claim
that both flags were enabled was incorrect. The differential test now captures
the actual completion save, and `_finalize` follows the source's device-specific
flag behavior. The full suite result above predates this correction; rerun the
focused contract and full suite before treating it as validated. Full controller
interleavings and line-by-line reconciliation remain open.

### CPU backend seeding differential (2026-10-23)

Added a source-executing Linux contract for the child backend seed path. It
compares explicit/environment/fallback seed resolution and the resulting
Python, NumPy, and Torch random sequences under CPU admission with accelerator
probing held deterministic. The generic implementation matched the copied
backend source (**1 test passed**). This verifies the CPU seed path only; GPU
generator and optional accelerator runtime behavior remain hardware-dependent.

After adding both runtime source comparisons, the backend and source-fidelity
modules passed together (**65 passed in 42.51 sec**) and the complete Linux
suite passed (**18,940 passed in 168.33 sec**). Python compilation, the runner
function-map check, and the ADP guide block-crosswalk check passed. The full
controller-interleaving and line-by-line guide/ledger audits remain open.

### CPU completion lane-flag source correction (2026-10-23)

Rechecked the successful-exit branch in the copied Linux controller after the
source/generic completion comparison exposed that the earlier ledger sentence
was wrong. The source releases device-mode gates, then sets
`gpu_launches_enabled=True` only for a GPU-mode completion and always sets
`cpu_launches_enabled=True`. The source differential now captures the persisted
state after the CPU child has actually completed (rather than the earlier
gate-close save); the generic `_finalize` path and its unit/integration
contracts now match this distinction. Updated the ledger's two completion
descriptions and the implementation-spec summary. The source differential and
three coordinator contracts passed (**4 passed**). The first post-change full
suite caught one stale GPU-terminal-failure assertion and the generated
function map; after correcting the assertion and regenerating the map, the
complete Linux suite passed (**18,942 passed in 168.87 sec**). The 46-file
reference snapshot integrity test passed, as did Bash syntax, runner function
map check, ADP guide block-crosswalk check, Python compilation, and
`git diff --check`. This resolves the lane-flag discrepancy; exhaustive
controller interleavings and full line-by-line runner/algorithm reconciliation
remain open. No dataset campaign or physical pressure workload was run.

### Scheduler-label CPU-hold source correction (2026-10-23)

Compared `select_launch_device` in the copied controller with
`runner.admission.choose_device` and traced the booleans supplied by the
controller loop. The source's `fixed` and `gpu_first` labels select GPU when
the GPU predicate is ready, even if the CPU lane is held. `pressure_aware`
selects GPU only if both GPU and host predicates are ready; because host
readiness includes the CPU post-launch hold, it returns no device in the
GPU-ready/CPU-held state. The generic selector had treated all three labels
the same. Updated it to preserve this source distinction, corrected the
repository ledger's selector summary and loop description, and replaced the
old same-label assertion with explicit ready/held cases. The source-executing
differential, three label contracts, and existing full-controller CPU-hold
contract passed (**5 passed**). This behavior conflicts with the supplied
plan's shorthand that all labels select GPU-first; source behavior and the
ledger's selector pseudocode govern this edge, and the discrepancy is recorded
here rather than silently preserving the inaccurate shorthand. The full Linux
suite passed (**18,943 passed in 174.62 sec**). Function-map and ADP crosswalk
checks, compilation, and `git diff --check` passed. The audit remains open for
other source/ledger and implementation gaps, controller interleavings outside
the tested domains, and live hardware behavior.

The selector differential was then expanded from representative CPU-hold
cases to a 55,296-state source/generic comparison over the valid discrete
selector inputs (all three labels, CPU-only and inherited masks, host/GPU
readiness, total/GPU caps, active counts, lane flags, both gates, and separate
CPU/GPU holds). All states matched the copied selector. The complete Linux
suite passed again with this expanded matrix (**18,943 passed in 173.92 sec**).
This bounded selector truth table strengthens that helper comparison; it does
not prove the entire controller loop for arbitrary timelines or live resources.

### Child telemetry CSV source correction (2026-10-23)

Compared the copied `TrainingTelemetry._build_csv_columns` and
`save_stage_state` path with `ChildTelemetryWriter`. The source creates a
separate RNG companion and records its name in JSON, but its current CSV schema
omits the active-stage flag, architecture vector, exact batch cursor, stage
checkpoint path, and RNG companion name. The generic writer had included these
fields in CSV, making fallback recovery more capable than the source. Removed
them from the default generic stage CSV schema and stage rows while preserving
the JSON sidecar and signed estimator checkpoint state. Added both a behavioral
contract for JSON-sidecar loss and a source-AST contract that checks the copied
CSV schema omits the exact-resume fields. Focused telemetry tests passed
(**30 passed**). The full suite had passed **18,944 tests in 174.25 sec** after
the earlier RNG-column correction; it is being rerun after this broader schema
alignment. This closes the identified CSV-schema mismatch, not the broader
runner or ADP fidelity audit.


After the stage-CSV schema was aligned with the copied source, the complete
Linux suite passed **18,945 tests in 172.65 sec**. The targeted telemetry group
passed **30 tests**; the source function map, ADP guide crosswalk, compilation,
and whitespace checks passed. This is a bounded source correction and suite
regression result; remaining partial ledger rows and live-resource evidence
still need their own audit.

The explicit model/worker smoke and telemetry group was rerun on the current
checkout: `.venv/bin/python -m pytest -q tests/test_smoke_suite.py
tests/test_runner_worker.py tests/test_runner_child_telemetry.py` — **369
passed in 21.96 sec**. Smoke settings remain one epoch and one batch covering
the full training partition; this does not run a benchmark campaign or a GPU
workload.

### Selector conflict cleanup (2026-10-23)

A reread found that ledger Section 13 still said all scheduler labels were
identical, contradicting its detailed Section 5 and Section 14 rules and the
copied selector. Corrected that ledger entry. The saved implementation
proposal also repeated the shorthand, so retained its original proposal text
and appended a source-fidelity addendum that supersedes that statement. The
source-executing 55,296-state selector differential and targeted label
contracts passed again (**4 passed**); both generated inventories and
`git diff --check` passed. The corrected selector behavior remains the one
implemented by `runner.admission.choose_device`.

### Linux watchdog master-liveness edge (2026-10-23)

Compared `_orphan_watchdog.sh`'s `kill -0 "$MASTER_PID"` loop with
`runner.watchdog.master_alive`. The Python implementation had used psutil's
`is_running()` plus a non-zombie check, which considers an unreaped zombie
dead while the shell's `kill -0` still succeeds. Replaced that check with
`os.kill(pid, 0)`, treating `OSError` (including permission failure) as not
alive, matching the shell predicate. Added a contract for successful probe
and `EPERM` behavior. The focused Linux watchdog/launcher tests passed (**8
passed, 40 deselected**). The watchdog still uses psutil to snapshot and
signal descendants; only the master-liveness predicate is covered by this
comparison. The broader source-to-ledger audit remains open.

After the master-liveness and signal-cleanup changes, the complete Linux test
suite passed (**18,953 passed in 176.87 sec**). Both generated inventories and
`git diff --check` passed.
This is regression evidence for the watchdog correction, not completion of the
remaining source-to-ledger or ADP guide audits.
The source-map row for static contract tests still carried an earlier full-suite
count; it now points to this latest run.

### Linux launcher signal cleanup (2026-10-23)

Compared the Linux launcher path to `_orphan_watchdog.sh`'s INT/TERM/HUP/QUIT
trap. The generic launcher previously had no immediate signal handler and
depended on detached orphan detection, delaying child cleanup by the watchdog
poll interval and 15-second grace. It now installs/restores those handlers on
the main thread. A received signal terminates descendants then the controller,
waits the fixed two seconds, rescans and kills remaining descendants/controller,
kills the detached watcher, and exits with `128 + signal`. The new process-tree
helper performs the second tree walk so children created during the grace
period are included. Focused launcher/process/watchdog tests passed (**67
passed**), including signal cleanup ordering, watcher cleanup, and discovery of
a late descendant in the KILL pass. Real Linux process-tree contracts passed
for both direct fixed-grace cleanup and launcher SIGTERM handling: a controller
and its grandchild were terminated, and the launcher returned `128 + SIGTERM`.
The full suite passed after this change (**18,953 passed in 176.87 sec**);
this still covers only the Linux outer-launcher signal path, not every
runner/source contract.
A separate Linux integration test launched a real parent/grandchild pair and
verified that the fixed-grace helper terminated both (**1 passed**).

### Linux child-runtime and process fidelity refresh (2026-10-23)

Rechecked the Linux runtime-tuning source and generic port, including CPU
frequency ordering, affinity fallback and partitioning, SHA-1 slot selection,
positive concurrency-hint precedence, thread and worker budgets, shared-CPU
environment values, POSIX priority attempts, `mlockall`, systemd-scope
properties and re-exec ordering. The generic child builder's backend/device
fields are covered separately by the source `make_child_env` differential;
they are task/backend adaptation fields, while the shared CPU controls match
the copied helper. No new mismatch was found in this bounded pass. Ran the
focused source-fidelity, runtime-tuning, process, watchdog, and launcher tests:
**151 passed in 52.46 sec**. These are Linux tests with mocked resource and
device state plus process-tree integration fixtures; they are not a physical
GPU or host-pressure certification. The complete Linux suite then passed
**18,953 tests in 178.51 sec**. Both generated inventories and `git diff
--check` passed. Remaining partial ledger rows, arbitrary controller
timelines, and live resource behavior remain unaudited or unverified.

### Standalone pressure-wrapper CLI error (2026-10-23)

Compared the empty-command path in the copied Linux
`pressure_aware_wrapper.py` with `runner.pressure_wrapper.main`. The source exits
with the message `No command provided`; the generic CLI previously raised an
uncaught constructor `ValueError`. Added the source-compatible CLI exit while
retaining `ValueError` for direct construction of an empty `PressureAwareWrapper`.
The direct source/generic error comparison and CLI contract passed (**2
passed**). The broader standalone wrapper remains covered by its source
differentials for gate rules, samplers, signal registration, and real
process-group pause/resume.

The CLI validation paths also now match the source's specific failures for
invalid host/GPU percentage ordering, nonpositive external-drop threshold, and
nonpositive poll interval. The source-versus-generic invalid-settings
differential plus standalone pressure-wrapper tests passed (**27 passed**).

### Source differential: job-grid order, repeat asymmetry, and baseline grid (2026-10-23)

Added a source-executing contract that calls the copied scheduler's
`build_suite_jobs` and compares its emitted route sequence with `SuitePlan`
for a fixture provider and adapter. Seven configurations cover the four ADP
surfaces, configured surface flags, `all`/`active`/`backlog`, unknown tasks,
zero outer repeats, ADP repeats both below and above `repeat_count`, post-plan
`job_start_index`/`job_limit` slicing, and vanilla emission across every outer
repeat. The comparison checks repeat, canonical route identity, depth/width,
and ordering. It confirms the individual vanilla routes match all 50 source
coordinates and separately asserts that the generic plan's additional
aggregate fixed-grid route carries the full 50-architecture grid. The focused
Linux source differential passed (**1 passed**); it uses a fixture provider
and does not download OpenML data or execute models. No grid mismatch was
found in these cases. Other partial rows and live resource behavior remain
open. The complete Linux suite passed after adding this differential:
**18,954 passed in 179.89 sec**. The runner function inventory, ADP block
crosswalk, and `git diff --check` also passed.

### Source correction: child stage-sidecar RNG recovery (2026-10-23)

Reconciled ledger §21's `TrainingTelemetry.load_stage_state` and
`_restore_stage_rng` behavior with `ChildTelemetryWriter`. The generic reader
previously returned legacy stage records without restoring their RNG companion,
and returned CSV fallback rows even when the copied loader would fail exact
resume because the companion field was absent. It now restores RNG for legacy
sidecars without architecture/checkpoint ownership, defers to a complete
estimator checkpoint for an active stage, and raises the documented
exact-resume error when a recovered row lacks a safe companion. Added tests for
Python/Torch RNG restoration, active-checkpoint deferral, default CSV recovery
failure, and legacy-header behavior. Focused telemetry verification passed
(**32 passed in 1.76 sec**). The full Linux suite then passed (**18,956 passed
in 178.54 sec**); the runner function inventory and ADP block crosswalk checks
passed, as did `git diff --check`. The broader source audit remains open.

### Combined algorithm and Linux source-differential audit refresh (2026-10-23)

The user's canonical `ADP_algorithms.docx` and the repository copy have the
same SHA-256. The focused algorithm, checkpoint/artifact, guide-crosswalk,
and copied-runner source-differential group passed **594 tests in 54.82 sec**.
The generated 458-block ADP crosswalk and runner function map (272 definitions)
both pass their freshness checks, and `git diff --check` passed. No mismatch
was found in this bounded pass. This is not a proof over arbitrary scheduler
timelines or unbounded model settings, and it does not exercise physical GPU or
host-memory pressure. The broader source-to-ledger audit remains open.

### Multi-pass OOM and hard-pressure overlap differential (2026-10-23)

Added a source-executing trajectory that first admits four GPU jobs, then
causes one job's CUDA OOM retry during a pass with simultaneous hard host and
GPU pressure. It verifies the controller pauses the highest architecture
priority peer for OOM, then chooses a distinct footprint-ranked pressure
victim, resumes both on GPU when resources recover, retries the failed job,
and completes the route set. It also confirms the remaining route falls back
to CPU while the GPU soft gate is closed. Source and generic runs matched
launch counts, pause order, device modes, pause counters, retry/attempt counts,
completion and transient backoff reset. The complete copied-runner
differential module passed (**46 passed in 50.65 sec**); the new trajectory
and immutable snapshot contract passed together (**2 passed**). The
runner function inventory, ADP block crosswalk and `git diff --check` passed.
This adds one coupled timeline to the bounded controller evidence; arbitrary
interleavings and physical pressure remain unverified.

### Linux regression refresh after the OOM/pressure trajectory (2026-10-23)

Reran the focused Linux group for copied-runner source differentials, model
routes, and the ADP guide crosswalk: **196 passed in 53.71 sec**. The complete
Linux suite passed **18,957 tests in 179.49 sec**. The 272-definition runner
function inventory and 458-block ADP crosswalk passed freshness checks;
`git diff --check` and the immutable runner snapshot test passed. No model or
coordinator implementation changed in this refresh. Passing the suite does not
close arbitrary controller interleavings, unbounded settings/custom callbacks,
or live GPU/host-pressure behavior; those fidelity boundaries remain open.

### Reference-snapshot independence guard (2026-10-23)

Added an AST-based contract that scans every Python module under
`src/simple_adp` and rejects imports or literal paths that bind runtime code to
`references/opf_adp_runner_suite`. The copied snapshot integrity test remains
separate and still verifies its 46-file manifest. Both reference contracts
passed; the complete Linux suite with the new guard passed **18,958 tests in
182.85 sec**. The source and model implementations were unchanged. This guards
the fresh-package architecture while preserving the copied files as a static
comparison oracle.

### Real worker pressure-pause and checkpoint-resume integration (2026-10-23)

Added an end-to-end coordinator test using a registered fixture provider and
model adapter that delegates training to the real `TabularADP` implementation.
An injected host sampler raises hard pressure after the worker writes its
estimator checkpoint. The coordinator terminates the first worker, resumes the
same route on CPU in a second process, and the estimator completes from the
checkpoint. The fixture pauses briefly inside its progress callback after the
per-batch estimator checkpoint is committed; the injected pressure sampler
reads that authoritative checkpoint and records its positive batch cursor.
The relaunched adapter reads the same cursor from the signed checkpoint before
training continues. Assertions cover one pressure pause and resume, two
launches, zero retry count, the same CPU lane, exact cursor reuse, and a
completed result. The focused test passed (**1 passed in 9.95 sec**), and the
complete Linux suite passed (**18,959 tests in 191.48 sec**). This verifies
the worker/checkpoint/controller handoff under deterministic synthetic
pressure; it does not exercise physical host pressure or every possible
interruption timing.

### Typed provider payload and metadata handoff (2026-10-23)

Added the public `TabularTaskData` protocol for provider payloads, including
features/targets, dataset identity and digest, feature schema, task kind,
optional task adapter, and dataset metadata. The OpenML `TaskData` object now
exposes `task_kind` and returns a fresh built-in `TaskAdapter` for each worker
route. Planning validates that provider metadata is a mapping and carries it
into the signed `DatasetTaskRef`/run specification; the worker persists the
metadata in its result record. `DatasetTaskRef`, `TabularTaskData`, and
`SplitSpec` are public lazy exports from `simple_adp.runner`. Contracts verify
the OpenML payload satisfies the protocol, fresh target adapters are returned
for separate routes, plugin metadata reaches planned specs and worker results,
and non-mapping provider metadata is rejected. The focused provider, OpenML
plan, and worker tests passed (**303 passed in 21.76 sec**). The complete Linux
suite passed (**18,961 passed in 188.34 sec**); runner
function inventory, ADP guide crosswalk, and `git diff --check` passed. This
strengthens the provider extension contract and provenance handoff; it does
not change registered model algorithms or establish live OpenML execution.


### Full Linux suite refresh after ledger reread (2026-10-23)

Reran `.venv/bin/python -m pytest -q` against the current checkout: **18,961
passed in 190.17 sec**. The immutable source-snapshot and ADP block-crosswalk
contracts also passed (**3 tests**); both generated inventories are current and
`git diff --check` is clean. This updates regression evidence only. The partial
ledger sections and live hardware-pressure boundary remain open as stated in
the section map.


### Nested Linux outer-watchdog signal traversal (2026-10-23)

The ledger's shell watchdog recursively signals each direct child's descendants
before signaling that child, waits the fixed grace interval, then performs a
fresh recursive KILL traversal. `terminate_process_tree_after_grace` previously
reversed a flat `psutil.children(recursive=True)` listing, which does not
preserve postorder for nested process trees. It now recursively visits direct
children (`recursive=False`) and signals each node postorder on both TERM and
the post-grace KILL rescan. The new contract includes a nested leaf/parent and
a descendant that appears during the grace interval. The focused watchdog and
process group passed (**5 passed**), and the complete Linux suite passed
**18,961 tests in 187.94 sec**. The runner function map was regenerated and
checked; the ADP crosswalk and `git diff --check` passed.


### Executing nested outer-watchdog source differential (2026-10-23)

Added a differential contract that sources the copied `_orphan_watchdog.sh`,
invokes its nested `_watchdog_kill_process_tree` helper against a deterministic
nested tree, then compares its TERM/KILL signal sequence with
`terminate_process_tree_after_grace`. The second traversal adds a new child
before KILL, verifying the source's fresh tree scan. The runner source-fidelity,
watchdog, and process contracts passed (**51 passed in 52.03 sec**). The full
Linux suite passed **18,962 tests in 190.43 sec**. Both generated inventories
and `git diff --check` passed. Other controller interleavings and live hardware
pressure remain outside this bounded differential.

### Full Linux suite refresh after audit status check (2026-10-23)

Reran `.venv/bin/python -m pytest -q` against the current Linux checkout:
**18,962 passed in 189.17 sec**. `scripts/generate_runner_function_map.py
--check`, `scripts/generate_adp_block_crosswalk.py --check`, and
`git diff --check` all passed. This is current regression evidence for the
registered ADP configurations, vanilla grids, runner contracts, and their
synthetic/source-differential tests. It does not upgrade the runner ledger's
partial source-parity status: source execution comparisons cover selected
trajectories, while the 16,128-case controller matrix is synthetic; real host
and GPU pressure behavior remains unverified.

### Source differential: active runtime with missing process handle (2026-10-23)

Added `test_source_main_and_generic_drop_active_route_with_missing_process_handle`.
It starts the copied runner's real `main()` and the generic coordinator with one
already-active runtime whose process handle is absent. Both controller loops
remove the runtime from the active set without manufacturing a completion,
failed attempt, retry, or exit code. The focused source differential passed
(**1 passed**). This closes that specific source-comparison gap; it does not
expand to every controller interleaving or establish physical pressure parity.

### Full Linux suite after missing-handle source differential (2026-10-23)

The full suite passed **18,963 tests in 191.68 sec** after adding the active
runtime/missing-process-handle source comparison. The test uses synthetic
controller state and CPU-only execution; all remaining source-parity and live
hardware limitations above remain in force.

### Source differential: central process-tree termination escalation (2026-10-23)

Added `test_source_and_generic_process_tree_termination_trace_match`. It invokes
the copied scheduler's `_terminate_process_tree` and the generic
`terminate_process_tree` with identical nested descendants and controlled
survivor sets. TERM order, each wait set and timeout, survivor KILL order, the
second wait, and final root wait match exactly. The focused comparisons for
this path and the missing-handle controller path passed (**2 passed**). This
covers one deterministic escalation trace; process races and OS-level failures
remain represented by separate Linux integration/contracts.

### Full Linux suite after central termination differential (2026-10-23)

The full Linux suite passed **18,964 tests in 191.12 sec** after adding the
central termination trace differential. Generated inventory checks and
`git diff --check` are rerun below; source parity remains bounded to the
contracts enumerated in this specification.

### Source differential: central termination failure boundaries (2026-10-23)

Added `test_source_and_generic_process_tree_termination_failure_boundaries_match`.
For an already-exited root, a root-PID lookup failure, and an unexpected
recursive-child enumeration error, the copied source and generic helpers have
the same effects: immediate return; best-effort root-handle terminate without
wait; or propagated enumeration exception before signaling, respectively.
The focused failure-boundary and normal-escalation tests passed (**2 passed**).

### Full Linux suite after termination-boundary differential (2026-10-23)

The full Linux suite passed **18,965 tests in 193.77 sec** after adding the
central termination early-return and exception-boundary comparison. The normal
escalation and failure-boundary source differentials passed; generated maps and
`git diff --check` are current. Controller source parity remains bounded, and
live hardware pressure remains unverified.

### Source differential: best-effort signal failures (2026-10-23)

Added `test_source_and_generic_tree_cleanup_swallow_signal_failures_and_continue`.
The copied scheduler and generic helper are exercised with TERM and KILL
methods that raise, while `wait_procs` reports survivors and then an empty set.
Both implementations swallow each signaling error, continue to the next
cleanup step, issue the same two waits, and perform the same final root wait.
The focused termination comparisons passed (**3 passed**).
The full Linux suite then passed **18,966 tests in 199.96 sec**. Generated
inventory checks and `git diff --check` also passed. This remains synthetic
process-helper evidence; live pressure and hardware behavior are unverified.

### Source differential: reap before hard-pressure victim selection (2026-10-23)

Added `test_source_and_generic_reap_success_before_hard_pressure_victim_selection`.
The copied `main()` loop and generic coordinator each run a child that succeeds
on the same sample where host available RAM falls below the hard-pressure
threshold. Both record the pressure sample, reap and complete the child before
victim selection, and emit no pause event (`pause_count=0`). Focused controller
and process-tree comparisons passed (**3 passed**).
The full Linux suite then passed **18,967 tests in 201.21 sec**; both
generated inventory checks and `git diff --check` passed. This remains
synthetic controller evidence and does not verify physical pressure response.

### Standalone shutdown process-tree fidelity (2026-10-23)

A helper-level differential exposed a real mismatch: the standalone source
always calls `wait_procs` a second time, even after the first wait has no
survivors, and waits on the root using the configured timeout. The generic
wrapper had reused the central scheduler helper, whose no-survivor branch
returns after a 0.1-second root wait. Added a dedicated
`terminate_standalone_pressure_tree` helper and routed standalone signal and
keyboard-interrupt cleanup through it, preserving the central helper's
different behavior. The source differential checks empty survivors, survivors
through the second wait (including repeated KILL), and root lookup failure.
Focused source/pressure/process tests passed (**47 passed**).
The full Linux suite passed **18,968 tests in 201.14 sec** after the fix and
expanded escalation comparison;
the immutable reference snapshot, both generated inventories, and
`git diff --check` passed.

### Scheduler-label wording reconciliation (2026-10-23)

A re-read of both supplied attachments found that the plan text and suite guide
still summarized all scheduler labels as GPU-first. Corrected the plan, suite
guide, and source-to-function map to match the copied selector and call-site
behavior: `fixed` and `gpu_first` may select an eligible GPU independently of
the host-lane hold predicate; `pressure_aware` requires both lane predicates
for GPU, so it can decline a GPU-ready launch during a CPU hold, then try CPU.
The 55,296-state selector differential and focused grid/selector tests passed
(**3 passed**); both generated inventories and `git diff --check` pass. No
controller or ADP implementation code changed in this documentation correction.

### Linux child CPU environment source differential (2026-10-23)

Added an executing comparison of the copied `launcher_child_env` and the
generic child environment's shared Linux CPU/thread controls. It covers 2,250
combinations of positive/absent/unbounded concurrency hints, shared-mode
arguments and inherited settings, valid/reversed/invalid affinity strings,
explicit affinity slots, and stable job keys. The source and generic helper
matched on all compared thread, worker, affinity, and shared-mode values (**1
test passed**). GPU/backend variables added by the generic task adapter are
outside this helper-level comparison. OS scheduling/affinity calls and
physical accelerator behavior remain separately bounded or unverified.

The same pass added a direct bootstrap differential: five concurrency
environment configurations matched the source's resolved core/thread/worker
values, exported environment, and systemd/affinity/priority/no-swap/Torch hook
order. These hooks are mocked, so this verifies their call contract and values,
not whether host privileges permit each OS operation.

### Full Linux verification after the ledger crosswalk audit (2026-10-23)

Reread the complete 582-line ledger and 74-line implementation plan. The focused
source-differential, ADP mechanics, and one-epoch smoke set passed **644 tests
in 73.99 sec**. The complete Linux suite passed **18,971 tests in 204.82 sec**;
the reference snapshot integrity contracts passed (**2 tests**), both generated
inventories are current, and the documentation checks pass. The crosswalk now
classifies dispatcher details according to the planned generic plugin/worker
interface rather than as an unfinished OPF-specific dispatcher port. Remaining
verification limits are still real: synthetic controller trajectories do not
exhaust every possible interleaving, and live host/GPU pressure and OS policy
effects have not been exercised. Dataset-specific behavior and Windows remain
outside the active scope.

### Pressure victim selection: full-loop source differential (2026-10-23)

Expanded `test_source_main_and_generic_coordinator_pressure_victim_resume_match`
to execute both the copied source `main()` and `SuiteCoordinator.run()` under
GPU-only, host-only, and simultaneous host/GPU hard pressure. Distinct child RSS
and VRAM rankings make the expected victim differ by pressure domain. The test
compares the selected job, pause reason/counters, relaunch attempts and device,
and final completion. All three cases passed. The complete Linux suite passed
**18,973 tests in 211.70 sec**. This closes the direct full-loop source
comparison gap for those three pressure modes; CPU/GPU live-resource behavior
and arbitrary multi-child interleavings remain unverified.
