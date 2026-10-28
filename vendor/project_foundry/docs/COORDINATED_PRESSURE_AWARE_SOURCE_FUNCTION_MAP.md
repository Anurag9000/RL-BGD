# Coordinated pressure-aware source-to-function map

> **Distribution note (2026-10-23):** This map preserves the original source-to-implementation audit. The source archive and source-differential test module are not included in this repository; test references to those historical comparisons are not runnable here. Current coverage is provided by the self-contained runner contract and integration tests.

This map links the canonical runner ledger's source anchors to the generic
`simple_adp.runner` implementation and the contract tests that exercise it.
The [runner ledger](COORDINATED_PRESSURE_AWARE_RUNNER_LEDGER.md) remains the
behavioral authority; this file is a navigation and coverage artifact, not a
replacement specification.

## Coverage labels

- **Ported**: the source-observed mechanism is represented in the generic
  package. This does not imply every branch has runtime evidence.
- **Adapted**: the mechanism is represented using generic dataset/model/task
  records or platform-neutral interfaces.
- **Excluded**: the ledger marks the source behavior task-, model-, or
  dispatcher-specific and outside this runner's scope.
- **Unverified**: implementation exists, but required OS or hardware execution
  evidence is not available in this checkout.

## Ledger source anchors to generic implementation

| Ledger concern and source anchor | Generic implementation | Contract evidence | Fidelity boundary |
|---|---|---|---|
| Host/GPU/process memory; ledger §§4, 14, 16-18; `utils/opf_massive_suite_runner.py:422-638` | `resources.py`: `sample_host_memory`, `sample_gpu_memory`, `process_tree_pids`, `sample_active_rss_mib`, `_gpu_process_rows`, `sample_child_memory`, `_is_python_process`, `sample_external_host_memory`, `sample_external_gpu_memory`, `external_drop_reopened`; `scheduler.py`: `SuiteCoordinator._sample_snapshot`, `_suspend_pressure_victim` | `tests/test_runner_resources.py`: external filtering, sensor failure, provider injection, all-or-zero active RSS, latest-only soft-gate sampling; `tests/test_runner_contracts.py`: `test_mixed_device_latest_child_sampling_retains_source_order_quirk` | **Adapted.** The source's active RSS sum and child-sample cadence are preserved; `ResourceProvider` allows synthetic resource sources. OS sampling remains platform-dependent. |
| Pressure reopen and OOM classifiers; §§5, 8, 14, 17; source `:601-700` | `admission.py`: `hard_pressure`, `pressure_pause_reopened`, gate helpers; `scheduler.py`: `_latest_attempt_text`, `_classify_failure`, `_handle_exit`, `_suspend_pressure_victim` | `tests/test_runner_contracts.py`: failure markers, exact-resume scan, pressure victim ordering, resume predicates | **Ported.** Task/model log contents are generic child logs. |
| CLI, settings validation, dynamic concurrency, launch capacity; §§1-2, 5, 14, 17; source `:758-980` | `launcher.py`: `_argument_parser`, `_settings`, `main`; `config.py`: `parse_epoch_limit`, `RunnerSettings.validate`, `resolve_concurrency`; `admission.py`: `resolve_launch_capacity`, `launch_budget` | `tests/test_runner_launcher.py`: precedence/resume/report; `tests/test_runner_contracts.py`: defaults, validation holes, reread concurrency, capacity boundaries | **Adapted.** OPF case/massive task switches and OPF-only no-op flags are not retained. |
| Child command and deterministic suite matrix; §§3, 6, 17; source `:981-1216` | `plan.py`: `SplitSpec`, `RunIdentity`, `RunSpec`, `SuitePlan.build`, `write_manifest`; `models.py`: `ModelRoute`, `SimpleADPModelAdapter.enumerate_routes`; `datasets.py`: provider interfaces and `OpenMLDatasetProvider` | `tests/test_runner_contracts.py`: manifest digest/alias, registered routes, repeat asymmetry, future plugins; `tests/test_smoke_suite.py`: one-epoch route execution | **Adapted.** Generic providers own task preparation and split identity; `universal_main.py` alias discovery is excluded. |
| Recovery, manifest/state views, retry predicate; §§3, 7-8, 17-18; source `:1217-1811` | `processes.py`: `find_exact_argv`, `reconcile_runtime`; `plan.py`: manifest digest and compatibility; `state.py`: `JobRuntime`, `JobStateStore`, `_write_compatibility_views`, `_summarize_compatibility_groups`, `seed_run_root_from_source`; `scheduler.py`: `_build_runtimes`, `_handle_exit`, `_should_retry`; `admission.py`: `AdmissionState.release_for_device` | `tests/test_runner_contracts.py`: recovery, manifest, unknown recovered exits, uncapped retry, source-root copy, CPU/GPU retry gate-release matrix; `tests/test_runner_source_fidelity.py::test_source_and_generic_recovered_child_unknown_exit_requeue_match` directly compares unknown-exit state, FIFO placement, and attempt/retry accounting; `tests/test_runner_processes.py`: exact argv normalization, process-table error handling, and deterministic earliest create-time/PID selection; `tests/test_runner_state_views.py`: aggregation rules, malformed nested identity/route handling including unhashable model/profile values; `tests/test_runner_source_fidelity.py::test_source_and_generic_final_report_fields_have_explicit_semantic_mapping` | **Ported/adapted.** OPF record names are genericized; PID-liveness and exact-argv limitations remain as documented. Retry/completion gate release is device-mode based: CPU reopens the host/CPU lane; GPU reopens host/CPU and GPU lanes. Final report outcome fields are source-mapped; generic identity, telemetry, and checkpoint fields extend the report. Candidate/group and CSV schemas intentionally use generic identity and paths. |
| Device admission, child environment and launch; §§5-6, 10, 14, 17; source `:1814-2008` | `admission.py`: `choose_device`; `runtime_tuning.py`: `launcher_child_env`, CPU partition/hash/thread helpers; `processes.py`: `launch_job`; `worker.py`: `run_spec` | `tests/test_runner_contracts.py`: parent CPU mask, scheduler labels, gate/cap/hold matrix, host resume boundary; `tests/test_runner_source_fidelity.py::test_scheduler_label_cpu_hold_edge_matches_reference` and the 55,296-state selector matrix; `tests/test_runner_tuning.py`: CPU/thread formulas; `tests/test_runner_source_fidelity.py::test_process_launch_order_and_linux_popen_contract_match_reference` directly compares actual source and generic launch ordering and POSIX Popen semantics | **Ported/adapted.** `fixed` and `gpu_first` select eligible GPU before CPU; `pressure_aware` requires both GPU and host predicates for GPU selection, then may fall back to eligible CPU. CPU-lane readiness includes the CPU post-launch hold while GPU readiness does not. CuPy child probing and optional cuDF/cuML hooks are initialized lazily before model adapters; optional-library behavior is mocked in tests, with physical GPU runtime still hardware-dependent. |
| Pressure victim, process-tree termination, same-device restart; §§7-8, 14, 17 | `scheduler.py`: `_architecture_priority`, `_suspend_pressure_victim`, `_pause`, `_resume`, `_resume_ready`, `_shutdown`; `processes.py`: `terminate_process_tree` | `tests/test_runner_contracts.py`: pressure ties, resume launch error, resume admission bypass, shutdown persistence, descendant termination; `tests/test_runner_source_fidelity.py::test_source_main_and_generic_coordinator_restart_a_real_child_after_host_pressure`: real Linux child termination and CPU-lane relaunch compared against source; `test_source_main_and_generic_coordinator_resume_all_paused_jobs_past_active_caps`: direct source/generic comparison for same-pass resume bypass; `test_source_main_and_generic_coordinator_pressure_victim_resume_match`: full source/generic main-loop comparison across GPU-only, host-only, and combined pressure, with opposing RSS/VRAM rankings, reason-specific pause counters, same-lane restart, and completion | **Ported.** Central pause is kill-and-relaunch; `pressure_wrapper.py` separately implements POSIX STOP/CONT. The pressure comparisons use synthetic sensor values; neither source comparison tests physical pressure or every multi-child interleaving. |
| Source-run-root seeding and result finalization; §§7, 11, 14, 17 | `state.py`: `seed_run_root_from_source`, `JobStateStore.save`, `append_event`; `scheduler.py`: `_finalize`, `_write_final_reports`, `write_persisted_state_report` | `tests/test_runner_contracts.py`: named-first copy/no-overwrite; `tests/test_runner_state_views.py`: report fields and atomic Markdown replacement; `tests/test_runner_launcher.py`: report-only mode | **Adapted.** Generic report and compatibility-view schemas replace OPF-specific task/result payloads. |
| Progress/status counters and outer scheduler loop; §§11, 14, 17 | `telemetry.py`: `ChildTelemetryWriter`, `count_job_log_progress`, `count_csv_records`, `summarize_offline`, `summarize_live`; `scheduler.py`: `_status`, `_sample_snapshot`, `run` | `tests/test_runner_child_telemetry.py`: progress precedence, callback output, reconciliation and resume; `tests/test_runner_contracts.py`: pass ordering, one-launch contract, and `test_full_pass_exit_pressure_and_peer_lane_cross_product` (16,128 synthetic passes across exit, pressure, peer-lane, cap, gate, and hold states; focused run passed 16,128 cases in 71.77 sec); `tests/test_runner_source_fidelity.py::test_source_and_generic_mixed_device_latest_child_sampling_match` directly pins the mixed-device latest-child sampling quirk; `tests/test_runner_source_fidelity.py`: normal retry, external-gate pause/reopen/resume, GPU OOM peer pause/retry/resume, GPU pressure victim selection/resume, real-child hard-pressure restart, multi-job resume admission bypass, and uncapped immediate launch-error retry until interruption | **Partial.** The source-executing comparisons exercise bounded trajectories, not every source/generic controller interleaving or physical pressure hardware. |
| Outer launcher, precedence, bundle restore/export; §§1, 3, 11, 17 | `launcher.py`: `_parse_value`, `_truthy`, `_settings`, `_detach`, `_run_with_watchdog`, `main`; `bundles.py`: bundle path, export, safe extraction, restore and shard rehydration; Bash/PowerShell launchers | `tests/test_runner_launcher.py`; `tests/test_runner_bundles.py`; static wrapper assertions in `tests/test_runner_contracts.py` | **Adapted/unverified.** Linux execution covers Bash. PowerShell has static checks only; restore-order behavior is exercised through launcher tests. |
| CPU/runtime tuning and OS controls; §§10, 15, 17 | `runtime_tuning.py`: CPU detection/parsing/partition/hash, thread budgets, affinity, priority, no-swap, systemd scope, bootstrap; `logging_context.py`: resource record factory | `tests/test_runner_tuning.py`: integer CPU frequency source precedence with malformed-value fallback and CPU-ID tie-break, affinity fallback, CPU ranges/partition/thread env, complete mocked systemd scope command/property/version/skip/fallback matrix, best-effort POSIX priority, and Linux `mlockall(3)` call; `tests/test_runner_source_fidelity.py::test_linux_child_cpu_environment_matches_reference_runtime_tuning` compares 2,250 source/generic child environment combinations; `test_linux_runtime_bootstrap_values_environment_and_call_order_match_reference` compares five bootstrap configurations and hook ordering; `tests/test_runner_contracts.py`: parent mask | **Linux scope.** OS mutations are best-effort; tests mock or inspect behavior without applying system changes. Windows verification is out of scope per user direction. |
| Backend admission, seed and logging context; §§12, 24 | `backend.py`: `backend_context`, CUDA probe, device resolution, seed resolution/application, CPU-value rejection; `logging_context.py`: defaults, provider overlay and console setup | `tests/test_core.py`; `tests/test_runner_contracts.py`: CPU mask/device routing | **Adapted.** Torch and optional CuPy probing, cuDF/cuML import hooks, array conversion, backend accuracy and seeding are implemented; OPF-specific log field names remain genericized. |
| Alternate pressure-aware wrapper; §§9, 17 | `pressure_wrapper.py`: pressure samples, threshold config, child pause/resume, signal lifecycle | `tests/test_runner_pressure.py`: defaults, missing GPU, both-domain resume, same-child resume, process accounting, lazy Python-usage sampling, and `test_standalone_posix_wrapper_stops_and_resumes_a_real_child_group` | **Ported, Linux scope.** The new integration test launches a real child in a new session and verifies actual `SIGSTOP`/`SIGCONT` delivery with deterministic synthetic pressure samples. Sensor sampling, memory-pressure events, and GPU behavior remain mocked; physical host/GPU pressure is not exercised. Windows process-tree suspension is out of scope per user direction. |
| Watchdogs and emergency process cleanup; §§8, 22 | `watchdog.py`: master liveness and monitor; `process_cleanup.py`: ownership and kill; `scripts/simple-adp-watchdog.ps1`, `scripts/kill_simple_adp_runners.py` | `tests/test_runner_contracts.py`: POSIX descendants and static Windows cleanup ordering | **Unverified on Windows.** PowerShell parser/runtime checks require Windows or an installed PowerShell runtime. |
| Resume bundles and telemetry shard restore; §§11, 17 | `bundles.py`: chunked archive, filter, safe extraction, restore guard, path/checksum-validated shard rehydration | `tests/test_runner_bundles.py`: telemetry exclusion, traversal rejection, checksums, malformed/missing shard cases, split archive restore | **Adapted.** Generic path roots and result directories replace OPF bundle naming. |
| Child name resolution and nested native/ported dispatch; §§6, 17, 25 | `worker.py`: one serialized route per child, plugin registry resolution, estimator adapter invocation | `tests/test_runner_contracts.py`: future adapter planning; `tests/test_smoke_suite.py`: route/task execution | **Excluded/adapted.** OPF `universal_main.py` model aliases, import-time dtype changes, unknown-argument coercion, nested ported-process stream tee, and task dispatch are intentionally not copied. |
| Child checkpoint and telemetry handoff; §§20-21 | `checkpoint.py`, `training.py`, `search.py`, `estimator.py`; `telemetry.py`: active-stage reconciliation, RNG companions, and stage checkpoint mirrors | `tests/test_search_training.py`; `tests/test_runner_child_telemetry.py`: interrupted fit resumed through `TabularADP.fit`, including depth-stage band and architecture-length rejection, plus byte-identical telemetry/coordinator last, best, and initial best-batch mirrors | **Adapted/partial.** Exact same-stage optimizer/scheduler/data/RNG resume exists for registered estimator routes. Generic depth-stage catch-up now checks identical stage bounds, architecture length `stage_index + 1`, every layer width in the inclusive band, monotonic catch-up, and best/current tensor key/shape equality. Worker persistence mirrors the full last record each batch and best-validation model at stage boundaries; a fresh architecture stage initializes the best-batch snapshot with infinite loss. The signed estimator checkpoint remains the exact-resume authority; no later best-batch selection rule is documented. OPF schema-2 loader/backend contracts remain different from the generic checkpoint schema. |

## Generic function and method inventory
The inventory below is generated from the current Python AST. Line spans refer to
the current checkout and should be refreshed after code movement. It covers each
function, method, and nested callback in `src/simple_adp/runner/`; excluded copied-source behavior
is identified in the mapping above rather than silently attributed to a generic
function.

Regenerate or verify it with `python scripts/generate_runner_function_map.py` or
`python scripts/generate_runner_function_map.py --check`.

| Module | Function or method | Current lines |
|---|---|---:|
| `src/simple_adp/runner/__init__.py` | `__getattr__` | 38-44 |
| `src/simple_adp/runner/__init__.py` | `__dir__` | 47-48 |
| `src/simple_adp/runner/admission.py` | `parent_cpu_masked` | 16-29 |
| `src/simple_adp/runner/admission.py` | `AdmissionState.invocation_reset` | 46-53 |
| `src/simple_adp/runner/admission.py` | `AdmissionState.release_for_device` | 55-67 |
| `src/simple_adp/runner/admission.py` | `resolve_launch_capacity` | 77-103 |
| `src/simple_adp/runner/admission.py` | `launch_budget` | 106-107 |
| `src/simple_adp/runner/admission.py` | `hard_pressure` | 110-113 |
| `src/simple_adp/runner/admission.py` | `update_soft_gates` | 116-139 |
| `src/simple_adp/runner/admission.py` | `reopen_external_gates` | 142-150 |
| `src/simple_adp/runner/admission.py` | `choose_device` | 153-197 |
| `src/simple_adp/runner/admission.py` | `pressure_pause_reopened` | 200-230 |
| `src/simple_adp/runner/artifacts.py` | `estimator_checkpoint_path` | 11-13 |
| `src/simple_adp/runner/artifacts.py` | `estimator_model_path` | 16-18 |
| `src/simple_adp/runner/backend.py` | `backend_context` | 42-57 |
| `src/simple_adp/runner/backend.py` | `_check_admission` | 60-77 |
| `src/simple_adp/runner/backend.py` | `_reject_masked_gpu_array` | 80-88 |
| `src/simple_adp/runner/backend.py` | `resolve_seed` | 91-103 |
| `src/simple_adp/runner/backend.py` | `_try_import_cupy` | 106-122 |
| `src/simple_adp/runner/backend.py` | `gpu_available` | 125-141 |
| `src/simple_adp/runner/backend.py` | `_try_enable_cudf_pandas` | 144-159 |
| `src/simple_adp/runner/backend.py` | `_try_enable_cuml_accel` | 162-177 |
| `src/simple_adp/runner/backend.py` | `enable_optional_gpu_accelerators` | 180-206 |
| `src/simple_adp/runner/backend.py` | `status` | 209-210 |
| `src/simple_adp/runner/backend.py` | `backend_pandas` | 213-217 |
| `src/simple_adp/runner/backend.py` | `backend_numpy` | 220-223 |
| `src/simple_adp/runner/backend.py` | `array_module` | 226-234 |
| `src/simple_adp/runner/backend.py` | `as_numpy` | 237-242 |
| `src/simple_adp/runner/backend.py` | `as_cupy` | 245-249 |
| `src/simple_adp/runner/backend.py` | `resolve_torch_device` | 252-261 |
| `src/simple_adp/runner/backend.py` | `require_cpu_value` | 264-273 |
| `src/simple_adp/runner/backend.py` | `seed_everything` | 276-292 |
| `src/simple_adp/runner/backend.py` | `backend_accuracy_score` | 295-327 |
| `src/simple_adp/runner/backend.py` | `backend_accuracy_score.<locals>.accuracy_score` | 298-325 |
| `src/simple_adp/runner/backend.py` | `iter_backend_names` | 330-335 |
| `src/simple_adp/runner/backend.py` | `reset_backend_cache` | 338-343 |
| `src/simple_adp/runner/bundles.py` | `_safe_name` | 17-18 |
| `src/simple_adp/runner/bundles.py` | `bundle_path_for_results_dir` | 21-28 |
| `src/simple_adp/runner/bundles.py` | `_manifest_path` | 31-32 |
| `src/simple_adp/runner/bundles.py` | `_parts` | 35-36 |
| `src/simple_adp/runner/bundles.py` | `_dir_size` | 39-49 |
| `src/simple_adp/runner/bundles.py` | `_manifest` | 52-62 |
| `src/simple_adp/runner/bundles.py` | `_bundle_filter` | 65-68 |
| `src/simple_adp/runner/bundles.py` | `export_results_bundle` | 71-96 |
| `src/simple_adp/runner/bundles.py` | `_safe_extract` | 99-119 |
| `src/simple_adp/runner/bundles.py` | `restore_results_bundle` | 122-154 |
| `src/simple_adp/runner/bundles.py` | `_target_path` | 157-165 |
| `src/simple_adp/runner/bundles.py` | `rehydrate_sharded_telemetry` | 168-208 |
| `src/simple_adp/runner/compat.py` | `ensure_dir` | 19-22 |
| `src/simple_adp/runner/compat.py` | `local_timestamp` | 25-27 |
| `src/simple_adp/runner/compat.py` | `env_int` | 30-39 |
| `src/simple_adp/runner/compat.py` | `env_float` | 42-51 |
| `src/simple_adp/runner/compat.py` | `safe_int` | 54-58 |
| `src/simple_adp/runner/compat.py` | `safe_float` | 61-65 |
| `src/simple_adp/runner/compat.py` | `parse_csv_ints` | 68-70 |
| `src/simple_adp/runner/compat.py` | `parse_range_tuple` | 73-78 |
| `src/simple_adp/runner/compat.py` | `format_kv` | 81-87 |
| `src/simple_adp/runner/compat.py` | `json_signature` | 90-93 |
| `src/simple_adp/runner/compat.py` | `load_json` | 96-101 |
| `src/simple_adp/runner/config.py` | `parse_epoch_limit` | 14-28 |
| `src/simple_adp/runner/config.py` | `_int_tuple` | 31-37 |
| `src/simple_adp/runner/config.py` | `RunnerSettings.__post_init__` | 149-164 |
| `src/simple_adp/runner/config.py` | `RunnerSettings.validate` | 166-252 |
| `src/simple_adp/runner/config.py` | `RunnerSettings.as_dict` | 256-257 |
| `src/simple_adp/runner/config.py` | `RunnerSettings.effective_concurrency` | 260-261 |
| `src/simple_adp/runner/config.py` | `RunnerSettings.resolve_concurrency` | 263-278 |
| `src/simple_adp/runner/config.py` | `RunnerSettings.launch_budget` | 280-281 |
| `src/simple_adp/runner/config.py` | `StandalonePressureSettings.__post_init__` | 297-304 |
| `src/simple_adp/runner/controller.py` | `main` | 17-40 |
| `src/simple_adp/runner/datasets.py` | `DatasetTaskRef.key` | 24-25 |
| `src/simple_adp/runner/datasets.py` | `DatasetProvider.list_tasks` | 59-59 |
| `src/simple_adp/runner/datasets.py` | `DatasetProvider.load_task` | 61-61 |
| `src/simple_adp/runner/datasets.py` | `DatasetProvider.list_splits` | 63-63 |
| `src/simple_adp/runner/datasets.py` | `DatasetRegistry.__init__` | 67-68 |
| `src/simple_adp/runner/datasets.py` | `DatasetRegistry.register` | 70-76 |
| `src/simple_adp/runner/datasets.py` | `DatasetRegistry.get` | 78-82 |
| `src/simple_adp/runner/datasets.py` | `DatasetRegistry.providers` | 85-86 |
| `src/simple_adp/runner/datasets.py` | `OpenMLDatasetProvider.list_tasks` | 94-100 |
| `src/simple_adp/runner/datasets.py` | `OpenMLDatasetProvider.load_task` | 102-107 |
| `src/simple_adp/runner/datasets.py` | `OpenMLDatasetProvider.list_splits` | 109-143 |
| `src/simple_adp/runner/launcher.py` | `_parse_value` | 29-33 |
| `src/simple_adp/runner/launcher.py` | `_truthy` | 36-44 |
| `src/simple_adp/runner/launcher.py` | `_argument_parser` | 47-74 |
| `src/simple_adp/runner/launcher.py` | `_settings` | 77-122 |
| `src/simple_adp/runner/launcher.py` | `_apply_runtime_environment_defaults` | 125-139 |
| `src/simple_adp/runner/launcher.py` | `_dataset_suites` | 142-162 |
| `src/simple_adp/runner/launcher.py` | `_registered_dataset_providers` | 165-172 |
| `src/simple_adp/runner/launcher.py` | `_detach` | 175-203 |
| `src/simple_adp/runner/launcher.py` | `_run_with_watchdog` | 206-283 |
| `src/simple_adp/runner/launcher.py` | `_run_with_watchdog.<locals>.stop_for_signal` | 217-237 |
| `src/simple_adp/runner/launcher.py` | `load_registries` | 286-290 |
| `src/simple_adp/runner/launcher.py` | `main` | 293-447 |
| `src/simple_adp/runner/launcher.py` | `cli_main` | 450-459 |
| `src/simple_adp/runner/logging_context.py` | `default_resource_context` | 16-51 |
| `src/simple_adp/runner/logging_context.py` | `install_resource_record_factory` | 54-77 |
| `src/simple_adp/runner/logging_context.py` | `install_resource_record_factory.<locals>.factory` | 63-74 |
| `src/simple_adp/runner/logging_context.py` | `configure_console_logging` | 80-103 |
| `src/simple_adp/runner/models.py` | `ModelAdapter.enumerate_routes` | 25-25 |
| `src/simple_adp/runner/models.py` | `ModelAdapter.fit_predict` | 27-27 |
| `src/simple_adp/runner/models.py` | `ModelRegistry.__init__` | 37-38 |
| `src/simple_adp/runner/models.py` | `ModelRegistry.register` | 40-46 |
| `src/simple_adp/runner/models.py` | `ModelRegistry.get` | 48-52 |
| `src/simple_adp/runner/models.py` | `ModelRegistry.adapters` | 55-56 |
| `src/simple_adp/runner/models.py` | `SimpleADPModelAdapter._take_rows` | 65-78 |
| `src/simple_adp/runner/models.py` | `SimpleADPModelAdapter.enumerate_routes` | 80-169 |
| `src/simple_adp/runner/models.py` | `SimpleADPModelAdapter.fit_predict` | 171-278 |
| `src/simple_adp/runner/plan.py` | `_json_default` | 25-34 |
| `src/simple_adp/runner/plan.py` | `canonical_json` | 37-38 |
| `src/simple_adp/runner/plan.py` | `sha256_json` | 41-42 |
| `src/simple_adp/runner/plan.py` | `manifest_compatibility_digest` | 45-53 |
| `src/simple_adp/runner/plan.py` | `derive_run_seed` | 56-60 |
| `src/simple_adp/runner/plan.py` | `SplitSpec.save` | 73-97 |
| `src/simple_adp/runner/plan.py` | `SplitSpec.load` | 100-111 |
| `src/simple_adp/runner/plan.py` | `_sha_indices` | 114-118 |
| `src/simple_adp/runner/plan.py` | `RunIdentity.run_id` | 136-155 |
| `src/simple_adp/runner/plan.py` | `RunIdentity.run_id.<locals>.safe` | 137-139 |
| `src/simple_adp/runner/plan.py` | `RunIdentity.as_dict` | 157-160 |
| `src/simple_adp/runner/plan.py` | `RunSpec.as_dict` | 175-186 |
| `src/simple_adp/runner/plan.py` | `SuitePlan.from_manifest` | 196-227 |
| `src/simple_adp/runner/plan.py` | `SuitePlan.build` | 230-506 |
| `src/simple_adp/runner/plan.py` | `SuitePlan.build.<locals>.route_group` | 276-296 |
| `src/simple_adp/runner/plan.py` | `SuitePlan.write_manifest` | 508-534 |
| `src/simple_adp/runner/plugins.py` | `_instantiate` | 12-13 |
| `src/simple_adp/runner/plugins.py` | `load_registries` | 16-28 |
| `src/simple_adp/runner/pressure_wrapper.py` | `PressureSample.external_host_mib` | 31-32 |
| `src/simple_adp/runner/pressure_wrapper.py` | `PressureSample.external_gpu_mib` | 35-36 |
| `src/simple_adp/runner/pressure_wrapper.py` | `_gpu_pressure_sample` | 39-57 |
| `src/simple_adp/runner/pressure_wrapper.py` | `_sample_host_gpu_pressure` | 60-72 |
| `src/simple_adp/runner/pressure_wrapper.py` | `_sample_python_usage` | 75-105 |
| `src/simple_adp/runner/pressure_wrapper.py` | `_with_python_usage` | 108-118 |
| `src/simple_adp/runner/pressure_wrapper.py` | `_external_drop_reopens_gates` | 121-133 |
| `src/simple_adp/runner/pressure_wrapper.py` | `sample_pressure` | 136-138 |
| `src/simple_adp/runner/pressure_wrapper.py` | `_pause_child` | 141-159 |
| `src/simple_adp/runner/pressure_wrapper.py` | `PressureAwareWrapper.__init__` | 163-173 |
| `src/simple_adp/runner/pressure_wrapper.py` | `PressureAwareWrapper._interrupt` | 175-183 |
| `src/simple_adp/runner/pressure_wrapper.py` | `PressureAwareWrapper.run` | 185-303 |
| `src/simple_adp/runner/pressure_wrapper.py` | `main` | 306-346 |
| `src/simple_adp/runner/process_cleanup.py` | `_entrypoint_markers` | 26-49 |
| `src/simple_adp/runner/process_cleanup.py` | `_argv_contains_root` | 52-62 |
| `src/simple_adp/runner/process_cleanup.py` | `owns_process` | 65-84 |
| `src/simple_adp/runner/process_cleanup.py` | `kill_owned_processes` | 87-123 |
| `src/simple_adp/runner/processes.py` | `ProcessHandle.poll` | 29-29 |
| `src/simple_adp/runner/processes.py` | `close_log_handle` | 32-43 |
| `src/simple_adp/runner/processes.py` | `RecoveredProcessHandle.poll` | 50-55 |
| `src/simple_adp/runner/processes.py` | `process_is_live` | 58-71 |
| `src/simple_adp/runner/processes.py` | `_normalize_argv` | 74-80 |
| `src/simple_adp/runner/processes.py` | `find_exact_argv` | 83-102 |
| `src/simple_adp/runner/processes.py` | `terminate_process_tree` | 105-156 |
| `src/simple_adp/runner/processes.py` | `terminate_standalone_pressure_tree` | 159-206 |
| `src/simple_adp/runner/processes.py` | `terminate_process_tree_after_grace` | 209-239 |
| `src/simple_adp/runner/processes.py` | `terminate_process_tree_after_grace.<locals>.signal_tree` | 211-233 |
| `src/simple_adp/runner/processes.py` | `terminate_process_tree_after_grace.<locals>.signal_tree.<locals>.signal_postorder` | 217-231 |
| `src/simple_adp/runner/processes.py` | `launch_job` | 242-344 |
| `src/simple_adp/runner/processes.py` | `reconcile_runtime` | 347-471 |
| `src/simple_adp/runner/processes.py` | `_safe_record_int` | 474-478 |
| `src/simple_adp/runner/processes.py` | `_safe_record_float` | 481-485 |
| `src/simple_adp/runner/processes.py` | `_persist_live_recovery` | 488-502 |
| `src/simple_adp/runner/processes.py` | `_save_reconciled_state` | 505-516 |
| `src/simple_adp/runner/resources.py` | `ResourceSnapshot.as_dict` | 57-61 |
| `src/simple_adp/runner/resources.py` | `ResourceProvider.sample_host` | 68-68 |
| `src/simple_adp/runner/resources.py` | `ResourceProvider.sample_gpu` | 69-69 |
| `src/simple_adp/runner/resources.py` | `ResourceProvider.process_tree` | 70-70 |
| `src/simple_adp/runner/resources.py` | `ResourceProvider.sample_active_rss` | 71-71 |
| `src/simple_adp/runner/resources.py` | `ResourceProvider.sample_child` | 72-72 |
| `src/simple_adp/runner/resources.py` | `ResourceProvider.sample_external_host` | 73-73 |
| `src/simple_adp/runner/resources.py` | `ResourceProvider.sample_external_gpu` | 74-74 |
| `src/simple_adp/runner/resources.py` | `CallbackResourceProvider.sample_host` | 88-89 |
| `src/simple_adp/runner/resources.py` | `CallbackResourceProvider.sample_gpu` | 91-92 |
| `src/simple_adp/runner/resources.py` | `CallbackResourceProvider.process_tree` | 94-95 |
| `src/simple_adp/runner/resources.py` | `CallbackResourceProvider.sample_active_rss` | 97-98 |
| `src/simple_adp/runner/resources.py` | `CallbackResourceProvider.sample_child` | 100-101 |
| `src/simple_adp/runner/resources.py` | `CallbackResourceProvider.sample_external_host` | 103-104 |
| `src/simple_adp/runner/resources.py` | `CallbackResourceProvider.sample_external_gpu` | 106-107 |
| `src/simple_adp/runner/resources.py` | `sample_host_memory` | 110-124 |
| `src/simple_adp/runner/resources.py` | `sample_gpu_memory` | 127-143 |
| `src/simple_adp/runner/resources.py` | `process_tree_pids` | 146-160 |
| `src/simple_adp/runner/resources.py` | `sample_active_rss_mib` | 163-169 |
| `src/simple_adp/runner/resources.py` | `_gpu_process_rows` | 172-194 |
| `src/simple_adp/runner/resources.py` | `sample_child_memory` | 197-213 |
| `src/simple_adp/runner/resources.py` | `_is_python_process_name` | 216-218 |
| `src/simple_adp/runner/resources.py` | `sample_external_host_memory` | 221-241 |
| `src/simple_adp/runner/resources.py` | `sample_external_gpu_memory` | 244-257 |
| `src/simple_adp/runner/resources.py` | `external_drop_reopened` | 260-272 |
| `src/simple_adp/runner/route.py` | `ModelRoute.route_label` | 31-34 |
| `src/simple_adp/runner/route.py` | `ModelRoute.as_dict` | 36-40 |
| `src/simple_adp/runner/runtime_tuning.py` | `detected_cpus` | 16-38 |
| `src/simple_adp/runner/runtime_tuning.py` | `parse_cpu_list` | 41-55 |
| `src/simple_adp/runner/runtime_tuning.py` | `partition_cpus` | 58-66 |
| `src/simple_adp/runner/runtime_tuning.py` | `stable_cpu_slot` | 69-71 |
| `src/simple_adp/runner/runtime_tuning.py` | `current_concurrency_hint` | 74-93 |
| `src/simple_adp/runner/runtime_tuning.py` | `resolve_thread_budget` | 96-99 |
| `src/simple_adp/runner/runtime_tuning.py` | `resolve_num_workers` | 102-103 |
| `src/simple_adp/runner/runtime_tuning.py` | `_apply_affinity` | 106-123 |
| `src/simple_adp/runner/runtime_tuning.py` | `_apply_affinity_from_env` | 126-136 |
| `src/simple_adp/runner/runtime_tuning.py` | `_apply_priority` | 139-185 |
| `src/simple_adp/runner/runtime_tuning.py` | `_enforce_no_swap` | 188-202 |
| `src/simple_adp/runner/runtime_tuning.py` | `launcher_child_env` | 205-299 |
| `src/simple_adp/runner/runtime_tuning.py` | `maybe_reexec_systemd_scope` | 311-364 |
| `src/simple_adp/runner/runtime_tuning.py` | `shutil_which` | 367-370 |
| `src/simple_adp/runner/runtime_tuning.py` | `bootstrap_runtime` | 373-418 |
| `src/simple_adp/runner/scheduler.py` | `_coerce_int` | 75-79 |
| `src/simple_adp/runner/scheduler.py` | `_summary_markdown` | 82-103 |
| `src/simple_adp/runner/scheduler.py` | `ControllerMemory.defaults` | 123-152 |
| `src/simple_adp/runner/scheduler.py` | `ControllerMemory.reset_for_invocation` | 154-155 |
| `src/simple_adp/runner/scheduler.py` | `ControllerMemory.admission_state` | 157-170 |
| `src/simple_adp/runner/scheduler.py` | `ControllerMemory.update_admission` | 172-185 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator.__init__` | 191-231 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._logging_resource_context` | 233-246 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator.batch_state_path` | 249-250 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._load_controller_state` | 252-263 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._save_controller_state` | 265-267 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._build_runtimes` | 269-290 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._counts` | 293-295 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._record_event` | 297-337 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._latest` | 340-342 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._sample_runtime_child` | 344-352 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._status` | 354-406 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._sample_snapshot` | 408-602 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._resume_ready` | 604-623 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._pause` | 625-657 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._resume` | 659-720 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._should_retry` | 723-724 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._latest_attempt_text` | 727-750 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._classify_failure` | 752-764 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._load_worker_result` | 766-772 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._release_gate_locks` | 774-788 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._finalize` | 790-843 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._handle_exit` | 845-932 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._architecture_priority` | 935-944 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._suspend_pressure_victim` | 946-996 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._install_signal_handlers` | 998-1009 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._install_signal_handlers.<locals>.handler` | 999-1002 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._shutdown` | 1011-1044 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator._write_final_reports` | 1046-1095 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator.write_persisted_state_report` | 1097-1174 |
| `src/simple_adp/runner/scheduler.py` | `SuiteCoordinator.run` | 1176-1393 |
| `src/simple_adp/runner/screening.py` | `build_screening_reports` | 12-158 |
| `src/simple_adp/runner/screening.py` | `expected_splits_fold_keys` | 161-166 |
| `src/simple_adp/runner/state.py` | `utc_now` | 30-31 |
| `src/simple_adp/runner/state.py` | `atomic_write_text` | 34-38 |
| `src/simple_adp/runner/state.py` | `atomic_write_json` | 41-42 |
| `src/simple_adp/runner/state.py` | `JobRuntime.job_id` | 77-78 |
| `src/simple_adp/runner/state.py` | `JobRuntime.is_gpu` | 81-82 |
| `src/simple_adp/runner/state.py` | `JobRuntime.live` | 85-93 |
| `src/simple_adp/runner/state.py` | `JobRuntime.derived_status` | 96-109 |
| `src/simple_adp/runner/state.py` | `JobRuntime.state_path` | 111-112 |
| `src/simple_adp/runner/state.py` | `JobRuntime.to_record` | 114-149 |
| `src/simple_adp/runner/state.py` | `_recovered_pid_live` | 152-159 |
| `src/simple_adp/runner/state.py` | `JobStateStore.__init__` | 177-182 |
| `src/simple_adp/runner/state.py` | `JobStateStore.job_dir` | 184-185 |
| `src/simple_adp/runner/state.py` | `JobStateStore.job_dir_from_id` | 187-188 |
| `src/simple_adp/runner/state.py` | `JobStateStore.state_path` | 190-191 |
| `src/simple_adp/runner/state.py` | `JobStateStore.save` | 193-195 |
| `src/simple_adp/runner/state.py` | `JobStateStore.save_state_only` | 197-203 |
| `src/simple_adp/runner/state.py` | `JobStateStore.load` | 205-211 |
| `src/simple_adp/runner/state.py` | `JobStateStore.append_event` | 213-232 |
| `src/simple_adp/runner/state.py` | `JobStateStore.append_progress` | 234-282 |
| `src/simple_adp/runner/state.py` | `JobStateStore._write_compatibility_views` | 284-344 |
| `src/simple_adp/runner/state.py` | `JobStateStore._slug` | 347-349 |
| `src/simple_adp/runner/state.py` | `JobStateStore._compatibility_group` | 351-370 |
| `src/simple_adp/runner/state.py` | `JobStateStore._safe_exit_code` | 373-380 |
| `src/simple_adp/runner/state.py` | `JobStateStore._summarize_compatibility_groups` | 382-540 |
| `src/simple_adp/runner/state.py` | `JobStateStore._summarize_compatibility_groups.<locals>.mapping_field` | 399-401 |
| `src/simple_adp/runner/state.py` | `JobStateStore._summarize_compatibility_groups.<locals>.same_task` | 403-407 |
| `src/simple_adp/runner/state.py` | `JobStateStore._summarize_compatibility_groups.<locals>.same_method` | 409-418 |
| `src/simple_adp/runner/state.py` | `JobStateStore._summarize_compatibility_groups.<locals>.counts` | 423-433 |
| `src/simple_adp/runner/state.py` | `JobStateStore._summarize_compatibility_groups.<locals>.latest_completed` | 435-450 |
| `src/simple_adp/runner/state.py` | `JobStateStore._summarize_compatibility_groups.<locals>.latest_completed.<locals>.ended_key` | 443-448 |
| `src/simple_adp/runner/state.py` | `JobStateStore._summarize_compatibility_groups.<locals>.family_key` | 454-462 |
| `src/simple_adp/runner/state.py` | `_RouteRecord.__init__` | 546-548 |
| `src/simple_adp/runner/state.py` | `seed_run_root_from_source` | 551-586 |
| `src/simple_adp/runner/state.py` | `seed_run_root_from_source.<locals>.copy_one` | 565-572 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.__init__` | 36-64 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._load_counters` | 66-81 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._atomic_json` | 84-88 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._atomic_torch_save` | 91-103 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.save_last_checkpoint` | 105-109 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.save_best_checkpoint` | 111-115 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.save_best_batch_checkpoint` | 117-121 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.persist_training_checkpoint` | 123-170 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._save_stage_state` | 172-217 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._write_progress_state` | 219-241 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.load_stage_state` | 243-257 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._prepare_loaded_stage_state` | 259-284 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._active_checkpoint_has_exact_state` | 286-316 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._restore_stage_rng` | 318-348 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._load_stage_state_from_csv` | 350-377 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.exists` | 379-381 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.read_text` | 383-388 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.write_text` | 390-403 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._architecture` | 406-415 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._state_shapes` | 418-425 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.reconcile_checkpoint` | 427-529 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._depth_recovery_is_bounded` | 532-569 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._stage_key` | 572-580 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter._append` | 582-618 |
| `src/simple_adp/runner/telemetry.py` | `ChildTelemetryWriter.__call__` | 620-637 |
| `src/simple_adp/runner/telemetry.py` | `telemetry_callback` | 640-657 |
| `src/simple_adp/runner/telemetry.py` | `count_text_lines` | 660-665 |
| `src/simple_adp/runner/telemetry.py` | `count_job_log_progress` | 668-677 |
| `src/simple_adp/runner/telemetry.py` | `count_csv_records` | 680-695 |
| `src/simple_adp/runner/telemetry.py` | `summarize_offline` | 698-728 |
| `src/simple_adp/runner/telemetry.py` | `summarize_live` | 731-790 |
| `src/simple_adp/runner/watchdog.py` | `master_alive` | 15-23 |
| `src/simple_adp/runner/watchdog.py` | `monitor` | 26-92 |
| `src/simple_adp/runner/watchdog.py` | `main` | 95-101 |
| `src/simple_adp/runner/worker.py` | `_rebuild_spec` | 26-36 |
| `src/simple_adp/runner/worker.py` | `load_registries` | 39-43 |
| `src/simple_adp/runner/worker.py` | `run_spec` | 46-187 |
| `src/simple_adp/runner/worker.py` | `main` | 190-195 |
