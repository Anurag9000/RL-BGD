# Coordinated pressure-aware suite guide

This guide describes how to plan and operate the generic suite runner. The
source-level behavior contract remains the
[runner ledger](COORDINATED_PRESSURE_AWARE_RUNNER_LEDGER.md); this guide is the
operator-facing entrypoint, and the
[implementation traceability map](COORDINATED_PRESSURE_AWARE_IMPLEMENTATION_SPEC.md)
records which contracts are implemented, partial, or platform-unverified.
The [source/function map](COORDINATED_PRESSURE_AWARE_SOURCE_FUNCTION_MAP.md)
links ledger source anchors to the generic modules, contract tests, and an
AST-generated inventory of runner functions.

## Components and extension points

The runner lives in `simple_adp.runner`:

- `SuitePlan` builds deterministic route identities, split archives, run
  specifications, and the SHA-256 signed manifest.
- `RunnerSettings` holds control-plane settings. Model training settings stay
  inside the model adapter and estimator.
- `DatasetProvider` lists task references, loads one `TabularTaskData` payload,
  and supplies split definitions. Payloads carry `X`, `y`, a stable dataset
  identity and content digest, feature schema, task kind, an optional
  `TaskAdapter`, and JSON-serializable `dataset_metadata`. The planner signs
  that metadata into each run identity/specification, and workers copy it into
  result records. The built-in provider adapts OpenML `cc18`, `tabzilla`, and
  `ctr23` pools; it creates a fresh classification/regression adapter for each
  worker route.
- `RunAdapter` (also exported as `ModelAdapter`) enumerates typed
  `ModelRoute`s and runs one route in a child process. A route declares its
  `suite_group` (`width_only`, `width_to_depth`, `depth_to_width`, `depth_only`,
  `vanilla`, or `screening`) so a new model family need not be inserted in the
  built-in ADP registry.
- `ResourceProvider` supplies host, GPU, child-process, process-tree, and
  external-process observations to `SuiteCoordinator`. The default callback
  provider uses psutil and `nvidia-smi`; tests or deployments can inject another
  implementation.
- `SuiteCoordinator` owns admission, state transitions, retries, pressure
  handling, child supervision, progress, and final reports. Each child handles
  one model × split × seed route.

Register provider and model adapter classes through the Python entry-point
groups `simple_adp.dataset_providers` and `simple_adp.model_adapters`. A custom
route must declare a valid `suite_group`; the planner does not infer unknown
model families. Providers own their stable task identities and split policy.

## Planning and execution

The Bash entrypoint is `scripts/simple-adp-suite.sh`; the PowerShell entrypoint
is `scripts/simple-adp-suite.ps1`. Both call
`python -m simple_adp.runner.launcher`. Direct CLI flags are accepted by the
launcher, and unknown `RunnerSettings` fields can be supplied through JSON or
`--set`.

```bash
scripts/simple-adp-suite.sh --list-suites --dataset-suites cc18,tabzilla,ctr23
scripts/simple-adp-suite.sh --list-suites --dataset-suites custom-provider:my-suite
scripts/simple-adp-suite.sh --dataset-suites tabzilla,ctr23 --dry-run \
  --output benchmark-results/coordinated-suite
scripts/simple-adp-suite.sh --dataset-suites tabzilla,ctr23 --smoke \
  --output benchmark-results/coordinated-suite
scripts/simple-adp-suite.sh --dataset-suites tabzilla,ctr23 \
  --output benchmark-results/coordinated-suite
scripts/simple-adp-suite.sh --dataset-suites tabzilla,ctr23 --resume \
  --output benchmark-results/coordinated-suite
scripts/simple-adp-suite.sh --status --output benchmark-results/coordinated-suite
scripts/simple-adp-suite.sh --report --output benchmark-results/coordinated-suite
```

`--dry-run` writes the plan and metadata and does not schedule child jobs.
`--smoke` overrides training to one epoch and sets each child's batch size to
the complete training partition. It does not change model architecture-search
rules. The runner never starts a suite just because it was installed or
imported. `--detach` starts a detached outer launcher and records its PID/log
under `<output>/run_root`; `--status` reads the current `status.json`.
`--resume` requires an existing `suite_plan.json`, reuses its identities, and
rejects changed settings, suites, paths, manifest digests, or missing plugin
registrations before starting the controller. `--report` rebuilds report files
from persisted job states and results without scheduling children. Its report
is labeled `persisted_state_snapshot`; invocation-only failure counters are
left unset because they cannot be reconstructed exactly from job states.

The default output directory is `benchmark-results/coordinated-suite`; the
default run root is `<output>/run_root`. Defaults enumerate all registered
routes and all tasks in the selected OpenML pools. Hardness screening is
diagnostic and does not silently filter routes in this coordinated suite.
Bare `--dataset-suites` entries remain OpenML suite names for compatibility.
Use `provider:suite` for any other registered provider; mixed selections such as
`cc18,custom-provider:my-suite` are supported by listing, planning, and exact
resume validation. `--list-suites` returns task IDs keyed by the provider-qualified
selector for non-OpenML providers.

## Configuration precedence

For each `RunnerSettings` field, values are applied in this order, with later
sources taking precedence:

1. Dataclass defaults.
2. Environment variable `SIMPLE_ADP_<FIELD_NAME>`; its value is parsed as JSON
   when valid, otherwise retained as text.
3. `--settings-json` object.
4. Repeated `--set KEY=VALUE` overrides, parsed as JSON scalar/list when valid.
5. Direct CLI flags such as `--output`, `--run-root`, `--profiles`, `--seeds`,
   `--task-limit`, `--fold-limit`, `--max-epochs`, `--batch-size`,
   `--concurrency`, `--gpu-device-index`, and `--cpu-only`.
6. `--smoke` and `--dry-run` set their corresponding boolean settings.

The concurrency control file is different: when configured, it is reread on
every scheduler pass. A nonempty nonnegative integer overrides configured
concurrency; zero maps to the unbounded sentinel `10**9`; empty content falls
back to configured concurrency; unreadable, malformed, or negative content is
an error. A lower live value affects later admission and does not terminate
already-active jobs.

## Identity, split, and resume rules

Each run identity includes provider, suite, task, split repeat/fold/sample,
suite repeat, model route/profile/fixed architecture, derived seed, split hash,
and suite signature. The run ID includes a sanitized readable identity and a
16-hex SHA-256 suffix. The job seed is SHA-256-derived from the base seed and
route identity fields. Manifests hash canonical JSON and are rejected when
their digest or current plan differs, except absolute Python executable
symlink aliases normalize to their real paths.

Split archives persist train, validation, and test indices plus a SHA-256 hash.
The OpenML adapter keeps test indices outside preprocessing, fitting,
validation, and screening. The estimator fits preprocessing only on training
rows. A child pressure pause terminates the process tree; a new process starts
on the same device and resumes from its signed estimator checkpoint. A
checkpoint whose data/task/configuration signature differs is rejected.

The suite startup also preserves the ledger's bundle ordering. The launcher
creates the default nested run root before attempting restore, and the bundle
restore guard declines any existing nonempty result directory. Therefore a
default output path with `<output>/run_root` may make automatic restore a
no-op; this recorded behavior is intentional compatibility behavior.

## Artifacts

The output directory contains split archives, per-route result directories,
`results.jsonl`, and optional gzip result bundles under the repository's
`resume_snapshots/` directory. The run root contains:

- `suite_launch_manifest.json`, `suite_plan.json`, and the controller's
  signed `manifest.json`;
- `run_metadata.json`, dynamic `status.json`, batch-backoff state, and
  append-only job events/progress;
- `jobs/<run-id>/run_spec.json`, `job_state.json`, command/context metadata,
  and append-only `job.log`;
- `final_report.json/.md`, `suite_summary.json/.md`, and rewritten
  `task_summary.csv` after the queue drains.

Each child result directory contains `result.json`, an atomic
`estimator.checkpoint.pt`, and `telemetry/simple_adp/<run-id>/` with
`telemetry.csv`, `progress_state.json`, `stage_state.json`, and uniquely named
RNG companion checkpoints. Telemetry contains orchestration progress; task
metrics and predictions are stored in `result.json` and model artifacts.
Bundle archives omit files named `telemetry.csv`; sharded telemetry can be
rehydrated from manifests with root containment checks and optional SHA-256
verification.

## Scheduler compatibility behavior

The compatibility defaults and edge behavior are specified in detail in the
ledger. Key values include: concurrency `0`/unbounded; total and GPU active
caps `0`/disabled; repeat/retry defaults `5/1`; host stop/resume
`1024/1280 MiB`; GPU stop/resume `100/500 MiB`; reserves `0/0 MiB`; child
buffers `500/100 MiB`; external drop `500 MiB`; poll `0.5 sec`; launch hold
`30 sec`; OOM batch backoff factor `0.5`; architecture caps
`512/5/1,000,000,000`; base seed `42`; and standalone-wrapper pressure/resume
`90/85%` with `0.5 sec` polling.

Admission uses strict `<` stop thresholds and resume thresholds for
hysteresis. Host available memory is not reduced by child RSS a second time.
`fixed` and `gpu_first` select an eligible GPU before CPU. `pressure_aware`
requires both GPU and host predicates for GPU selection, then falls back to
eligible CPU. Since CPU-lane readiness includes the CPU post-launch hold and
GPU readiness does not, `pressure_aware` can decline a GPU-ready launch during
that hold. The controller resumes paused jobs before polling exits or admitting queued work;
resume admission bypasses capacity, gates, holds, and active caps. It pauses
at most one pressure victim per pass. Central pressure pause kills and
relaunches from checkpoint; `PressureAwareWrapper` is separate and uses
OS-level stop/continue behavior.

These reports describe process outcomes and progress counts. They do not
certify convergence, metric validity, or scientific model quality. The
implementation traceability document lists remaining fidelity gaps. Windows
process and PowerShell behavior is outside the Linux implementation scope.
