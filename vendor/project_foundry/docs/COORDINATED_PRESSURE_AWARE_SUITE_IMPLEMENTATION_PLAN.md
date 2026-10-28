<proposed_plan>
# Generic, Ledger-Faithful Suite Coordinator for simple-adp

## Summary

Add a centralized coordinator for reproducible runs across the models and datasets registered in `simple-adp`, with extension points for future model families, tasks, and dataset sources. Keep the implementation self-contained; do not ship copied OPF_ADP source files in ProjectFoundry. Implement a generic port of the ledger’s orchestration behavior and recorded defaults around the repo’s estimator and benchmark interfaces.

The default suite enumerates every registered route: the 14 canonical model IDs, both generic-search profiles where applicable, `MLP-Fixed`, its 50 individual vanilla architectures, and all tasks provided by registered dataset adapters. Hardness screening runs as a report; it does not exclude routes. Suite smoke testing uses **one epoch and a batch equal to the training partition size**. It does not launch automatically.

## Coordinator and extension interfaces

- Add typed interfaces for `SuitePlan`, `RunnerSettings`, `ModelRoute`, `DatasetProvider`, `RunAdapter`, `RunIdentity`, and `SuiteCoordinator`.
  - A `DatasetProvider` lists task identities and loads data, metadata, task adapter, and split definitions.
  - A `RunAdapter` resolves a registered model route into a serializable worker specification and invokes the existing estimator or baseline.
  - Register current models and OpenML pools through built-in adapters; future models and dataset sources register through the same interfaces. Unregistered routes are never inferred or silently added.
- Build deterministic run identities from dataset/task, split, model ID, generic profile, fixed architecture when applicable, seed, and run configuration. Write a signed run manifest before execution; reject incompatible resume manifests.
- Use **one child process per model × dataset split × seed route**. The worker loads its assigned data and runs one route. The coordinator owns scheduling, resource supervision, run state, retry decisions, and result aggregation.
- Serialize estimator search and optimizer controls in each run specification and pass them through the worker adapter: width/depth increments, epoch and architecture patience, validation margins, architecture caps, optimizer/scheduler values, generic penalty weights, candidate limit, and minimum generic depth seed. The one-epoch/full-training-batch smoke profile overrides only epoch count and batch size. Custom model adapters can interpret their own registered route settings without changing coordinator scheduling.
- Preserve OpenML’s existing split discipline: keep test rows out of preprocessing, training, validation, and screening; persist split indices and hashes. Custom providers supply their own split policy and stable dataset identity.
- Add a CLI for suite listing, dry-run planning, smoke runs, full runs, resume, and status/report generation. Produce a suite document covering interfaces, configuration precedence, run identity, emitted artifacts, operational procedures, and faithful behavior boundaries.

## Ledger behavior and values to preserve

Implement and document the ledger’s recorded behavior as the coordinator’s compatibility contract. Do not silently “fix” a documented behavior. Tests cover the implemented mechanics and recorded edge cases; source-snapshot differential tests remain in the originating repository.

- Preserve the full controller cycle: reread concurrency each pass; sample host/GPU and child/external memory; calculate capacity and active caps; evaluate strict pressure thresholds and child-footprint gates; handle external-memory-drop reopen signals; resume paused runs; reap exits and classify failures; pause at most one pressure victim per pass; then attempt queue admission.
- Preserve process semantics: process-tree supervision, one successful launch per scheduling pass, per-device post-launch holds, FIFO queue admission, retry-to-tail behavior, OOM backoff, and same-device restart after a pressure pause. A central pressure pause terminates the process tree and relaunches from checkpoint; the standalone pressure wrapper remains a distinct OS STOP/CONT mechanism.
- Preserve state and recovery behavior: queued/launching/running/paused/retrying/completed/failed states; startup reconciliation; live-process recovery; exact-resume failure handling; stage checkpoint handoff; optimizer/scheduler/RNG restoration within an interrupted stage; optimizer/scheduler reset at architecture-stage boundaries; append-only events and atomic state writes where recorded.
- Preserve run packaging and reporting: signed manifests, run metadata, per-run logs, state files, progress/status output, final reports, bundle export/restore, and telemetry shard rehydration. Preserve the recorded restore ordering and empty-directory guard behavior.
- Copy the ledger’s scheduler defaults without substituting benchmark or estimator defaults:

  | Setting | Ledger value and meaning |
  |---|---|
  | Concurrency | `0` means unbounded, represented internally as `10**9`; a configured concurrency file is reread each loop. |
  | Active caps | Total and GPU caps `0` mean disabled. |
  | Repeats and retries | `repeat_count=5`; `max_retries=1`; retry limit `0` means uncapped. Preserve the recorded ADP/vanilla repeat dispatch distinction. |
  | Host pressure | Stop/resume available RAM `1024/1280 MiB`. |
  | GPU pressure | Stop/resume free VRAM `100/500 MiB`; device index `0`. |
  | Reserves and child gates | Host/GPU reserve `0/0 MiB`; child RSS/VRAM buffers `500/100 MiB`. |
  | External process recovery | Memory drop `500 MiB`, with disappearance of a previously tracked PID required for the main-runner gate reopen. |
  | Timing and backoff | Poll `0.5 sec`; post-launch hold `30 sec`; batch backoff factor `0.5`. |
  | CPU, swap, and child environment | CPU-only false; swap pressure/resume `70/70%` telemetry-only; CUDA visibility default `0`; allocator default `expandable_segments:True,max_split_size_mb:128`; shared CPU mode/workers `0/0`. |
  | Child architecture and epoch limits | Width/depth/neuron caps `512/5/1,000,000,000`; vanilla/ADP epoch limit `0` means unlimited. |
  | Seed | Base seed `42`, with deterministic SHA-256-derived job seeds. |
  | Grids | Vanilla depths `[1,2,3,4,5]`; widths `[1,2,4,8,16,32,64,128,256,512]`. Preserve the recorded ADP width/depth route grids and defaults. |
  | Standalone wrapper | Host and GPU pressure/resume `90/85%`; external drop `500 MiB`; poll `0.5 sec`; swap `70/70%` remains nongating. |

- Preserve the ledger’s precise edge behavior, including strict `<` pressure checks; host available memory not reduced by child RSS again; `fixed` and `gpu_first` selecting an eligible GPU before CPU, while `pressure_aware` requires both GPU and host predicates for GPU selection and then falls back to eligible CPU; resume launches not rechecked against capacity/gates/active caps; one victim paused per pass; OOM victim ordering; uncapped retry behavior; and distinction between central kill-and-relaunch and standalone OS suspension.
- Preserve recorded Linux runtime controls: CPU affinity detection/partitioning and per-job hashing; thread environment settings; worker count resolving to zero; best-effort process priority, memory locking, and no-swap/systemd scope controls. Windows process, PowerShell, and power-scheme behavior is out of scope per the later Linux-only user clarification.
- Keep model algorithms inside their existing implementations. The coordinator passes the one-epoch/full-batch smoke profile or explicit longer-run settings through the worker adapter; it does not alter ADP growth/search logic.

## Tests and acceptance

- Contract tests assert the generic coordinator’s manifest defaults and state transitions against explicit compatibility expectations.
- Exercise dry-run route enumeration for every current model/profile/baseline and every task in each registered OpenML suite; verify future providers can register without coordinator edits.
- Test deterministic identities, seeds, split hashes, manifest mismatch rejection, restart reconciliation, stage resume, retry limits, OOM classification/backoff, pressure pause/resume, external-drop reopening, gates, active caps, holds, FIFO order, and per-pass launch limits.
- Test all ledger-observed edge behavior, including resume admission bypass, scheduler-label equivalence, strict threshold equality, restore ordering/guard behavior, recovered-child unknown exit handling, and uncapped retry semantics.
- Test CPU-only and GPU worker environments, runtime-control fallbacks, child process-tree termination, bundle safety checks, telemetry rehydration, partial-run reports, and failure isolation.
- Smoke acceptance: every route can be planned and run with one epoch and full training-partition batches; preprocessing fits only on training rows; held-out test data remains untouched; results and resumable state are emitted. Do not run the suite as part of implementation unless separately requested.

## Assumptions

- “All datasets” means every task exposed by a registered provider; current built-in coverage is the existing OpenML `cc18`, `tabzilla`, and `ctr23` pools. Future sources must register explicitly.
- The hardness screen is diagnostic only, so it cannot silently remove dataset/model routes.
- Both generic-search profiles remain separate run identities. The fixed-grid aggregate and all 50 individual vanilla architectures are included.
- Ledger fidelity includes its documented quirks and limitations. The generic port reproduces them; any later corrected behavior would require a separately named, explicitly requested compatibility profile.
- The runner’s operational settings remain separate from each model’s algorithm and optimizer settings. The suite smoke profile overrides training to one epoch/full batch only for smoke runs.
</proposed_plan>
<oai-mem-citation>
<citation_entries>
MEMORY.md:816-843|note=[used prior central runner audit context and preserved static evidence boundaries]
</citation_entries>
<rollout_ids>
01996c41-90d0-7b1b-a45a-eae3b4e7d16d
</rollout_ids>
</oai-mem-citation>

## Source-fidelity correction recorded after the proposal

The scheduler-label shorthand above was inaccurate. The copied selector and
controller call site show that `fixed` and `gpu_first` choose an eligible GPU
without requiring the host-lane predicate; `pressure_aware` requires both GPU
and host predicates for GPU selection, then may fall back to eligible CPU.
Because CPU-lane readiness includes the CPU post-launch hold while GPU readiness
does not, `pressure_aware` can decline a GPU-ready launch during that hold.
The implementation follows the source and corrected ledger. A 55,296-state
differential test compares all valid discrete selector inputs with the copied
source. The plan summary above has been corrected to state the source behavior.
