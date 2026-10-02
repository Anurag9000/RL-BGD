# Benchmark Protocols

## CARL 1.1.1

RL-BGD's first CARL integration targets the published carl-bench 1.1.1 API.
The package exposes controllable physics contexts and returns observations as a
dictionary containing base state plus context. CARL selects a context on reset,
and its environment base exposes the active context and environment-specific
context update hook.

Strict task-agnostic RL-BGD runs remove both the context observation and
context_id metadata before data reaches the training loop. Context evolution is
driven by a global environment-step schedule; the agent receives no task ID or
switch callback.

The base schedule layer supports abrupt, smooth, recurring, periodic, random
walk, and multidimensional context trajectories. Initial runnable configs cover
CARLPendulum abrupt/smooth/recurring gravity and physical-parameter changes.

CARL 1.1.1 declares gymnasium<1.0.0, so CARL is maintained as an optional
benchmark environment rather than a base dependency. This incompatibility must
not be silently bypassed by forcing a newer Gymnasium into the same environment.

## Continual World

Canonical CW10/CW20 evaluation and strict task-agnostic variants remain
distinct. The canonical path appends an occurrence one-hot, routes shared-body
actor/critics through occurrence-specific heads, resets FIFO replay and Adam
state at task changes, keeps critic weights, and restarts the exploration/update
clock per task. These controls reproduce the published baseline defaults and
are intentionally unavailable to strict task-agnostic experiments.

Strict TA training receives no task ID, switch callback, task head routing,
optimizer/posterior reset, per-task normalization, or task-routed replay. Both
protocols use physically separate evaluation environments and preserve global
Python/NumPy/PyTorch RNG state across evaluator-only rollouts.

## Hidden context / 3RL style

Recurrent Adam and recurrent BGD must share architecture so recurrence/context
inference gains are not misattributed to BGD.


## ContinualBench

RL-BGD pins sail-sg/ContinualBench to a specific Git revision because the
repository is the authoritative implementation and its current package
metadata does not fully enumerate runtime dependencies. The source contract is
used instead of the README alone: current SawyerXYZEnv code returns a legacy
four-value step tuple with a reward dictionary, while the README documents a
five-value Gymnasium-style tuple. The adapter accepts both.

The active reward/task is selected inside the environment wrapper. In strict
mode, task name, task index, task success, and reward-dictionary identity are
not exposed to the agent. Hidden switches may be success-triggered,
fixed-budget, or success-or-budget. The evaluator can inspect the active task
through evaluation_context.

## CORA

CORA is retained as a protocol and metric compatibility target. RL-BGD now
implements dependency-free representations of the canonical Atari and Procgen
sequential schedules plus CORA-style isolated forgetting and isolated zero-shot
forward-transfer calculations directly from numeric evaluation traces.

The legacy CORA environment runtime remains intentionally isolated from the
modern Python/Gymnasium base. Dedicated workflows now pass real pinned-upstream
reset/step smokes for all three video-game families: Atari
(`PongNoFrameskip-v4` with ALE/AutoROM), Procgen, and NetHack/MiniHack.
MiniHack uses a narrowly scoped compatibility bridge for CORA's historical
private `_vardir`/seed assumptions; the bridge delegates directly when the
installed NLE no longer exposes that private directory and does not change task,
reward, observation, or action semantics.

CHORES/ALFRED remains separate. Its pinned runtime, Xvfb path, `crl_alfred`
integration, and reset/step smoke are implemented. The recovery workflow derives
the required corpus from CORA's four pinned CHORES metadata files and fail-closes
unless all 27 unique train/valid_seen trajectories, their trajectory JSON, and
all raw goal images consumed by `crl_alfred` are present. CORA's documented
regenerated-trajectory archive is no longer available from its historical
OneDrive URL, so the workflow requires an authoritative replacement URL (and
optionally verifies its SHA-256) rather than silently substituting generic
ALFRED data.
