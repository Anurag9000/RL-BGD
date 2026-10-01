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
