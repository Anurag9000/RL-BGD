# Benchmark Protocols

Benchmark adapters are not yet implemented.

## Continual World
Canonical CW10/CW20 evaluation and strict task-agnostic variants remain distinct. Strict TA training receives no task ID, switch callback, task head routing, optimizer/posterior reset, per-task normalization, or task-routed replay.

## CARL
Context variables may define drift but are hidden from the policy in strict task-agnostic runs. Ground-truth context/switch metadata is evaluation-only unless an experiment is explicitly oracle.

## Hidden context / 3RL style
Recurrent Adam and recurrent BGD must share architecture so recurrence/context-inference gains are not misattributed to BGD.
