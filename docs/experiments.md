# Experiments

Implementation proceeds from mathematical mechanism tests to stationary RL,
controlled nonstationarity, recurrent task-agnostic agents, and finally large
external benchmarks. Passing imports is never empirical evidence, and a
runnable benchmark path is not labelled as an executed result without raw run
artifacts.

## Current evidence levels

Synthetic and controlled tests currently exercise BGD posterior mechanics,
tempering, replay-evidence accounting, surprise-driven retention, feed-forward
SAC/PPO, recurrent PPO, recurrent SAC, and recurring hidden-context LQR streams.
Stationary learning acceptance exists for the core feed-forward SAC/PPO paths.

Canonical task-aware Continual World and strict TA-CW10/TA-CW20 now have
runnable Meta-World adapters, training runners, evaluation matrices, configs,
RNG isolation, and simulator cleanup. The full one-million-step-per-occurrence
CW runs remain **not executed** in the repository evidence registry.

CARL's strict hidden-context adapter is now exercised against the real pinned
carl-bench 1.1.1 package in an isolated workflow. The live Pendulum smoke
validates reset/step behavior, scheduled context changes, and context stripping
without importing CARL's legacy Gymnasium constraints into the base environment.

EWC, Online EWC, Synaptic Intelligence, and MAS now have actor/critic SAC
integration. The default comparison uses update-count-driven consolidation with
no task ID or boundary input; a separate oracle-boundary trainer exists and is
labelled as privileged. Both share the same regularizer implementation.

Recurrent BGD-SAC and recurrent BGD-PPO have both passed stationary learning
acceptance under matched recurrent controls. The recurrent PPO acceptance was
reworked after the first naturally stable LQR criterion proved uninformative;
the current gate uses a control-requiring stationary LQR with a matched
recurrent-Adam comparison rather than accepting a trivial stable policy.

ContinualBench now has a strict hidden-task adapter matching its pinned source
contract, including reward dictionaries and both four-/five-value step APIs.
The pinned live reset/step workflow passes. Compatibility handling is confined
to the adapter boundary: reachable missing assets are restored from canonical
Meta-World paths, an upstream debug-only undefined symbol is supplied without
changing reward computation, and the pinned no-op close stub is tolerated.

CORA now has dependency-free canonical protocol metadata plus isolated
forgetting and isolated zero-shot forward-transfer metrics with tests. Its
legacy Atari environment runtime is also validated in a separate Python 3.10
workflow against pinned CORA revision `f2754bb282757829765beb4703f24b87efa13ff9`.
The smoke constructs CORA's own wrapped `PongNoFrameskip-v4` task, resets it,
and steps it under Gym 0.25.2/ALE while keeping all legacy dependencies outside
the modern RL-BGD environment.


UCL-PPO is implemented as an explicitly oracle-boundary comparator because the
original method snapshots the previous-task posterior at known task boundaries.
FOO-VB Diagonal is represented through an exact equivalence contract with the
eta=1 untempered diagonal BGD update rather than duplicated numerical code.

Phase 13-15 infrastructure now includes a reproducible mechanistic analysis
suite, curated paper experiment manifests/launchers, and automatic raw-run
aggregation with deterministic bootstrap confidence intervals. These
infrastructure paths are distinct from expensive experiment execution: paper
claims remain pending until corresponding raw run artifacts exist.
