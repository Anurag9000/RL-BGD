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

Recurrent BGD-SAC has passed stationary learning acceptance. Recurrent
BGD-PPO remains intentionally open after both actor+value and actor-only BGD
failed the first naturally stable LQR acceptance; its replacement acceptance
uses a matched recurrent-Adam control on a control-requiring stationary LQR.

ContinualBench now has a strict hidden-task adapter matching its current source
contract, including reward dictionaries and both four-/five-value step APIs.
Its real pinned live smoke remains pending after the first attempt exposed and
fixed a project-metadata direct-reference issue. CORA has been source-audited;
its legacy runtime pins are not installed into the modern base environment.
