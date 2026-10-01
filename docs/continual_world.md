# Continual World Protocols

This repository separates the published task-aware Continual World protocol from
strict task-agnostic variants so task information cannot leak accidentally.

## Canonical task order

CW10 follows the benchmark order:

1. hammer
2. push-wall
3. faucet-close
4. push-back
5. stick-pull
6. handle-press-side
7. push
8. shelf-place
9. window-close
10. peg-unplug-side

The original benchmark used Meta-World v1 task IDs. The modern adapter maps
those task names to the corresponding v3 IDs. CW20 is CW10 repeated twice and
preserves sequence-occurrence identity instead of collapsing repeated tasks.

## Canonical task-aware path

The canonical implementation is now wired end to end:

- `CanonicalContinualWorldStreamEnv` exposes the sequence occurrence through an
  appended one-hot and emits the published stage truncation behavior.
- `TaskAwareSACAgent` uses a shared feature body with occurrence-specific actor
  and critic heads. The shared body consumes only the physical observation; the
  one-hot suffix selects the appropriate head.
- `train_canonical_task_aware_sac` restarts the per-task exploration/update
  clock and, by default, resets FIFO replay and Adam state on every task change
  while retaining critic weights.
- `run_canonical_continual_world_sac` builds the modern Meta-World protocol,
  evaluates stage-by-task return and success matrices, isolates evaluator RNG
  state, and closes simulator resources deterministically.
- Canonical CW10/CW20 configs are in `configs/benchmarks/`.

These task-aware controls are deliberate benchmark-oracle information and must
not be reused in strict task-agnostic experiments.

## Strict TA-CW10 / TA-CW20

`ContinualWorldStreamEnv` is the strict task-agnostic path. Training receives:

- no task ID or one-hot suffix;
- no task name/index in reset or step info;
- no synthetic boundary flag solely because a stage changes;
- no task-dependent head selector;
- no algorithm callback at a hidden switch;
- no task-specific replay routing or reset;
- no optimizer/posterior reset at a hidden switch;
- no per-task normalization state.

The transition crossing a hidden task switch bootstraps from the next task's
actual reset observation. If the previous episode also ended naturally, that
observation is cached so the trainer's following reset does not reset the new
task twice.

The TA runner may use stage knowledge only inside evaluator-only bookkeeping.
Evaluation uses physically separate environments. Python, NumPy, PyTorch CPU,
and CUDA RNG states are restored after evaluator rollouts so evaluation cadence
cannot perturb the subsequent training trajectory.

## Evaluation matrices

Both canonical and strict TA runners record evaluator-known stage-by-task:

- mean return;
- success rate;
- final average performance;
- per-task and mean forgetting;
- backward transfer.

CW20 keeps the two CW10 passes as separate occurrences, allowing recurrence and
reacquisition analyses without silently merging them.

## Commands

Install the optional benchmark stack:

```bash
pip install -e ".[continual-world]"
```

Run canonical task-aware CW10:

```bash
python scripts/run_canonical_continual_world_sac.py --benchmark CW10
```

Run strict TA-CW10 with Adam SAC:

```bash
python scripts/run_ta_continual_world_sac.py --benchmark CW10 --optimizer adam
```

Run strict TA-CW20 with BGD-SAC:

```bash
python scripts/run_ta_continual_world_sac.py --benchmark CW20 --optimizer bgd
```

The default benchmark budget is one million environment steps per task
occurrence. These are expensive experiments. Runnable implementation, unit or
smoke validation, and executed paper-scale benchmark results are tracked as
separate evidence levels.
