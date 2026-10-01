# Continual World Protocols

This repository separates the historical task-aware Continual World protocol from
strict task-agnostic variants so that task information cannot leak accidentally.

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

The original implementation used Meta-World v1 task IDs. The modern adapter maps
those names to the corresponding v3 IDs. CW20 is CW10 repeated twice, preserving
occurrence identity rather than collapsing repeated tasks.

## Historical task-aware path

The original Continual World implementation appends a one-hot sequence occurrence
identifier to observations. Its default multi-head architecture uses that code to
select actor and critic heads even when the shared network body hides the task ID.
The original training loop also observes the sequence index and can reset replay,
optimizers, or critics at task changes.

`CanonicalContinualWorldStreamEnv` preserves the environment-level task-aware
information contract: occurrence one-hot observations and explicit stage
truncations are visible. The exact historical multi-head/reset learner is tracked
separately and is not claimed complete yet.

## Strict TA-CW10 / TA-CW20

`ContinualWorldStreamEnv` is the strict task-agnostic path. Training receives:

- no task ID or one-hot suffix;
- no task name/index in reset or step info;
- no synthetic boundary flag when a stage changes;
- no task-dependent head selector;
- no algorithm callback at a switch;
- no task-specific replay routing or reset;
- no optimizer/posterior reset at a switch;
- no per-task normalization state.

Evaluation is allowed to know the benchmark stage. The TA runner uses a
post-step evaluation observer that receives no return value into the optimization
path. It evaluates separate environment instances, so evaluation resets and
randomness cannot mutate the training stream.

## Evaluation matrices

At each evaluator-known stage boundary the TA runner evaluates the deterministic
policy on every benchmark occurrence and records:

- mean-return stage-by-task matrix;
- success-rate stage-by-task matrix;
- final average performance;
- per-task and mean forgetting;
- backward transfer.

CW20 keeps both passes as separate columns/occurrences, which permits later
recurrence and reacquisition analysis without silently merging them.

## Commands

Install the optional benchmark stack:

```bash
pip install -e ".[continual-world]"
```

Run strict TA-CW10 with Adam SAC:

```bash
python scripts/run_ta_continual_world_sac.py --benchmark CW10 --optimizer adam
```

Run strict TA-CW20 with BGD-SAC:

```bash
python scripts/run_ta_continual_world_sac.py --benchmark CW20 --optimizer bgd
```

The default benchmark budget is one million environment steps per task occurrence.
These are expensive experiments. Implementation and smoke paths must not be
confused with executed paper results.
