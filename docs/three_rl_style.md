# 3RL-style task-agnostic Continual World

This repository includes a recurrent strict task-agnostic Continual World path
inspired by Caccia et al., Task-Agnostic Continual Reinforcement Learning:
Gaining Insights and Overcoming Challenges (CoLLAs 2023).

## Source-verified reference protocol

The official 3RL configuration declares experience replay, a recurrent context
model, no task ID, and task-agnostic training. Its Meta-World configuration
uses SAC learning rate 3e-4, actor and critic hidden sizes 400 by 400,
automatic entropy tuning, history length 15, context dimension 30, replay
capacity 10,000,000, CW10 total environment steps 10,000,000, maximum episode
length 200, and batch size 1028.

The official context RNN consumes previous action, previous reward, and previous
observation over the history window, then concatenates the resulting context
with the current observation.

Reference implementation:
https://github.com/amazon-science/replay-based-recurrent-rl

## RL-BGD implementation

The recurrent task-agnostic runner preserves the information assumptions:

- no task ID;
- no task-boundary callback to the learner;
- no task-specific heads;
- no per-task replay, optimizer, or posterior reset;
- replay persists across hidden task changes;
- recurrent hidden state resets only on natural episode reset;
- evaluator-only stage knowledge is used solely to construct performance
  matrices, and the online training recurrent state is restored after
  evaluation.

The recurrent input contains the current observation plus previous action,
previous reward, and previous done. The GRU accumulates longer history online,
while sequence replay uses explicit burn-in and truncated unrolling.

## Important deviations

This path is deliberately named 3RL-style rather than an exact reproduction.

1. The repository uses the current Meta-World v3 adapter instead of the older
   Meta-World stack used by the original code.
2. RL-BGD integrates the GRU into recurrent actor and critic sequence models.
   The original code used a separate fixed-window context GRU whose output was
   concatenated with the current observation.
3. RL-BGD sequence replay represents history through chronological windows
   with explicit burn-in and unroll semantics rather than the original
   flattened history-buffer representation.
4. Recurrent BGD-SAC and recurrent adaptive-BGD-SAC are new comparison arms
   and are not claimed as methods from the 3RL paper.

Run examples:

    python scripts/run_recurrent_continual_world.py --benchmark CW10 --optimizer adam
    python scripts/run_recurrent_continual_world.py --benchmark CW10 --optimizer bgd
    python scripts/run_recurrent_continual_world.py --benchmark CW20 --optimizer adaptive_bgd
