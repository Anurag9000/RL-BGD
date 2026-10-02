# CORA Compatibility

RL-BGD treats CORA as two separable targets: metric/protocol compatibility and
legacy runtime compatibility.

## Metric compatibility

The dependency-free rl_bgd.metrics.cora module reproduces the current CORA
develop branch's isolated-forgetting and isolated zero-shot forward-transfer
semantics. It deliberately preserves CORA's strict region selection
(x > low, x < high) and its task-specific normalization by the maximum absolute
return observed for that task across runs.

Canonical metadata is included for the published CORA Atari 6-task/5-cycle and
Procgen 6-task/5-cycle sequences, including their source step budgets and task
orders.

This bridge operates on numeric evaluation traces directly. It does not require
TensorFlow summary readers, Plotly, or CORA's event-file layout.

## Runtime compatibility

CORA's develop branch remains a legacy stack whose historical Gym/Atari and
setuptools constraints conflict with RL-BGD's Python 3.11+ modern
PyTorch/Gymnasium base. RL-BGD therefore does not install CORA into the base or
"all" environment.

A dedicated isolated workflow now validates one real upstream execution path.
It pins CORA revision `f2754bb282757829765beb4703f24b87efa13ff9` in Python
3.10 with NumPy 1.23.5, Gym 0.25.2, CPU PyTorch, ALE/AutoROM, and accepted Atari
ROMs. The smoke creates CORA's own wrapped `PongNoFrameskip-v4` task, resets
the environment, samples an action, executes a step, verifies a finite reward
and observation/info contract, and closes the environment. That workflow is
green.

This establishes isolated **Atari** runtime compatibility only. It does not
establish that CORA's Procgen, NetHack, CHORES, or complete historical
dependency matrix can be exercised together, and it does not make the legacy
stack a dependency of the modern RL-BGD process. Those broader runtime families
remain intentionally isolated rather than inferred from a single Atari smoke.
