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

CORA's current develop branch remains a legacy stack: its package metadata pins
gym[atari]<=0.25.2, atari-py==0.2.5, and setuptools==59.5.0, while RL-BGD
targets Python 3.11+ and modern PyTorch / Gymnasium benchmark integrations.

Therefore RL-BGD does not install CORA into the base or all environment. A
future native runtime comparison should run CORA in a separately pinned legacy
environment and exchange only declared artifacts/metrics. This is reported as a
compatibility constraint rather than hidden by dependency overrides.
