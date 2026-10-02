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

CORA's historical benchmark stack conflicts with RL-BGD's Python 3.11+
PyTorch/Gymnasium base, so every live compatibility check runs in an isolated
legacy environment. None of these dependencies are installed into the modern
RL-BGD process.

The isolated workflow pins CORA revision
`f2754bb282757829765beb4703f24b87efa13ff9` and validates real upstream task
construction, reset, action sampling, step, finite reward, and close behavior.

- **Atari**: validated with Python 3.10, NumPy 1.23.5, Gym 0.25.2,
  ALE/AutoROM, and CORA's wrapped `PongNoFrameskip-v4` task.
- **Procgen**: validated in the isolated legacy workflow with CORA's own
  Procgen task constructor and a real reset/step path.
- **MiniHack/NLE**: live-validated in Python 3.8 with NLE 0.9.0 and MiniHack
  0.1.5. CORA assumes historical private `_vardir` and three-argument seed
  behavior; RL-BGD confines a compatibility shim to the smoke, preserves the
  cwd hop when that private API exists, and delegates directly when it does
  not. Real upstream task construction, reset, sampled action, step, finite
  reward, and close all pass without changing task/reward/action semantics.
- **CHORES/ALFRED**: the runtime path is scripted separately in
  `.github/workflows/cora-chores.yml`, pinned to CORA and `crl_alfred`.
  Faithful execution requires CORA's regenerated 2021 trajectory archive,
  including raw goal images. The official OneDrive URL in CORA's installation
  guide currently fails, and upstream issue #14 ("Install Problem for
  Benchmarks") independently reports the same broken archive link and requests
  a replacement. RL-BGD therefore does not substitute older ALFRED trajectories
  or fabricated images. The recovery workflow is manual and requires an
  authoritative archive URL, optionally a SHA-256, then verifies the exact
  published CORA trajectory before running under Xvfb.

These live smokes establish runtime compatibility; they do not reproduce
published CORA learning curves or make the legacy stack a dependency of the
modern repository.
