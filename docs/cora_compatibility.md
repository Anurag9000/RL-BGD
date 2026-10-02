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

CORA's develop branch remains a legacy stack whose historical benchmark
dependencies conflict with RL-BGD's Python 3.11+ modern PyTorch/Gymnasium base.
RL-BGD therefore keeps each legacy family in an isolated workflow rather than
downgrading the primary environment.

The pinned CORA revision is
`f2754bb282757829765beb4703f24b87efa13ff9`.

### Live-validated families

- **Atari:** a Python 3.10/Gym 0.25.2/ALE/AutoROM workflow creates CORA's own
  wrapped `PongNoFrameskip-v4` task and passes reset + real action step.
- **Procgen:** an isolated pinned Procgen workflow creates CORA's own Procgen
  task and passes reset + real action step.

### MiniHack compatibility

CORA's historical MiniHack wrapper assumes an old NLE implementation detail:
`self.env.env._vardir`. The pinned NLE/MiniHack stack that can still be built
today no longer guarantees that attribute. RL-BGD's smoke therefore preserves
the old working-directory behavior when `_vardir` exists and delegates
directly to the modernized child environment when it does not. This is a narrow
wrapper-compatibility bridge; it does not alter the MiniHack task, reward,
observation, or action semantics. Live reset/step validation remains the
authority for whether that bridge is sufficient.

### CHORES data blocker

The CHORES/ALFRED runtime code, pinned `crl_alfred` integration, Xvfb path,
official task factory, exact published trajectory lookup, and reset/step smoke
are implemented in a separate workflow. However, CORA's documented ~1 GB
trajectory archive URL is no longer downloadable. Upstream CORA issue #14,
opened 2025-12-22, reports this exact broken OneDrive URL and currently has no
maintainer response or replacement archive.

The CHORES workflow is therefore manual and data-explicit: it requires an
authoritative trajectory archive URL and optionally verifies a supplied
SHA-256. RL-BGD will not silently substitute the general ALFRED dataset or an
unverified mirror for CORA's curated trajectories.

These isolated workflows validate runtime compatibility only. They do not claim
reproduction of the published CORA training results, and none of the legacy
dependencies are installed into RL-BGD's modern base environment.
