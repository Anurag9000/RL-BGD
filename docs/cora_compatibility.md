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
  a replacement. The provenance requirement is also explicit in CORA's own
  history: commit `4347f76d0abad6b50ab069efc509809c0c9ea1ba` states that
  repeatability requires the trajectories to be exactly the same and that they
  should be published rather than regenerated; later commits
  `de52a692a96a0353607d1550e0b2147d9fee0c53` and
  `474c32dd311eccaeb3b598332a1a41940757a8c7` introduce certified/validated
  2021 trajectories used by the benchmark metadata. RL-BGD therefore does not
  substitute older 2019 ALFRED trajectories, newly regenerated approximations,
  or fabricated images. The recovery workflow is manual and requires an
  authoritative archive URL, optionally a SHA-256. Before launching AI2-THOR it
  loads all four CHORES metadata files from the pinned CORA revision, requires
  the expected 27 unique train/valid_seen trajectory references, validates every
  referenced `traj_data.json`, validates the low-action/image indices consumed
  by `crl_alfred`, and requires every raw goal image named by those trajectories.
  The recovery workflow now defaults to CORA's official historical OneDrive URL
  and uses a repository-owned downloader that rejects HTML/login responses,
  empty/non-ZIP payloads, oversized archives, unsafe ZIP paths, and optional
  SHA-256 mismatches before extraction. Invalid archives are diagnosed using
  bounded prefix reads rather than loading an entire multi-gigabyte payload.
  Each attempt uses an independently allocated temporary file, avoiding
  staging-path collisions when recovery processes overlap. An authoritative replacement mirror can
  be supplied without code changes and its exact source URL, byte count, and
  SHA-256 are emitted as provenance. Only after that download gate and the
  complete 27-trajectory archive validator pass does the workflow run the exact
  published CORA trajectory under Xvfb.

These live smokes establish runtime compatibility; they do not reproduce
published CORA learning curves or make the legacy stack a dependency of the
modern repository.
