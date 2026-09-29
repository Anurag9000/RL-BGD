# Capability Ledger

Status vocabulary: COMPLETE, PARTIAL, BLOCKED, NOT STARTED. COMPLETE requires implementation, wiring, tests, and a runnable path where applicable.

| Capability | Status | Evidence / next requirement |
|---|---|---|
| Package metadata/install layout | COMPLETE | pyproject + src layout + smoke paths |
| Base CI | COMPLETE | tests/lint workflows |
| Coding standards | COMPLETE | Ruff/mypy/pre-commit config |
| Architecture documentation | COMPLETE | docs/architecture.md |
| Initial literature audit | PARTIAL | docs/literature_review.md has explicit backlog |
| Diagonal Gaussian posterior | COMPLETE | bayes/diagonal_gaussian.py + tests |
| FP32 Bayesian state | COMPLETE | posterior state tensors enforce FP32 |
| BGD Monte Carlo update | COMPLETE | bayes/bgd.py + quadratic/integration tests |
| K=1/2/4/8 support | COMPLETE | generic mc_samples; antithetic requires even K |
| Antithetic sampling | COMPLETE | sampler + pair-cancellation test |
| Sigma bounds/nonfinite failure | COMPLETE | PosteriorBounds + clamp/assert |
| Gaussian tempering | COMPLETE | bayes/tempering.py + exact tests |
| Posterior diagnostics | COMPLETE | sigma/precision/entropy/effective LR |
| Posterior/updater checkpoint | COMPLETE | versioned round-trip test |
| Generalized-Bayes RL | PARTIAL | formulation documented; BGD-SAC wiring pending |
| InformationAccessConfig | COMPLETE | central contract + leakage tests |
| Device auto/CUDA/CPU | COMPLETE | utils/device.py + tests |
| Deterministic seeding | COMPLETE | Python/NumPy/PyTorch CPU/CUDA |
| Diagonal quadratic benchmark | COMPLETE | arbitrary dimension + changing optimum/curvature |
| Rotated/non-diagonal quadratic | COMPLETE | dense SPD construction + test |
| Abrupt/smooth/recurring quadratic streams | COMPLETE | QuadraticStream + tests |
| Low-sigma movement mechanism | COMPLETE | identical-gradient movement test |
| Long-horizon tempering mechanism | COMPLETE | vanilla-vs-tempered uncertainty test |
| Synthetic benchmark family overall | PARTIAL | bandits still pending; LQR-style control now added |
| Continuous LQR-style smoke environment | COMPLETE | tensor-native finite-horizon control environment |
| SAC actor/twin critics/targets | COMPLETE | agents/sac + unit tests |
| Automatic entropy tuning | COMPLETE | SAC alpha optimizer + finite-update tests |
| SAC deterministic evaluation | COMPLETE | act(deterministic=True) + evaluation runner |
| SAC replay buffer | COMPLETE | device-aware buffer, IDs/usage/fresh metadata |
| terminated vs truncated bootstrap | COMPLETE | ReplayBatch.bootstrap_mask + test |
| SAC/replay checkpointing | COMPLETE | agent and replay round-trip tests |
| Stationary SAC learning smoke | COMPLETE | marked slow LQR learning test; deterministic return improves |
| BGD-SAC | NOT STARTED | phase 4 |
| Replay evidence accounting modes | NOT STARTED | phase 5 (metadata already present) |
| Surprise estimators | NOT STARTED | phase 8 |
| PPO + Adam/BGD | NOT STARTED | phase 9 |
| Recurrent agents/sequence replay | NOT STARTED | phase 11 |
| CARL | NOT STARTED | phase 7 |
| Continual World CW10/CW20 | NOT STARTED | phase 10 |
| ContinualBench/CORA adapter | NOT STARTED | optional external benchmark |
| EWC/Online-EWC/SI/MAS | NOT STARTED | baseline phase |
| Mechanistic experiments | PARTIAL | curvature + movement + long-horizon synthetic mechanisms executable |
| Paper tables/figures | NOT STARTED | phase 15 |
