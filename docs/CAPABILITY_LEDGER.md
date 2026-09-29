# Capability Ledger

Status vocabulary: COMPLETE, PARTIAL, BLOCKED, NOT STARTED. COMPLETE requires implementation, wiring, tests, and a runnable path where applicable.

| Capability | Status | Evidence / next requirement |
|---|---|---|
| Package metadata/install layout | COMPLETE | pyproject + src layout + smoke path |
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
| Generalized-Bayes RL | PARTIAL | formulation documented; SAC/PPO wiring pending |
| InformationAccessConfig | COMPLETE | central contract + leakage tests |
| Device auto/CUDA/CPU | COMPLETE | utils/device.py + tests |
| Deterministic seeding | COMPLETE | Python/NumPy/PyTorch CPU/CUDA |
| Diagonal quadratic benchmark | COMPLETE | arbitrary dimension, changing optimum/curvature |
| Rotated/non-diagonal quadratic | COMPLETE | deterministic dense SPD construction + test |
| Abrupt quadratic stream | COMPLETE | QuadraticStream + runner/test |
| Smooth quadratic stream | COMPLETE | interpolated optimum/Hessian + test |
| Recurring quadratic stream | COMPLETE | cyclic schedule + test |
| Low-sigma movement mechanism | COMPLETE | exact identical-gradient movement test |
| Long-horizon tempering mechanism | COMPLETE | positive-curvature vanilla-vs-tempered test |
| Synthetic benchmark family overall | PARTIAL | bandits and optional LQR still pending |
| Replay evidence accounting | NOT STARTED | phase 5 |
| Surprise estimators | NOT STARTED | phase 8 |
| SAC + Adam | NOT STARTED | phase 3 |
| BGD-SAC | NOT STARTED | phase 4 |
| PPO + Adam/BGD | NOT STARTED | phase 9 |
| Recurrent agents/sequence replay | NOT STARTED | phase 11 |
| CARL | NOT STARTED | phase 7 |
| Continual World CW10/CW20 | NOT STARTED | phase 10 |
| ContinualBench/CORA adapter | NOT STARTED | optional external benchmark |
| EWC/Online-EWC/SI/MAS | NOT STARTED | baseline phase |
| Mechanistic experiments | PARTIAL | curvature + movement + long-horizon synthetic mechanisms now executable |
| Paper tables/figures | NOT STARTED | phase 15 |
