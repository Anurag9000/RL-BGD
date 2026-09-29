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
| Separate mean/evidence gradient channels | COMPLETE | BGDLoss + replay-evidence mechanism tests |
| Dynamic per-update retention override | COMPLETE | BGDUpdater retention override + tempering test |
| K=1/2/4/8 support | COMPLETE | generic mc_samples; antithetic requires even K |
| Antithetic sampling | COMPLETE | sampler + pair-cancellation test |
| Sigma bounds/nonfinite failure | COMPLETE | PosteriorBounds + clamp/assert |
| Gaussian tempering | COMPLETE | bayes/tempering.py + exact tests |
| Posterior diagnostics | COMPLETE | sigma/precision/entropy/effective LR |
| Posterior/updater checkpoint | COMPLETE | versioned round-trip test |
| Generalized-Bayes RL | PARTIAL | formulation + SAC mapping; evidence temperature study pending |
| InformationAccessConfig | COMPLETE | central contract + leakage tests |
| Device auto/CUDA/CPU | COMPLETE | utils/device.py + tests |
| Deterministic seeding | COMPLETE | Python/NumPy/PyTorch CPU/CUDA |
| Diagonal quadratic benchmark | COMPLETE | arbitrary dimension + changing optimum/curvature |
| Rotated/non-diagonal quadratic | COMPLETE | dense SPD construction + test |
| Abrupt/smooth/recurring quadratic streams | COMPLETE | QuadraticStream + tests |
| Generic context schedules | COMPLETE | abrupt/smooth/periodic/random-walk/recurring + tests |
| Low-sigma movement mechanism | COMPLETE | identical-gradient movement test |
| Long-horizon tempering mechanism | COMPLETE | vanilla-vs-tempered uncertainty test |
| Synthetic benchmark family overall | PARTIAL | bandits pending; LQR-style control added |
| Continuous LQR-style smoke environment | COMPLETE | tensor-native finite-horizon control environment |
| Boundary-free nonstationary LQR stream | COMPLETE | ScheduledLQREnv + leakage/unit test |
| SAC actor/twin critics/targets | COMPLETE | agents/sac + unit tests |
| Automatic entropy tuning | COMPLETE | SAC alpha optimizer + finite-update tests |
| SAC deterministic evaluation | COMPLETE | deterministic policy runner |
| SAC replay buffer | COMPLETE | device-aware buffer, IDs/usage/fresh metadata |
| terminated vs truncated bootstrap | COMPLETE | bootstrap_mask + test |
| SAC/replay checkpointing | COMPLETE | round-trip tests |
| Stationary SAC learning smoke | COMPLETE | slow deterministic-return learning test |
| BGD-SAC critic-only | COMPLETE | mean-target semantics + mode test + stationary learning test |
| BGD-SAC actor-only | COMPLETE | functional-call sampled actor update + mode test |
| BGD-SAC actor+critic | COMPLETE | both posterior paths + checkpoint round-trip |
| BGD-SAC posterior diagnostics | COMPLETE | sigma/effective-LR metrics per Bayesian module |
| Replay evidence all_replay | COMPLETE | evidence module + BGD-SAC wiring |
| Replay evidence fresh_only_uncertainty | COMPLETE | separate gradient channel + zero-evidence sigma test |
| Replay evidence inverse_reuse_weight | COMPLETE | usage-weight tests + synthetic precision comparison |
| Replay evidence normalized_batch_evidence | COMPLETE | uniform batch-scale test + BGD-SAC wiring |
| Replay evidence diagnostics / ESS | COMPLETE | per-update metrics |
| Fixed controlled forgetting | COMPLETE | exact Gaussian tempering + BGD fixed retention |
| TD surprise estimator | COMPLETE | online normalized TD surprise + tests |
| TD adaptive retention in BGD-SAC | COMPLETE | no-boundary SAC wiring + checkpoint state + metrics |
| Ensemble-disagreement surprise primitive | COMPLETE | estimator + state API; adaptive-agent wiring pending |
| Predictive-NLL surprise primitive | COMPLETE | estimator + state API; world-model wiring pending |
| Final average / forgetting / BWT / generic FWT metrics | COMPLETE | metrics/continual.py + tests |
| Lifetime AUC / plasticity retention | COMPLETE | metrics/continual.py + tests |
| T80/T90 primitive / post-change AUC / recurrence metrics | COMPLETE | metrics/adaptation.py + tests |
| Surprise change-detection event metrics | COMPLETE | delay/FPR/precision/recall/F1/false alarms + tests |
| Surprise AUROC primitive | COMPLETE | rank-based binary AUROC + tests |
| Adaptive surprise causal timeline runner | COMPLETE | adaptive BGD-SAC on recurring LQR + slow integration test |
| CARL 1.1.1 strict hidden-context adapter | PARTIAL | adapter/config/unit tests; real CARL smoke pending |
| CARL abrupt/smooth/recurring configs | PARTIAL | schedule semantics complete; real CARL smoke pending |
| PPO Adam baseline | COMPLETE | clipped surrogate + GAE + minibatch epochs + value clipping + checkpoint + stationary LQR learning smoke |
| BGD-PPO actor/value/both | PARTIAL | implementation + posterior checkpoint + finite-mode tests + stationary runner; stationary BGD learning acceptance test pending |
| BGD-PPO repeated-rollout evidence accounting | COMPLETE | first-epoch-only default plus all-epochs/normalized experimental modes + tests |
| PPO task-agnostic recurring synthetic stream | COMPLETE | Adam/BGD recurring LQR runner + integration test without task ID/boundary input |
| PPO rollout checkpointing | COMPLETE | partial/update-ready behavior-policy statistics + GAE state round-trip test |
| PPO + Adam/BGD overall | PARTIAL | phase 9 core complete; close after stationary BGD-PPO learning validation |
| Recurrent agents/sequence replay | NOT STARTED | phase 11 |
| Continual World CW10/CW20 | NOT STARTED | phase 10 |
| ContinualBench/CORA adapter | NOT STARTED | optional external benchmark |
| EWC/Online-EWC/SI/MAS | NOT STARTED | baseline phase |
| Mechanistic experiments | PARTIAL | curvature + movement + long-horizon + replay evidence + surprise timeline |
| Paper tables/figures | NOT STARTED | phase 15 |
