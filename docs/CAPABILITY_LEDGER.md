# Capability Ledger

Status vocabulary: COMPLETE, PARTIAL, BLOCKED, NOT STARTED. COMPLETE requires implementation, wiring, tests, and a runnable path where applicable.

| Capability | Status | Evidence / next requirement |
|---|---|---|
| Package metadata/install layout | COMPLETE | pyproject + src layout + smoke paths |
| Base CI | COMPLETE | tests/lint workflows |
| Coding standards | COMPLETE | Ruff/mypy/pre-commit config |
| Architecture documentation | COMPLETE | docs/architecture.md |
| Paper method/limitations documentation | COMPLETE | paper/METHOD.md specifies implemented posterior/RL/replay/tempering/recurrence/statistics semantics; paper/LIMITATIONS.md records Bayesian, replay, benchmark, statistical, and external-validity limits |
| Core literature authority | COMPLETE | docs/literature_review.md now covers BGD/FOO-VB/generalized Bayes, EWC/Online-EWC/SI/MAS/UCL, controlled forgetting/plasticity, CRL benchmarks/world models, exploration, and replay-evidence caveats; residual 2026 additions remain a living audit |
| Diagonal Gaussian posterior | COMPLETE | bayes/diagonal_gaussian.py + tests |
| FP32 Bayesian state | COMPLETE | posterior state tensors enforce FP32 |
| BGD Monte Carlo update | COMPLETE | bayes/bgd.py + quadratic/integration tests |
| FOO-VB diagonal equivalence reference | COMPLETE | exact diagonal update equivalence baseline + regression tests |
| Separate mean/evidence gradient channels | COMPLETE | BGDLoss + replay-evidence mechanism tests |
| Dynamic per-update retention override | COMPLETE | BGDUpdater retention override + tempering test |
| K=1/2/4/8 support | COMPLETE | generic mc_samples; antithetic requires even K |
| Antithetic sampling | COMPLETE | sampler + pair-cancellation test |
| Sigma bounds/nonfinite failure | COMPLETE | PosteriorBounds + clamp/assert |
| Gaussian tempering | COMPLETE | bayes/tempering.py + exact tests |
| Posterior diagnostics | COMPLETE | sigma/precision/entropy/effective LR |
| Posterior/updater checkpoint | COMPLETE | versioned round-trip test |
| Generalized-Bayes RL | PARTIAL | evidence temperature is a real BGD power with matched SAC sweep + math tests; latest CI validation pending |
| InformationAccessConfig | COMPLETE | central contract + leakage tests |
| Device auto/CUDA/CPU | COMPLETE | utils/device.py + tests |
| Deterministic seeding | COMPLETE | Python/NumPy/PyTorch CPU/CUDA |
| Diagonal quadratic benchmark | COMPLETE | arbitrary dimension + changing optimum/curvature |
| Rotated/non-diagonal quadratic | COMPLETE | dense SPD construction + test |
| Abrupt/smooth/recurring quadratic streams | COMPLETE | QuadraticStream + tests |
| Generic context schedules | COMPLETE | abrupt/smooth/periodic/random-walk/recurring + tests |
| Low-sigma movement mechanism | COMPLETE | identical-gradient movement test |
| Long-horizon tempering mechanism | COMPLETE | vanilla-vs-tempered uncertainty test |
| Synthetic benchmark family overall | COMPLETE | quadratic streams + hidden-context Gaussian bandit with oracle regret + nonstationary LQR |
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
| Ensemble-disagreement surprise primitive | COMPLETE | twin-critic BGD-SAC adaptive-retention wiring + state/checkpoint tests |
| Predictive-NLL adaptive retention | COMPLETE | online Gaussian transition/reward model + pre-update NLL surprise + BGD-SAC retention wiring + checkpoint/tests |
| Final average / forgetting / BWT / generic FWT metrics | COMPLETE | metrics/continual.py + tests |
| Lifetime AUC / plasticity retention | COMPLETE | metrics/continual.py + tests |
| T80/T90 primitive / post-change AUC / recurrence metrics | COMPLETE | metrics/adaptation.py + tests |
| Surprise change-detection event metrics | COMPLETE | delay/FPR/precision/recall/F1/false alarms + tests |
| Surprise AUROC primitive | COMPLETE | rank-based binary AUROC + tests |
| Adaptive surprise causal timeline runner | COMPLETE | adaptive BGD-SAC on recurring LQR + slow integration test |
| CARL 1.1.1 strict hidden-context adapter | COMPLETE | isolated real CARL 1.1.1 reset/step smoke + strict context stripping |
| CARL abrupt/smooth/recurring configs | COMPLETE | v1.1.1-valid context keys + live Pendulum schedule smoke |
| PPO Adam baseline | COMPLETE | clipped surrogate + GAE + minibatch epochs + value clipping + checkpoint + stationary LQR learning smoke |
| BGD-PPO actor/value/both | COMPLETE | feed-forward BGD-PPO + stationary learning acceptance |
| BGD-PPO repeated-rollout evidence accounting | COMPLETE | first-epoch-only default plus all-epochs/normalized modes + tests |
| PPO task-agnostic recurring synthetic stream | COMPLETE | Adam/BGD recurring LQR runner without task ID/boundary input |
| PPO rollout checkpointing | COMPLETE | partial/update-ready behavior statistics + GAE round trip |
| PPO + Adam/BGD overall | COMPLETE | stationary and recurring task-agnostic validation |
| Recurrent PPO Adam | COMPLETE | GRU actor/value + sequence rollout + checkpoint + smoke test |
| Recurrent BGD-PPO | COMPLETE | matched GRU architecture + posterior/update/checkpoint tests + matched Adam/BGD stationary learning acceptance |
| Recurrent sequence rollout semantics | COMPLETE | behavior hidden snapshots + episode masks + truncated-BPTT chunks |
| Recurrent hidden-state task-leakage guard | COMPLETE | recurring LQR runner verifies resets only on episode end |
| Recurrent SAC / sequence replay | COMPLETE | recurrent Adam SAC + burn-in/unroll sequence replay + hidden-state lifecycle + recurring LQR integration test |
| Recurrent BGD-SAC | COMPLETE | actor/critic/all posterior modes + evidence accounting + checkpoint + recurring LQR integration + stationary learning acceptance |
| Continual World canonical task-aware protocol | PARTIAL | modern Meta-World stream + task-ID multihead SAC + published replay/optimizer lifecycle + matrix runner/configs; full long benchmark runs pending |
| Continual World strict task-agnostic CW10/CW20 | PARTIAL | hidden-ID stream + protocol bundles + SAC matrix runner; full long benchmark runs pending |
| ContinualBench adapter | COMPLETE | strict reward/task stream + 4/5-step compatibility + asset repair + pinned-runtime debug/close compatibility shims + unit tests + live pinned-package smoke |
| CORA metric/protocol compatibility | COMPLETE | canonical Atari/Procgen sequence metadata + isolated-forgetting/zero-shot-forward-transfer formulas + tests |
| CORA legacy environment runtime | BLOCKED | upstream develop stack pins gym<=0.25.2/atari-py/setuptools versions incompatible with the modern base; keep isolated rather than downgrade primary runtime |
| EWC/Online-EWC/SI/MAS | COMPLETE | actor/critic SAC wiring + fixed-update task-agnostic consolidation + oracle-boundary protocol + checkpoint/tests |
| UCL-PPO oracle-boundary baseline | COMPLETE | independent Bayesian hidden-layer PPO + UCL saved-posterior regularizer + explicit boundary snapshots + checkpoint/unit/integration tests; original task-ID dependence is not mislabelled task-agnostic |
| FOO-VB diagonal baseline mapping | COMPLETE | exact eta=1 untempered diagonal BGD equivalence + regression test; generalized-Bayes RL scope explicitly distinguished from original likelihood objective |
| External baseline phase | COMPLETE | EWC/Online-EWC/SI/MAS/UCL plus FOO-VB diagonal reference are implemented, documented, and information-access-labelled; structured matrix-variate FOO-VB remains an optional extension rather than a required duplicate |
| 3RL-style recurrent Continual World | PARTIAL | strict hidden-ID recurrent Adam/BGD/adaptive-BGD CW10/CW20 runners + source-verified protocol/deviation docs; full long benchmark runs pending |
| CARL SAC/BGD training path | PARTIAL | strict hidden-context Adam/BGD/adaptive-BGD runner + isolated live benchmark test; latest training-path CI execution pending |
| Mechanistic experiments | PARTIAL | Phase-13 curvature/movement/perturbation/freezing/uncertainty-quality artifact engine + replay/surprise mechanisms implemented; full external comparative runs pending |
| Curated paper suites | PARTIAL | smoke/dev/CARL/CW10/CW20/task-agnostic/ablation/uncertainty/mechanism/compute manifests + strict canonical run conversion; end-to-end smoke CI pending |
| Canonical raw-run artifact schema | COMPLETE | manifest/config/metrics/summary schema + strict loader/provenance hashes + failed-run status + converter tests |
| Paper tables/figures | PARTIAL | canonical raw-run aggregation, scalar/task bootstrap statistics, matched-seed differences, provenance/information-access tables, multi-format figures/tables; final artifacts await full benchmark runs |
