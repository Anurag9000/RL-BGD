# Experiment Registry

No expensive experiment is marked executed until raw run artifacts exist.

| ID | Hypothesis | Status |
|---|---|---|
| SYN-Q1 | Positive curvature contracts posterior sigma | IMPLEMENTED / TESTED |
| SYN-Q2 | E[g epsilon] approaches H sigma in small-sigma diagonal quadratic | IMPLEMENTED / TESTED |
| A | Vanilla BGD can learn stationary RL tasks | IMPLEMENTED / TESTED + five-seed SAC/PPO Adam-vs-BGD `stationary_core` jobs defined; execution pending |
| B | BGD can reduce short-stream forgetting | EWC/Online-EWC/SI/MAS + matched unregularized SAC control are five-seed `baseline_core` jobs; bounded dev comparators also IMPLEMENTED; execution pending |
| C | Vanilla BGD loses late-life plasticity under repeated evidence | CONTROLLED late-shift runner + causal regression test + matched vanilla/tempered five-seed ablation jobs IMPLEMENTED; multi-seed execution pending |
| D | Fixed tempering prevents trivial sigma collapse | IMPLEMENTED / TESTED mechanism + explicit fixed-retention ablation jobs; multi-seed execution pending |
| E | Surprise-driven tempering adapts without boundary callbacks | IMPLEMENTED / TESTED on recurring LQR with matched five-seed none/TD/ensemble/predictive surprise-source ablation plus CARL schedules; external multi-seed execution pending |
| F | Replay reuse accelerates posterior overconfidence without correction | four replay-evidence modes IMPLEMENTED / TESTED + matched ablation jobs defined; multi-seed execution pending |
| G | Actor/critic Bayesianization have different tradeoffs | actor-only, critic-only, and actor+critic modes IMPLEMENTED / TESTED + matched ablation jobs defined; multi-seed execution pending |
| GB-T | Generalized-Bayes evidence temperature changes evidence strength independently of eta/retention | IMPLEMENTED / TESTED math + matched per-temperature SAC jobs defined; multi-seed execution pending |
| H | Recurrence and Bayesian consolidation address complementary failures | five-way matched hidden-context SAC suite IMPLEMENTED / TESTED plus matched CW20 recurrent Adam/BGD/adaptive-BGD controls; multi-seed comparative study pending |
| CW-CAN10 | Canonical task-aware SAC reproduces the CW10 information/lifecycle protocol | IMPLEMENTED / NOT YET FULLY EXECUTED |
| CW-CAN20 | Canonical task-aware SAC reproduces the CW20 occurrence-aware protocol | IMPLEMENTED / NOT YET FULLY EXECUTED |
| CW-TA10 | Task-agnostic BGD-SAC can retain/adapt across CW10 without task identity | IMPLEMENTED / NOT YET FULLY EXECUTED |
| CW-TA20 | Recurrence in CW20 exposes retention and reacquisition behavior under hidden task identity | IMPLEMENTED with matched strict task-agnostic Adam/BGD controls and recurrent Adam/BGD/adaptive-BGD controls; NOT YET FULLY EXECUTED |

| I | Posterior uncertainty predicts perturbation importance | MECHANISTIC PIPELINE IMPLEMENTED / TESTED + five seeded artifact replicates defined; multi-seed execution pending |
| J | Posterior sigma predicts future parameter movement | MECHANISTIC PIPELINE IMPLEMENTED / TESTED + five seeded artifact replicates defined; multi-seed execution pending |
| UCL | Oracle-boundary UCL-PPO provides a documented Bayesian continual-RL comparator | IMPLEMENTED / TESTED + five-seed UCL and phase-matched Adam-PPO oracle control in `baseline_core`; execution pending |
