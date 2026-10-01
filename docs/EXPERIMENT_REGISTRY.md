# Experiment Registry

No expensive experiment is marked executed until raw run artifacts exist.

| ID | Hypothesis | Status |
|---|---|---|
| SYN-Q1 | Positive curvature contracts posterior sigma | IMPLEMENTED / TESTED |
| SYN-Q2 | E[g epsilon] approaches H sigma in small-sigma diagonal quadratic | IMPLEMENTED / TESTED |
| A | Vanilla BGD can learn stationary RL tasks | IMPLEMENTED / TESTED on synthetic continuous control; large benchmark runs pending |
| B | BGD can reduce short-stream forgetting | NOT YET IMPLEMENTED |
| C | Vanilla BGD loses late-life plasticity under repeated evidence | NOT YET IMPLEMENTED |
| D | Fixed tempering prevents trivial sigma collapse | NOT YET IMPLEMENTED |
| E | Surprise-driven tempering adapts without boundary callbacks | NOT YET IMPLEMENTED |
| F | Replay reuse accelerates posterior overconfidence without correction | NOT YET IMPLEMENTED |
| G | Actor/critic Bayesianization have different tradeoffs | NOT YET IMPLEMENTED |
| H | Recurrence and Bayesian consolidation address complementary failures | NOT YET IMPLEMENTED |\n| CW-TA10 | Task-agnostic BGD-SAC can retain/adapt across CW10 without task identity | IMPLEMENTED / NOT YET EXECUTED |\n| CW-TA20 | Recurrence in CW20 exposes retention and reacquisition behavior under hidden task identity | IMPLEMENTED / NOT YET EXECUTED |
