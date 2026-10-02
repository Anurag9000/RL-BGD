# Metrics

RL-BGD keeps generic continual metrics separate from benchmark-specific
definitions. In particular, the generic forward-transfer helper is not used as
a substitute for Continual World's canonical AUC-based forward transfer.

Implemented generic metrics include final average performance, diagonal
forgetting, backward transfer, reference-relative forward transfer, lifetime
AUC, plasticity retention, T80/T90-style time-to-fraction recovery,
post-change AUC, and revisit/reacquisition metrics.

Change detection is evaluation-only. Known environment switch times may be
compared with detected surprise events after training has produced its signals.
The learning algorithm never receives those switch times. Event metrics include
detection delay, precision, recall, F1, false-positive rate, and false alarms
per million environment steps. Per-step surprise traces can additionally be
scored with AUROC using post-change windows as evaluation labels.

The adaptive synthetic stream runner records the causal chain used by the paper artifact pipeline: surprise, retention lambda, posterior sigma, effective learning rate, and available return/recovery traces. The Phase-15 builder creates diagnostic timelines and adaptation curves only from raw recorded columns; evaluator task/change labels are not injected into the learner to construct those plots.
