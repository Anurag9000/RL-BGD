# Metrics

RL-BGD keeps generic continual metrics separate from benchmark-specific
definitions. In particular, the generic forward-transfer helper is not used as
a substitute for Continual World's canonical AUC-based forward transfer.

Implemented generic metrics include final average performance, diagonal
forgetting, backward transfer, reference-relative forward transfer, lifetime
AUC, plasticity retention, T80/T90-style time-to-fraction recovery,
post-change AUC, and revisit/reacquisition metrics.

For strict task-agnostic/recurrent CW20, recurrence is summarized separately
from the generic trace helper. CW20 is CW10 repeated twice. Evaluator-only task
names identify the repeated occurrences; they are never returned to the learner.
For each second occurrence, the evaluator uses that same occurrence's evaluation
column at three checkpoints: immediately after the corresponding first
occurrence was learned (reference), immediately before the revisit (zero-shot),
and immediately after the revisit stage (recovered). The paper records the
mean reference success, zero-shot success, recovered success, pre-revisit
change (zero-shot minus reference), and relearning gain (recovered minus
zero-shot). These scalars are not reported for canonical occurrence-head CW20,
where a different one-hot/head is intentionally assigned to each occurrence.

Change detection is evaluation-only. Known environment switch times may be
compared with detected surprise events after training has produced its signals.
The learning algorithm never receives those switch times. Event metrics include
detection delay, precision, recall, F1, false-positive rate, and false alarms
per million environment steps. Per-step surprise traces can additionally be
scored with AUROC using post-change windows as evaluation labels.

The adaptive synthetic stream runner records the causal chain used by the paper artifact pipeline: surprise, retention lambda, posterior sigma, effective learning rate, and available return/recovery traces. The Phase-15 builder creates diagnostic timelines and adaptation curves only from raw recorded columns; evaluator task/change labels are not injected into the learner to construct those plots.
