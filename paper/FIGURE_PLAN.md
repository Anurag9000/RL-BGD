# Figure Plan

1. Environment stream -> RL surrogate -> posterior update -> surprise/tempering -> sigma -> effective learning rate.
2. Quadratic curvature vs BGD c signal.
3. Adam vs BGD short-stream forgetting.
4. Long-stream sigma collapse/plasticity loss.
5. Tempered/adaptive uncertainty retention.
6. CARL drift adaptation.
7. CW10/CW20 aggregate metrics.
8. A->B->C->A recurrence.
9. Old sigma quantile vs future movement.
10. Perturbation importance.
11. Surprise -> lambda -> sigma -> update magnitude -> return recovery.
12. Compute/performance tradeoff.


## Generation status

The Phase-15 builder now generates figures only from canonical raw run
artifacts. Learning curves, stage-average adaptation/revisit curves, continual
performance matrices, diagnostic surprise/retention/sigma/effective-LR
timelines, final-performance intervals, and compute/performance tradeoffs are
automatic when their required raw columns exist. Phase-13 produces the
curvature, movement-vs-sigma, perturbation, freezing, and uncertainty-quality
mechanistic figures.

CW10/CW20, CARL, and other expensive final figures remain intentionally
run-dependent: they are not considered empirical results until the declared
multi-seed benchmark suites have produced complete canonical run directories.
