# Curated paper experiment suites

Phase 14 provides one registry and launcher for the paper-oriented experiment
portfolio.

Registered suites:

- smoke
- stationary_core
- dev
- baseline_core
- carl_core
- cw10_core
- cw20_final
- task_agnostic_final
- ablation_core
- uncertainty_analysis
- mechanism_analysis
- compute_analysis

Each job records its hypothesis ID, callable target, exact kwargs, seeds,
algorithm, environment, information protocol, source config where applicable,
primary/secondary metrics, explicit comparison group, optional dependency extra,
runtime class, and notes. Suite revision 3 also records fully resolved
runner-call defaults and makes paired statistics opt-in: jobs are paired only
when they deliberately share the same comparison group,
protocol, benchmark, seed set, and primary outcome. Jobs without a declared
comparison family remain aggregate-only and cannot be cross-paired implicitly.
Hidden-context methods and evidence-temperature values are separate jobs, so
seed-level uncertainty is computed per experimental condition rather than over
a composite JSON blob.

Dry-run manifest generation:

    python scripts/run_paper_suite.py smoke --output-root artifacts/suites

Execution:

    python scripts/run_paper_suite.py smoke --output-root artifacts/suites --execute

Execution resumes strict matching successes by default. Use `--no-resume` to
force reruns. Large suites can be partitioned across independent workers by
repeating `--job-id`, `--seed`, or exact expanded `--run-id` filters. The
filters intersect and are validated before training begins; unknown selectors
fail closed. Each filtered invocation writes a selection-specific execution
summary, while the complete suite manifest is written atomically. This allows
Slurm arrays or one-process-per-GPU launchers to share an output root without
mixing run directories or overwriting the full-suite summary.

GPU-first parallel execution is also provided directly:

    python scripts/run_paper_suite_parallel.py cw10_core \
      --output-root artifacts/suites \
      --gpu-ids 0,1,2,3 \
      --min-free-vram-mb 12000 \
      --max-gpu-utilization 25

The parallel launcher materializes the same canonical suite and invokes the
existing one-run launcher for each selected expanded run; it does not bypass
resume, supersession archives, strict artifacts, or provenance checks. By
default it creates one worker per detected GPU and sets a per-child
`CUDA_VISIBLE_DEVICES`, so an `device=auto` runner sees only its assigned GPU.
If no GPU is visible it falls back to one CPU worker. `--workers-per-gpu` and
`--cpu-workers` are explicit opt-ins for higher concurrency. Optional free-VRAM
and utilization thresholds are polled before a GPU worker launches its next
run (30 s default). Worker exceptions and non-zero child exits are retained in
`parallel_execution_summary.json`; rerunning the command safely resumes strict
matching successes.

Successful jobs are converted into the canonical Phase-15 schema:

- manifest.json
- config.yaml
- metrics.csv
- summary.json

The suite manifest and `config.yaml` record both the explicitly supplied kwargs
and `resolved_call_kwargs`, obtained by binding the target function signature
and applying defaults. The resolved call is the authoritative call-level
contract used for resume/provenance checks. A referenced YAML file is retained
as supporting configuration context rather than assumed to be identical to
runtime defaults.

Raw stdout/stderr are retained only for debugging. Failed jobs receive an
explicit failed manifest; downstream paper aggregation refuses such a results
root rather than silently omitting failures.

Large benchmark suites are intentionally not executed by routine CI. The
registry statically validates callable signatures/config references, while a
separate bounded smoke pipeline executes end to end to validate launcher,
canonical artifact conversion, and paper aggregation.

The `baseline_core` suite is explicitly backbone-matched. SAC-Adam,
SAC-EWC, SAC-Online-EWC, SAC-SI, and SAC-MAS all use the same
`recurring_lqr_matched_v1` stream, 600 environment steps, phase length 120,
horizon 32, network width, optimizer settings, replay budget, and five seeds.
The regularized methods differ only by the continual regularizer and its
fixed-update consolidation state.

PPO-UCL is paired with a phase-matched PPO-Adam oracle control under the same
`oracle_boundary` protocol, stream, phase length, horizon, PPO
hyperparameters, model widths, phases, and five seeds. The Adam control receives
the same evaluator-owned phase segmentation but performs no Bayesian posterior
snapshot or UCL penalty. This lets Phase-15 form matched-seed differences within
each backbone family instead of comparing heterogeneous SAC and PPO conditions.

The `compute_analysis` suite contains two separate matched factors. The
bayesianization-cost family compares SAC-Adam with SAC-BGD critic-only,
actor-only, and actor-and-critic at K=2. The Monte-Carlo-cost family holds
critic-only Bayesianization fixed and compares K=1/2/4/8. Every arm runs 600
environment steps on the same synthetic LQR protocol with seeds 0, 1, and 2.
Launcher wall-clock duration is the primary resource metric and post-training
return/improvement are matched secondary outcomes. Phase 15 never pairs methods
across these two factors.


Both `cw10_core` and `cw20_final` are comparison-matched rather than
method-only. Their feed-forward strict task-agnostic families contain SAC-Adam,
SAC-BGD, SAC-EWC, SAC-Online-EWC, SAC-SI, and SAC-MAS with the same benchmark
stream, one-million steps per task, five evaluation episodes per stage, and
seeds 0-4. EWC/SI/MAS consolidation is driven by a fixed optimizer-update
interval and receives no ground-truth task ID or boundary. The recurrent
3RL-style families contain recurrent SAC-Adam, recurrent SAC-BGD, and recurrent
SAC-adaptive-BGD with the same benchmark stream, one-million steps per task,
ten evaluation episodes per stage, recurrent architecture/config, and seeds
0-4. CW20 task-agnostic jobs additionally record the same occurrence-aware
revisit/reacquisition metrics; Bayesian jobs expose posterior uncertainty and
adaptive jobs additionally expose retention.


The `uncertainty_analysis` retention-policy block is also explicitly matched.
A no-tempering BGD control, fixed retention 0.97, TD-residual surprise,
twin-critic ensemble disagreement, and predictive transition/reward NLL all use
the same recurring LQR stream, 900 environment steps, phase length 300,
critic-only BGD settings, and seeds 0-4. The adaptive source arms additionally
share the same EMA normalization and retention mapping. Thresholded event
precision/recall/F1 is complemented by evaluator-only surprise AUROC, while
predictive NLL additionally records its online model loss.
