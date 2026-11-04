# Project Foundry native execution integration

The pinned Project Foundry source at `vendor/project_foundry/` is unchanged.
The new `run_all_training.py` uses the actual vendored
`simple_adp.runner.scheduler.SuiteCoordinator`, with Foundry process/resource
admission, and a per-job subprocess bridge into the original native trainer.

`python run_all_training.py` lists the supported native scientific runs
without launching training. `python run_all_training.py execute --execute
--job ID` is the execution path. Routes without confirmed bit-exact
checkpoint restoration require explicit `--accept-restart-risk`: Foundry
can terminate and relaunch children during memory pressure. Do not enable
restart risk for destructive/non-idempotent jobs.

`python run_all_training.py status` reads Foundry's persisted controller state.
The original native training pipeline retains all scientific authority;
existing CLI and SLURM launchers are not deleted or nested as outer schedulers.

**Verification status:** Source integrity is Git-verified; native adapters
are source-implemented. Real training, pressure, GPU, exact-resume and
scientific regression are NOT certified without a relevant environment,
datasets and executed checks. A zero native subprocess exit code does not
by itself certify scientific correctness. All blocked native routes must
remain visible, not presented as runnable or trained.

Native `--job` and `--seed` selectors are strictly validated against the
source-derived catalog before planning. Unknown IDs/seeds, including unknown
entries mixed with valid selectors, and mutually incompatible valid ID/seed
filters are rejected with a non-zero CLI error; they are never silently
ignored or treated as completed work. Inventory without selectors remains
read-only and can still display blocked jobs.
