# Project Foundry: coordinated pressure-aware runner

This repository currently contains the Linux-focused coordinated experiment runner transferred from `simple-adp`, plus its runner contracts and operational documentation. It is a reusable runner component; it is not the complete ProjectFoundry framework described by the broader ProjectFoundry design prompts.

The implementation provides deterministic suite planning, dataset and model adapter registries, one-worker-per-route execution, host/GPU resource sampling, pressure-aware admission and recovery, checkpoint and telemetry reconciliation, resume bundles, watchdog supervision, and reporting. The exact mechanisms and compatibility behaviors are recorded in [`docs/COORDINATED_PRESSURE_AWARE_RUNNER_LEDGER.md`](docs/COORDINATED_PRESSURE_AWARE_RUNNER_LEDGER.md), with the implementation map in [`docs/COORDINATED_PRESSURE_AWARE_SOURCE_FUNCTION_MAP.md`](docs/COORDINATED_PRESSURE_AWARE_SOURCE_FUNCTION_MAP.md).

## Linux usage

Install the package and development tests:

```bash
python3 -m pip install -e '.[dev,benchmark]'
```

Show CLI options without launching a suite:

```bash
simple-adp-suite --help
```

Or use the shell launcher, which sets Linux thread defaults and starts the coordinator:

```bash
scripts/simple-adp-suite.sh --help
```

The standalone POSIX pressure wrapper is available as `simple-adp-pressure-wrapper`.

## Validation

Run the runner-focused test suite with:

```bash
python3 -m pytest -q
```

Tests distinguish synthetic resource samples from real Linux subprocess and signal behavior. Source-differential tests and the external source archive are excluded from this repository. Tests do not certify physical GPU/host pressure response or constitute a full OpenML campaign. Windows scripts remain outside the target platform.

## Package structure

- `src/simple_adp/runner/`: coordinated runner implementation.
- Other `src/simple_adp/` modules: estimator, task, and dataset adapter dependencies used by built-in model/provider integrations.
- `tests/test_runner_*.py`: self-contained runner contracts and integration checks.
