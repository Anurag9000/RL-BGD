# Tracked-source census and static evidence

Run:

```bash
python scripts/audit_source_inventory.py --output artifacts/audit/source_inventory.json
```

The source census reads the **actual Git index** (including vendor, tests,
scripts, documentation, workflows, configurations and root controller).
It reads and hashes every tracked file, reports byte and line counts, parses
Python declarations/imports/call expressions without executing them, validates
JSON/YAML/TOML syntax when present, and reports candidate broken relative
Markdown links. Git SHA and content hashes permit source-grounded comparisons.

The report is deterministic for the same tracked revision and file content.
It reports a static evidence level on every Python symbol; declarations,
imports and call tokens do **not** demonstrate runtime reachability,
checkpoint correctness, scientific equivalence, simulator fidelity, passing
tests, completed experiments, or evidence supporting a research hypothesis.
Candidate broken Markdown links are heuristic and must be reviewed rather
than being automatically treated as confirmed source defects.

Use `--fail-on-static-errors` where a clean syntax/structured-data census
is required. The pinned vendor source is included for provenance and byte
coverage but must not be modified by native audit repair logic. The
independent `docs/foundry_integration/verify_source.py` verifier continues
to own its pinned vendor integrity contract.

The original `python scripts/audit_repository.py` remains the separate
fail-closed native repository acceptance audit. Its source coverage is
not a substitute for mathematical, functional, integration, recovery, or
GPU hardware tests.
