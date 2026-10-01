# Benchmark configs

Benchmark-specific configs are added only when the corresponding adapter and
runner are executable and leakage-tested.

- `continual_world_ta_cw10.yaml`: strict task-agnostic CW10 with evaluator-only
  stage knowledge.
- `continual_world_ta_cw20.yaml`: strict task-agnostic CW20, preserving the
  published CW10 + CW10 occurrence order.

Canonical task-aware configs remain intentionally absent until the canonical
training runner is wired end to end.
