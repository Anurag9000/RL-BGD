# Benchmark configs

Benchmark-specific configs are added only when the corresponding adapter and
runner are executable and leakage-tested.

- `continual_world_ta_cw10.yaml`: strict task-agnostic CW10 with evaluator-only
  stage knowledge.
- `continual_world_ta_cw20.yaml`: strict task-agnostic CW20, preserving the
  published CW10 + CW10 occurrence order.
- `continual_world_canonical_cw10.yaml`: published task-aware CW10 baseline
  with occurrence one-hot routing, multi-head SAC, replay reset, and optimizer reset.
- `continual_world_canonical_cw20.yaml`: the same canonical protocol on the
  repeated CW10 + CW10 sequence.
