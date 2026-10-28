"""Canonical 14-configuration taxonomy from ADP_algorithms.docx."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from .config import ModelConfig

_CONFIGS = (
    ModelConfig("LGND-W-Greedy", "lgnd", "width", "all_layers", description="Width-only, stop at first non-improvement."),
    ModelConfig("ADP-W-RR", "width_den", "width", "round_robin", description="Patience-based round-robin width densification."),
    ModelConfig("ADP-W-All", "width_den", "width", "all_layers", description="Patience-based all-layer width densification."),
    ModelConfig("ADP-WD-RR-D1", "phase_search", "width_then_depth", "round_robin", "one", generic=True, description="Round-robin width phase, then depth seeded at width one."),
    ModelConfig("ADP-WD-All-Dcur", "phase_search", "width_then_depth", "all_layers", "current", generic=True, description="All-layer width phase, then matched-width depth."),
    ModelConfig("ADP-WD-RR-Dcur", "phase_search", "width_then_depth", "round_robin", "current", generic=True, description="Round-robin width phase, then matched-width depth."),
    ModelConfig("ADP-WD-All-D1", "phase_search", "width_then_depth", "all_layers", "one", generic=True, description="All-layer width phase, then depth seeded at width one."),
    ModelConfig("ADP-DW-D1-RR", "phase_search", "depth_then_width", "round_robin", "one", generic=True, description="Depth seeded at width one, then round-robin width phase."),
    ModelConfig("ADP-DW-Dcur-All", "phase_search", "depth_then_width", "all_layers", "current", generic=True, description="Matched-width depth phase, then all-layer width phase."),
    ModelConfig("ADP-DW-Dcur-RR", "phase_search", "depth_then_width", "round_robin", "current", generic=True, description="Matched-width depth phase, then round-robin width phase."),
    ModelConfig("ADP-DW-D1-All", "phase_search", "depth_then_width", "all_layers", "one", generic=True, description="Depth seeded at width one, then all-layer width phase."),
    ModelConfig("ADP-D-D1", "depth_den", "depth", "all_layers", "one", progressive_depth=True, description="Progressive width-one depth insertion."),
    ModelConfig("ADP-D-Dcur", "depth_den", "depth", "all_layers", "current", description="Matched-width depth insertion."),
    ModelConfig("MLP-Fixed", "fixed", "fixed", description="Caller-specified one-shot fixed-architecture grid."),
)

MODEL_REGISTRY: Mapping[str, ModelConfig] = MappingProxyType({config.id: config for config in _CONFIGS})


def get_model_config(model_id: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        choices = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"unknown model id {model_id!r}; choose one of: {choices}") from exc
