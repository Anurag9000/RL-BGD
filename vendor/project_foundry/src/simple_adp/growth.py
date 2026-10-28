"""Width/depth architecture mutations with overlap-copy warm starts."""

from __future__ import annotations

from .models import TabularMLP, rebuild_mlp


def total_neurons(model: TabularMLP) -> int:
    """Guide-compatible capacity proxy: hidden units plus output units."""
    return sum(model.hidden_widths) + model.output_dim


def scalar_width(model: TabularMLP, strategy: str) -> int:
    """Controller width summary: max for synchronized growth, min for RR rounds."""
    if strategy in {"round_robin", "rr", "per_layer"}:
        return min(model.hidden_widths)
    if strategy in {"all_layers", "uniform", "simultaneous"}:
        return max(model.hidden_widths)
    raise ValueError(f"unknown width growth strategy {strategy!r}")


def expand_width_all_layers(model: TabularMLP, ex_k: int, max_width: int) -> TabularMLP | None:
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    increment = max(1, ex_k)
    widths = [min(max_width, width + increment) for width in model.hidden_widths]
    if widths == model.hidden_widths or any(width > max_width for width in widths):
        return None
    return rebuild_mlp(model, widths, cursor=int(getattr(model, "_width_cursor", 0)))


def expand_width_round_robin(
    model: TabularMLP,
    ex_k: int,
    max_width: int,
    *,
    increment_override: int | None = None,
) -> TabularMLP | None:
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    widths = list(model.hidden_widths)
    cursor = int(getattr(model, "_width_cursor", 0)) % len(widths)
    increment = max(1, increment_override if increment_override is not None else ex_k)
    for offset in range(len(widths)):
        index = (cursor + offset) % len(widths)
        if widths[index] < max_width:
            widths[index] = min(max_width, widths[index] + increment)
            new = rebuild_mlp(model, widths, cursor=(index + 1) % len(widths))
            return new
    return None


def expand_width_den_round_robin(
    model: TabularMLP,
    ex_k: int,
    max_width: int,
) -> TabularMLP | None:
    """Dedicated DEN RR rule: choose the first layer below min(widths)+k."""
    if ex_k <= 0 or max_width <= 0:
        raise ValueError("ex_k and max_width must be positive")
    widths = list(model.hidden_widths)
    target_width = min(widths) + ex_k
    cursor = int(getattr(model, "_width_cursor", 0)) % len(widths)
    for offset in range(len(widths)):
        index = (cursor + offset) % len(widths)
        if widths[index] < target_width and widths[index] < max_width:
            # Dedicated DEN's proposal is exactly +k. Unlike generic
            # adp_search.expand_width(), it must not silently turn a larger
            # configured proposal into a smaller increment at the width cap.
            proposed_width = widths[index] + ex_k
            if proposed_width > max_width:
                return None
            widths[index] = proposed_width
            return rebuild_mlp(model, widths, cursor=(index + 1) % len(widths))
    return None


def expand_width(
    model: TabularMLP,
    ex_k: int,
    max_width: int,
    *,
    strategy: str = "round_robin",
) -> TabularMLP | None:
    if strategy in {"round_robin", "rr", "per_layer"}:
        return expand_width_round_robin(model, ex_k, max_width)
    if strategy in {"all_layers", "uniform", "simultaneous"}:
        return expand_width_all_layers(model, ex_k, max_width)
    raise ValueError(f"unknown width growth strategy {strategy!r}")


def expand_depth(
    model: TabularMLP,
    max_depth: int,
    min_new_layer_width: int = 1,
) -> TabularMLP | None:
    if model.depth >= max_depth:
        return None
    if len(set(model.hidden_widths)) > 1:
        return None
    return rebuild_mlp(
        model, [*model.hidden_widths, max(1, min_new_layer_width)],
        cursor=int(getattr(model, "_width_cursor", 0)),
    )


def expand_depth_match_width(model: TabularMLP, max_depth: int) -> TabularMLP | None:
    if model.depth >= max_depth or not model.hidden_widths:
        return None
    return rebuild_mlp(model, [*model.hidden_widths, model.hidden_widths[-1]], cursor=int(getattr(model, "_width_cursor", 0)))


def can_grow_width_all_layers(model: TabularMLP, ex_k: int, max_width: int, max_neurons: int) -> bool:
    return (
        all(width + ex_k <= max_width for width in model.hidden_widths)
        and total_neurons(model) + model.depth * ex_k <= max_neurons
    )


def can_grow_width_round_robin(model: TabularMLP, ex_k: int, max_width: int, max_neurons: int) -> bool:
    target_width = min(model.hidden_widths) + ex_k
    cursor = int(getattr(model, "_width_cursor", 0)) % len(model.hidden_widths)
    selected = next(
        (
            model.hidden_widths[(cursor + offset) % len(model.hidden_widths)]
            for offset in range(len(model.hidden_widths))
            if model.hidden_widths[(cursor + offset) % len(model.hidden_widths)] < target_width
            and model.hidden_widths[(cursor + offset) % len(model.hidden_widths)] < max_width
        ),
        None,
    )
    return (
        selected is not None
        and selected + ex_k <= max_width
        and total_neurons(model) + ex_k <= max_neurons
    )


def can_add_depth(model: TabularMLP, max_depth: int, max_neurons: int, seed_width: int) -> bool:
    return (
        model.depth > 0
        and len(set(model.hidden_widths)) == 1
        and model.depth + 1 <= max_depth
        and total_neurons(model) + seed_width <= max_neurons
    )
