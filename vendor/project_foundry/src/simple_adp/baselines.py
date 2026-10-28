"""Standard uniform-width vanilla MLP architecture grid."""

from __future__ import annotations


def vanilla_mlp_grid() -> list[list[int]]:
    """Return all uniform-width MLPs with depth 1..5 and widths 1..512 by powers of two.

    Each candidate is a hidden-width vector. For example, depth 3 and width 16
    is represented as ``[16, 16, 16]``. The resulting grid has 50 candidates.
    """
    widths = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    return [[width] * depth for depth in range(1, 6) for width in widths]
