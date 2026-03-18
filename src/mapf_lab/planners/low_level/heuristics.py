from __future__ import annotations

"""Heuristic distance functions for A* and related grid planners."""

from math import sqrt


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Return the Manhattan distance between two grid cells.

    Args:
        a: First grid cell coordinate ``(x, y)``.
        b: Second grid cell coordinate ``(x, y)``.

    Returns:
        Manhattan distance, sum of absolute coordinate differences.
    """
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Return the Euclidean distance between two grid cells.

    Args:
        a: First grid cell coordinate ``(x, y)``.
        b: Second grid cell coordinate ``(x, y)``.

    Returns:
        Euclidean distance, straight-line distance between the points.
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float(sqrt(dx * dx + dy * dy))