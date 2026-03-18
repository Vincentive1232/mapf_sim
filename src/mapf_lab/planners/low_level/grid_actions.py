from __future__ import annotations

"""Grid move generation for low-level path planning."""


def get_grid_moves(connectivity: int) -> list[tuple[int, int, float]]:
    """Return the available moves for a given grid neighborhood connectivity.

    Args:
        connectivity: Neighborhood type, either 4 or 8.

    Returns:
        List of moves as ``(dx, dy, cost)`` tuples, where ``(dx, dy)`` is the
        cell offset and ``cost`` is the transition cost.

    Raises:
        ValueError: If the connectivity is unsupported.
    """
    if connectivity == 4:
        return [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
        ]

    if connectivity == 8:
        diag = 2.0**0.5
        return [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, diag),
            (1, -1, diag),
            (-1, 1, diag),
            (-1, -1, diag),
        ]

    raise ValueError(f"Unsupported connectivity: {connectivity}")