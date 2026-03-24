from __future__ import annotations

"""Shared data types used by low-level path planners(A*)."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(order=True)
class PriorityState:
    """Heap entry pairing a sortable priority with a planner node.

    Attributes:
        priority: Priority value used by the open set ordering.
        node: Planner node payload associated with the priority.
    """

    priority: tuple[float, int, float]
    node: Any = field(compare=False)


@dataclass
class GridNode:
    """Search node for grid-based path planning.

    Attributes:
        pos: Grid cell coordinate as ``(x, y)``.
        g: Cost-to-come from the start node.
        h: Heuristic estimate to the goal.
        parent: Parent node coordinate used for path reconstruction.
    """

    pos: tuple[int, int]
    g: float
    h: float
    parent: tuple[int, int] | None = None

    @property
    def f(self) -> float:
        """Return the total estimated cost ``f = g + h``."""
        return self.g + self.h


@dataclass
class PathResult:
    """Result returned by a low-level planner.

    Attributes:
        success: Whether a valid path was found.
        path: Planned state sequence.
        cost: Total path cost.
        expanded: Number of expanded search nodes.
        message: Optional diagnostic message.
    """

    success: bool
    path: list[np.ndarray]
    cost: float
    expanded: int
    message: str = ""