from __future__ import annotations

"""Constraint-tree node type used by the CBS high-level search."""

from dataclasses import dataclass, field

from mapf_lab.core.constraints import Constraint
from mapf_lab.core.conflicts import Conflict
from mapf_lab.core.paths import DiscretePath


@dataclass(order=True)
class CTNode:
    """Node in the CBS constraint tree.

    Attributes:
        priority: Heap priority tuple used for best-first node ordering.
        cost: Objective cost of the node's current multi-agent solution.
        constraints: Accumulated constraints applied at this CT node.
        paths: Current path for each agent under ``constraints``.
        conflict: First detected conflict in ``paths``, or ``None`` if conflict-free.
        id: Unique node identifier.
        depth: Node depth in the constraint tree.
    """
    priority: tuple[float, int, int]
    cost: float = field(compare=False)
    constraints: list[Constraint] = field(compare=False, default_factory=list)
    paths: dict[int, DiscretePath] = field(compare=False, default_factory=dict)
    conflict: Conflict | None = field(compare=False, default=None)
    id: int = field(compare=False, default=0)
    depth: int = field(compare=False, default=0)