from __future__ import annotations

"""Constraint types used by conflict-based multi-agent planning."""

from dataclasses import dataclass
from typing import Literal

from mapf_lab.core.conflicts import EdgeConflict, VertexConflict


ConstraintKind = Literal["vertex", "edge"]


@dataclass(frozen=True)
class VertexConstraint:
    """Constraint forbidding an agent from occupying a cell at a time step.

    Attributes:
        agent: Constrained agent identifier.
        time: Discrete time step of the constraint.
        cell: Forbidden grid cell as ``(x, y)``.
        kind: Constraint discriminator, always ``"vertex"``.
    """

    agent: int
    time: int
    cell: tuple[int, int]
    kind: ConstraintKind = "vertex"


@dataclass(frozen=True)
class EdgeConstraint:
    """Constraint forbidding an agent from traversing an edge at a time step.

    Attributes:
        agent: Constrained agent identifier.
        time: Discrete time step of the constraint.
        edge: Forbidden directed edge as ``((x1, y1), (x2, y2))``.
        kind: Constraint discriminator, always ``"edge"``.
    """

    agent: int
    time: int
    edge: tuple[tuple[int, int], tuple[int, int]]
    kind: ConstraintKind = "edge"


Constraint = VertexConstraint | EdgeConstraint


def split_conflict_to_constraints(conflict: VertexConflict | EdgeConflict) -> tuple[Constraint, Constraint]:
    """Convert a detected conflict into one constraint per involved agent.

    Args:
        conflict: Vertex or edge conflict detected between two agents.

    Returns:
        Pair of constraints, one for each conflicting agent.

    Raises:
        ValueError: If the conflict type is unsupported.
    """
    if isinstance(conflict, VertexConflict):
        c1 = VertexConstraint(
            agent=conflict.agent_i,
            time=conflict.time,
            cell=conflict.cell,
        )
        c2 = VertexConstraint(
            agent=conflict.agent_j,
            time=conflict.time,
            cell=conflict.cell,
        )
        return c1, c2

    elif isinstance(conflict, EdgeConflict):
        c1 = EdgeConstraint(
            agent=conflict.agent_i,
            time=conflict.time,
            edge=conflict.edge_i,
        )
        c2 = EdgeConstraint(
            agent=conflict.agent_j,
            time=conflict.time,
            edge=conflict.edge_j,
        )
        return c1, c2
    else:
        raise ValueError(f"Unsupported conflict type: {type(conflict)}")