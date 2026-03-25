from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mapf_lab.core.constraints import Constraint, split_conflict_to_constraints
from mapf_lab.core.conflicts import Conflict
from mapf_lab.core.paths import DiscretePath
from mapf_lab.planners.icbs.mdd import MDD


Cardinality = Literal["cardinal", "semi-cardinal", "non-cardinal"]

@dataclass
class ClassifiedConflict:
    """
    Conflict annotated with cardinality and constraint impact information.

    Attributes:
        conflict: The original conflict detected in the solution.
        cardinality: The classification of the conflict's cardinality.
        increased_i: Whether the first agent's path cost increases under its constraint.
        increased_j: Whether the second agent's path cost increases under its constraint.
    """
    conflict: Conflict
    cardinality: Cardinality
    increased_i: bool
    increased_j: bool


def _classify_vertex_conflict(conflict: Conflict, mdd_i: MDD, mdd_j: MDD) -> ClassifiedConflict:
    t = conflict.time
    cell = conflict.cell  # assumes your vertex conflict has .cell

    forced_i = mdd_i.width(t) == 1 and mdd_i.has_vertex(cell, t)
    forced_j = mdd_j.width(t) == 1 and mdd_j.has_vertex(cell, t)

    if forced_i and forced_j:
        cardinality: Cardinality = "cardinal"
    elif forced_i or forced_j:
        cardinality = "semi-cardinal"
    else:
        cardinality = "non-cardinal"

    return ClassifiedConflict(
        conflict=conflict,
        cardinality=cardinality,
        increased_i=forced_i,
        increased_j=forced_j,
    )


def _classify_edge_conflict(conflict: Conflict, mdd_i: MDD, mdd_j: MDD) -> ClassifiedConflict:
    t = conflict.time

    # assumes your edge conflict stores the two directed edges as:
    # agent_i: u_i -> v_i, agent_j: u_j -> v_j
    u_i, v_i = conflict.edge_i
    u_j, v_j = conflict.edge_j

    forced_i = (
        mdd_i.width(t) == 1
        and mdd_i.width(t + 1) == 1
        and mdd_i.has_edge(u_i, v_i, t)
    )
    forced_j = (
        mdd_j.width(t) == 1
        and mdd_j.width(t + 1) == 1
        and mdd_j.has_edge(u_j, v_j, t)
    )

    if forced_i and forced_j:
        cardinality: Cardinality = "cardinal"
    elif forced_i or forced_j:
        cardinality = "semi-cardinal"
    else:
        cardinality = "non-cardinal"

    return ClassifiedConflict(
        conflict=conflict,
        cardinality=cardinality,
        increased_i=forced_i,
        increased_j=forced_j,
    )


def classify_conflict(
    *,
    conflict: Conflict,
    mdd_i: MDD,
    mdd_j: MDD,
) -> ClassifiedConflict:
    """Classify a conflict using precomputed MDDs."""

    if conflict.kind == "vertex":
        return _classify_vertex_conflict(conflict, mdd_i, mdd_j)

    if conflict.kind == "edge":
        return _classify_edge_conflict(conflict, mdd_i, mdd_j)

    raise ValueError(f"Unsupported conflict kind: {conflict.kind}")