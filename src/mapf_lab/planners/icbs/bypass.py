from __future__ import annotations

from dataclasses import dataclass

from mapf_lab.core.conflicts import Conflict, EdgeConflict, VertexConflict, detect_all_conflicts
from mapf_lab.core.paths import DiscretePath
from mapf_lab.core.constraints import Constraint


def _conflict_to_hashable(conflict: Conflict) -> tuple:
    """Convert a Conflict object to a hashable tuple representation."""
    if isinstance(conflict, VertexConflict):
        return ("vertex", conflict.agent_i, conflict.agent_j, conflict.time, conflict.cell)
    elif isinstance(conflict, EdgeConflict):
        return ("edge", conflict.agent_i, conflict.agent_j, conflict.time, conflict.edge_i, conflict.edge_j)
    else:
        raise ValueError(f"Unknown conflict type: {type(conflict)}")


@dataclass
class BypassCandidate:
    """Candidate node produced by adding one branch constraint.

    Attributes:
        constraints: New node constraints after applying one conflict split branch.
        paths: Replanned multi-agent paths.
        conflict: Next selected conflict for the candidate node.
        conflicts: All detected conflicts in ``paths``.
        cost: Candidate node objective value.
    """

    constraints: list[Constraint]
    paths: dict[int, DiscretePath]
    conflict: Conflict | None
    conflicts: list[Conflict]
    cost: float


def count_conflicts(paths: dict[int, DiscretePath]) -> int:
    """Return the total number of conflicts in the given path set."""
    return len(detect_all_conflicts(paths))


def choose_bypass_candidate(
    candidates: list[BypassCandidate],
    *,
    parent_cost: float,
    parent_conflicts: list[Conflict],
) -> BypassCandidate | None:
    """Pick a bypass candidate that improves conflicts without increasing cost.

    ICBS bypass requires a branch child with the same ``f``-value and whose
    conflict set is a strict subset of the parent's conflict set.
    """
    parent_conflict_set = {_conflict_to_hashable(c) for c in parent_conflicts}

    feasible = [
        c
        for c in candidates
        if c.cost <= parent_cost and {_conflict_to_hashable(cf) for cf in c.conflicts} < parent_conflict_set
    ]
    if not feasible:
        return None

    # Prefer lowest conflict count, then lowest cost.
    return min(feasible, key=lambda c: (len(c.conflicts), c.cost))
