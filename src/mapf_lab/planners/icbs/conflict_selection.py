from __future__ import annotations

from mapf_lab.core.conflicts import Conflict
from mapf_lab.planners.icbs.cardinal import ClassifiedConflict

def conflict_time(conflict: Conflict) -> int:
    """Return the discrete time step at which the conflict occurs."""
    return conflict.time


def _priority(cardinality: str) -> int:
    """Return an integer priority for the given conflict cardinality. Lower values indicate higher priority.
    
    Args:
        cardinality: The cardinality of the conflict ("cardinal", "semi-cardinal", "non-cardinal").

    Returns:
        An integer priority value.
    """
    if cardinality == "cardinal":
        return 0
    if cardinality == "semi-cardinal":
        return 1
    return 2


def select_conflict(conflicts: list[Conflict], mode: str = "earliest") -> Conflict | None:
    """Select one conflict from the list according to the specified mode.

    Args:
        conflicts: List of detected conflicts in a multi-agent solution.
        mode: Conflict selection strategy. Supported values are:
            - "earliest": Select the conflict with the smallest time step.
            - "random": Select a random conflict from the list.

    Returns:
        The selected conflict, or ``None`` if the input list is empty.
    """
    if not conflicts:
        return None

    if mode == "first":
        return conflicts[0]

    if mode == "earliest":
        return min(conflicts, key=conflict_time)

    raise ValueError(f"Unsupported conflict selection mode: {mode}")


def select_classified_conflict(
    classified_conflicts: list[ClassifiedConflict],
    mode: str = "cardinality_then_earliest",
) -> Conflict | None:
    """Select one conflict from the list of classified conflicts according to the specified mode.

    Args:
        classified_conflicts: List of conflicts annotated with cardinality information.
        mode: Conflict selection strategy. Supported values are:
            - "cardinality_then_earliest": Select the conflict with the highest cardinality (cardinal > semi-cardinal > non-cardinal), breaking ties by earliest time step.

    Returns:
        The selected conflict, or ``None`` if the input list is empty.
    """
    if not classified_conflicts:
        return None

    if mode == "cardinal_then_earliest":
        best = min(
            classified_conflicts,
            key=lambda cc: (_priority(cc.cardinality), cc.conflict.time),
        ) # first sort by cardinality priority, then by earliest time
        return best.conflict

    raise ValueError(f"Unsupported classified conflict selection mode: {mode}")