from __future__ import annotations

"""Conflict data types and detection utilities for MAPF paths."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from mapf_lab.core.paths import DiscretePath


ConflictKind = Literal["vertex", "edge"]


@dataclass
class VertexConflict:
    """Conflict where two robots occupy the same cell at the same time.

    Attributes:
        agent_i: First conflicting agent id.
        agent_j: Second conflicting agent id.
        time: Conflict time step.
        cell: Shared occupied cell as ``(x, y)``.
        kind: Conflict discriminator, always ``"vertex"``.
    """

    agent_i: int
    agent_j: int
    time: int
    cell: tuple[int, int]
    kind: ConflictKind = "vertex"


@dataclass
class EdgeConflict:
    """Conflict where two robots swap cells at the same time.

    Attributes:
        agent_i: First conflicting agent id.
        agent_j: Second conflicting agent id.
        time: Conflict time step (transition from ``t`` to ``t+1``).
        edge_i: Directed edge traversed by ``agent_i``.
        edge_j: Directed edge traversed by ``agent_j``.
        kind: Conflict discriminator, always ``"edge"``.
    """

    agent_i: int
    agent_j: int
    time: int
    edge_i: tuple[tuple[int, int], tuple[int, int]]
    edge_j: tuple[tuple[int, int], tuple[int, int]]
    kind: ConflictKind = "edge"


Conflict = VertexConflict | EdgeConflict


def state_to_cell(state: np.ndarray) -> tuple[int, int]:
    """Convert a robot state vector to a grid cell coordinate."""
    return (int(state[0]), int(state[1]))

def detect_first_conflict(paths: dict[int, "DiscretePath"]) -> Conflict | None:
    """Detect the first conflict between any pair of robot paths.

    Args:
        paths: Dictionary mapping robot IDs to their planned discrete paths.

    Returns:
        The first detected Conflict object, or None if no conflicts are found.
    """
    from mapf_lab.core.paths import DiscretePath

    if not paths:
        return None

    horizon = max(len(path) for path in paths.values())
    agent_ids = list(paths.keys())

    # vertex conflicts
    # loop over time steps and pairs of agents to check for vertex conflicts (same cell at same time)
    # always return the first conflict found, which is the earliest in time and lowest agent IDs due to the ordering of loops
    for t in range(horizon):
        for i_idx in range(len(agent_ids)):
            for j_idx in range(i_idx + 1, len(agent_ids)):
                ai = agent_ids[i_idx]
                aj = agent_ids[j_idx]

                pi = paths[ai][t]
                pj = paths[aj][t]

                ci = state_to_cell(pi)
                cj = state_to_cell(pj)

                if ci == cj:
                    return VertexConflict(
                        agent_i=ai,
                        agent_j=aj,
                        time=t,
                        cell=ci,
                    )

    # edge conflicts
    # loop over time steps and pairs of agents to check for edge conflicts (swapping cells at same time)
    for t in range(horizon - 1):
        for i_idx in range(len(agent_ids)):
            for j_idx in range(i_idx + 1, len(agent_ids)):
                ai = agent_ids[i_idx]
                aj = agent_ids[j_idx]

                ai_t = state_to_cell(paths[ai][t])
                ai_t1 = state_to_cell(paths[ai][t + 1])

                aj_t = state_to_cell(paths[aj][t])
                aj_t1 = state_to_cell(paths[aj][t + 1])

                if ai_t == aj_t1 and ai_t1 == aj_t:
                    return EdgeConflict(
                        agent_i=ai,
                        agent_j=aj,
                        time=t,
                        edge_i=(ai_t, ai_t1),
                        edge_j=(aj_t, aj_t1),
                    )

    return None


def detect_all_conflicts(paths: dict[int, "DiscretePath"]) -> list[Conflict]:
    """Detect all conflicts between any pair of robot paths.

    Args:
        paths: Dictionary mapping robot IDs to their planned discrete paths.

    Returns:
        A list of all detected Conflict objects.
    """
    conflicts: list[Conflict] = []
    if not paths:
        return conflicts

    horizon = max(len(path) for path in paths.values())
    agent_ids = list(paths.keys())

    # vertex conflicts
    # loop over time steps and pairs of agents to check for vertex conflicts (same cell at same time)
    # always return the first conflict found, which is the earliest in time and lowest agent IDs due to the ordering of loops
    for t in range(horizon):
        for i_idx in range(len(agent_ids)):
            for j_idx in range(i_idx + 1, len(agent_ids)):
                ai = agent_ids[i_idx]
                aj = agent_ids[j_idx]

                pi = paths[ai][t]
                pj = paths[aj][t]

                ci = state_to_cell(pi)
                cj = state_to_cell(pj)

                if ci == cj:
                    conflicts.append(
                        VertexConflict(
                            agent_i=ai,
                            agent_j=aj,
                            time=t,
                            cell=ci,
                        )
                    )

    # edge conflicts
    # loop over time steps and pairs of agents to check for edge conflicts (swapping cells at same time)
    for t in range(horizon - 1):
        for i_idx in range(len(agent_ids)):
            for j_idx in range(i_idx + 1, len(agent_ids)):
                ai = agent_ids[i_idx]
                aj = agent_ids[j_idx]

                ai_t = state_to_cell(paths[ai][t])
                ai_t1 = state_to_cell(paths[ai][t + 1])

                aj_t = state_to_cell(paths[aj][t])
                aj_t1 = state_to_cell(paths[aj][t + 1])

                if ai_t == aj_t1 and ai_t1 == aj_t:
                    conflicts.append(
                        EdgeConflict(
                            agent_i=ai,
                            agent_j=aj,
                            time=t,
                            edge_i=(ai_t, ai_t1),
                            edge_j=(aj_t, aj_t1),
                        )
                    )

    return conflicts