from __future__ import annotations

from dataclasses import dataclass, field

Cell = tuple[int, int]
Edge = tuple[Cell, Cell]

@dataclass
class ConflictAvoidanceTable:
    """Conflict Avoidance Table (CAT) for CBS low-level replanning.

    CAT stores how many other agents occupy a vertex or traverse an edge at
    each timestep. It is used as a secondary criterion in low-level A* to
    prefer paths with fewer likely future conflicts.

    This is not a hard-constraint table.

    Attributes:
        vertex_counts: Mapping from timestep to per-cell occupancy counts.
        edge_counts: Mapping from timestep to directed-edge traversal counts.
        goal_occupancy: Counts of how many agents end at each goal cell.
    """

    vertex_counts: dict[int, dict[Cell, int]] = field(default_factory=dict)
    edge_counts: dict[int, dict[Edge, int]] = field(default_factory=dict)
    goal_occupancy: dict[Cell, int] = field(default_factory=dict)

    def add_vertex(self, cell: Cell, t: int) -> None:
        """Increment occupancy count for a cell at a specific timestep.

        Args:
            cell: Grid cell to mark as occupied.
            t: Discrete timestep of the occupancy.
        """
        bucket = self.vertex_counts.setdefault(t, {})
        bucket[cell] = bucket.get(cell, 0) + 1

    def add_edge(self, u: Cell, v: Cell, t: int) -> None:
        """Increment traversal count for a directed edge at a timestep.

        Args:
            u: Start cell of the move.
            v: End cell of the move.
            t: Timestep when the move from ``u`` to ``v`` occurs.
        """
        bucket = self.edge_counts.setdefault(t, {})
        edge = (u, v)
        bucket[edge] = bucket.get(edge, 0) + 1

    def add_path(self, path: list[Cell]) -> None:
        """Insert all vertex/edge usages from a path into the CAT.

        The last cell is also recorded in ``goal_occupancy`` to represent
        persistent occupancy after arrival.

        Args:
            path: Time-indexed sequence of visited cells.
        """
        if not path:
            return

        for t, cell in enumerate(path):
            self.add_vertex(cell, t)
            if t + 1 < len(path):
                self.add_edge(cell, path[t + 1], t)

        # goal stays occupied after arrival
        goal = path[-1]
        self.goal_occupancy[goal] = self.goal_occupancy.get(goal, 0) + 1

    def vertex_penalty(self, cell: Cell, t: int) -> int:
        """Get soft conflict penalty for occupying a cell at a timestep.

        Args:
            cell: Candidate cell.
            t: Candidate timestep.

        Returns:
            Occupancy count at ``(cell, t)``.
        """
        return self.vertex_counts.get(t, {}).get(cell, 0)

    def edge_penalty(self, u: Cell, v: Cell, t: int) -> int:
        """Get soft conflict penalty for traversing a directed edge.

        Args:
            u: Start cell of the candidate move.
            v: End cell of the candidate move.
            t: Timestep when the move occurs.

        Returns:
            Traversal count for edge ``(u, v)`` at timestep ``t``.
        """
        return self.edge_counts.get(t, {}).get((u, v), 0)

    def future_goal_penalty(self, cell: Cell, t: int, path_len: int) -> int:
        """Penalty for entering a goal cell after another agent has arrived.

        This is useful because many MAPF paths remain at goal forever.

        Args:
            cell: Candidate cell.
            t: Candidate timestep.
            path_len: Original length of the candidate path.

        Returns:
            Goal occupancy count if ``t`` is after arrival time, else 0.
        """
        if t < path_len:
            return 0
        return self.goal_occupancy.get(cell, 0)

    @classmethod
    def from_other_paths(
        cls, 
        paths: dict[int, list[Cell]],
        exclude_agent: int | None = None,
    ) -> "ConflictAvoidanceTable":
        """Build a CAT from all paths except an optional agent.

        Args:
            paths: Mapping from agent id to its path.
            exclude_agent: Agent id to skip when building the table.

        Returns:
            A CAT containing occupancy/traversal statistics from selected paths.
        """
        cat = cls()
        for agent_id, path in paths.items():
            if exclude_agent is not None and agent_id == exclude_agent:
                continue
            cat.add_path(path)
        return cat