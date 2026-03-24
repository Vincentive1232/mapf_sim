from __future__ import annotations

"""Grid-based A* low-level planner implementation."""

import heapq

import numpy as np

from mapf_lab.core.constraints import EdgeConstraint, VertexConstraint
from mapf_lab.planners.low_level.grid_actions import get_grid_moves
from mapf_lab.planners.low_level.heuristics import euclidean, manhattan
from mapf_lab.planners.low_level.types import GridNode, PathResult, PriorityState
from mapf_lab.planners.low_level.conflict_reservation_table import ConflictAvoidanceTable


class GridAStarPlanner:
    """A* planner operating on 2D occupancy grids.

    The planner expands grid cells using 4- or 8-connected moves from the
    world model and computes path cost with the selected heuristic.
    """

    def __init__(self, heuristic: str = "manhattan", max_time: int = 1000) -> None:
        """Initialize a grid-based A* planner with the specified heuristic.

        Args:
            heuristic: Heuristic function to use, either "manhattan" or "euclidean".
            max_time: Maximum allowed planning time steps.
        """
        if heuristic not in {"manhattan", "euclidean"}:
            raise ValueError(f"Unsupported heuristic: {heuristic}")
        self.heuristic_name = heuristic
        self.max_time = max_time

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Compute the heuristic cost between two grid cells."""
        if self.heuristic_name == "manhattan":
            return manhattan(a, b)
        elif self.heuristic_name == "euclidean":
            return euclidean(a, b)
        else:
            raise ValueError(f"Unsupported heuristic: {self.heuristic_name}")

    def _reconstruct_path(
        self, 
        parents: dict[tuple[tuple[int, int], int], tuple[tuple[int, int], int] | None],
        goal_key: tuple[tuple[int, int], int],
    ) -> list[np.ndarray]:
        """Reconstruct the path from start to goal using the parent pointers."""
        path_cells: list[tuple[int, int]] = []
        cur: tuple[tuple[int, int], int] | None = goal_key
        while cur is not None:   # when cur is None, we have reached the start node
            cell, _time = cur
            path_cells.append(cell)
            cur = parents[cur]
        path_cells.reverse()     # reverse to get path from start to goal
        return [np.asarray([x, y], dtype=float) for x, y in path_cells]

    def _build_constraints_tables(
        self,
        agent_id: int,
        constraints: list[VertexConstraint | EdgeConstraint] | None,
    ) -> tuple[dict[int, set[tuple[int, int]]], dict[int, set[tuple[tuple[int, int], tuple[int, int]]]]]:
        """Build lookup tables for vertex and edge constraints.

        Args:
            agent_id: ID of the agent for which to build the constraint tables.
            constraints: List of vertex and edge constraints to consider.
        
        Returns:
            Tuple of two dictionaries: (vertex_constraints, edge_constraints).
            - vertex_constraints maps time steps to sets of forbidden cells.
            - edge_constraints maps time steps to sets of forbidden edges.
        """
        vertex_table: dict[int, set[tuple[int, int]]] = {}
        edge_table: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {}

        if not constraints:
            return vertex_table, edge_table

        for c in constraints:
            # Only consider constraints relevant to the current agent
            if c.agent != agent_id:
                continue
            if isinstance(c, VertexConstraint):
                # Add the forbidden cell to the vertex constraint table for the specified time step
                vertex_table.setdefault(c.time, set()).add(c.cell)
            elif isinstance(c, EdgeConstraint):
                # Add the forbidden edge to the edge constraint table for the specified time step
                edge_table.setdefault(c.time, set()).add(c.edge)
            else:
                raise ValueError(f"Unsupported constraint type: {type(c)}")

        return vertex_table, edge_table

    def _violates_vertex(
        self,
        cell: tuple[int, int],
        time: int,
        vertex_table: dict[int, set[tuple[int, int]]],
    ) -> bool:
        """Check if a given cell at a specific time violates any vertex constraints."""
        # check if the cell is in the set of forbidden cells for the given time step
        return cell in vertex_table.get(time, set())

    def _violates_edge(
        self,
        edge: tuple[tuple[int, int], tuple[int, int]],
        time: int,
        edge_table: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]],
    ) -> bool:
        """Check if a given edge at a specific time violates any edge constraints."""
        # check if the edge is in the set of forbidden edges for the given time step
        return edge in edge_table.get(time, set())

    def solve(
        self, 
        world, 
        robot,
        constraints: list[VertexConstraint | EdgeConstraint] | None = None,
        cat: ConflictAvoidanceTable | None = None,
    ) -> PathResult:
        """Plan a path for the given robot in the specified world.

        Args:
            world: GridWorld instance representing the environment.
            robot: Robot instance with start and goal states.
            constraints: List of vertex and edge constraints to consider.
            cat: Conflict Avoidance Table used as a secondary tie-breaker.

        Returns:
            PathResult containing the planned path and associated information.
        """
        start = tuple(int(v) for v in robot.start[:2])
        goal = tuple(int(v) for v in robot.goal[:2])

        # Check whether start and goal cells are valid and unoccupied
        if world.is_occupied_xy(*start):
            return PathResult(
                success=False, 
                path=[], 
                cost=float("inf"), 
                expanded=0, 
                message=f"Start {start} is occupied",
            )

        if world.is_occupied_xy(*goal):
            return PathResult(
                success=False, 
                path=[], 
                cost=float("inf"), 
                expanded=0, 
                message=f"Goal {goal} is occupied",
            )

        vertex_table, edge_table = self._build_constraints_tables(robot.id, constraints)

        if self._violates_vertex(start, 0, vertex_table):
            return PathResult(
                success=False, 
                path=[], 
                cost=float("inf"), 
                expanded=0, 
                message=f"Start {start} violates vertex constraints at time 0",
            )

        # (cell, time)
        start_key = (start, 0)
        
        open_heap: list[PriorityState] = []
        g_score: dict[tuple[tuple[int, int], int], float] = {start_key: 0.0}
        cat_score: dict[tuple[tuple[int, int], int], int] = {start_key: 0}
        parents: dict[tuple[tuple[int, int], int], tuple[tuple[int, int], int] | None] = {start_key: None}
        closed: set[tuple[tuple[int, int], int]] = set()

        # compute heuristic for the start node and add it to the open set
        h0 = self._heuristic(start, goal)
        heapq.heappush(open_heap, PriorityState(priority=(h0, 0, -0.0), node=(start, 0, 0.0, 0)))

        expanded = 0
        # get the possible moves based on the world's connectivity, also include the wait action (0, 0) with a cost of 1.0
        moves = get_grid_moves(world.connectivity) + [(0, 0, 1.0)]   

        while open_heap:
            current_state = heapq.heappop(open_heap)
            current_cell, current_t, current_g, current_cat_hits = current_state.node
            current_key = (current_cell, current_t)

            if current_key in closed:
                continue
            closed.add(current_key)
            expanded += 1

            if current_cell == goal:
                path = self._reconstruct_path(parents, current_key)
                return PathResult(
                    success=True, 
                    path=path, 
                    cost=current_g, 
                    expanded=expanded, 
                    message="Path found."
                )

            if current_t >= self.max_time:
                continue

            cx, cy = current_cell
            for dx, dy, step_cost in moves:
                nx, ny = cx + dx, cy + dy
                next_cell = (nx, ny)
                next_t = current_t + 1
                next_key = (next_cell, next_t)

                if not world.in_bounds_xy(nx, ny):
                    continue
                if   world.is_occupied_xy(nx, ny):
                    continue
                if self._violates_vertex(next_cell, next_t, vertex_table):
                    continue
                if self._violates_edge((current_cell, next_cell), current_t, edge_table):
                    continue

                tentative_g = current_g + step_cost

                extra_cat = 0
                if cat is not None:
                    extra_cat += cat.vertex_penalty(next_cell, next_t)
                    extra_cat += cat.edge_penalty(current_cell, next_cell, current_t)

                tentative_cat_hits = current_cat_hits + extra_cat

                old_g = g_score.get(next_key, float("inf"))
                old_cat = cat_score.get(next_key, float("inf"))

                # If this path to neighbor is worse than any previously recorded, skip it
                if tentative_g > old_g:
                    continue
                if tentative_g == old_g and tentative_cat_hits >= old_cat:
                    continue
                
                g_score[next_key] = tentative_g
                cat_score[next_key] = tentative_cat_hits
                parents[next_key] = current_key

                h = self._heuristic(next_cell, goal)
                f = tentative_g + h
                heapq.heappush(
                    open_heap, 
                    PriorityState(
                        priority=(f, tentative_cat_hits, -tentative_g), 
                        node=(next_cell, next_t, tentative_g, tentative_cat_hits),
                        ),
                    )

        return PathResult(
            success=False, 
            path=[], 
            cost=float("inf"), 
            expanded=expanded, 
            message="No path found."
        )
