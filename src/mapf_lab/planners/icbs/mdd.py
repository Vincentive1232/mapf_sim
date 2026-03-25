from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from mapf_lab.core.constraints import EdgeConstraint, VertexConstraint
from mapf_lab.planners.low_level.grid_actions import get_grid_moves


Cell = tuple[int, int]
TimedCell = tuple[Cell, int]


@dataclass
class MDD:
    """Multi-value Decision Diagram (MDD) for representing all optimal paths of a single agent.

    An MDD is a leveled directed acyclic graph where each level corresponds to a timestep,
    and nodes at that level represent the possible cells the agent can occupy at that time
    while still being on an optimal path to the goal. Edges represent valid moves between
    cells in consecutive timesteps.

    Attributes:
        cost: The optimal path cost from start to goal.
        levels: List of sets of cells reachable at each timestep on an optimal path.
        edges: Mapping from (cell, timestep) to sets of next-step cells.
    """

    cost: int
    levels: list[set[Cell]] = field(default_factory=list)
    edges: dict[int, set[tuple[Cell, Cell]]] = field(default_factory=dict)

    def width(self, t: int) -> int:
        """Return the number of nodes at a given level (timestep) in the MDD.
        Args:
            t: Timestep for which to query the MDD width.
        Returns:
            Number of cells reachable at timestep t on an optimal path.
        """
        if t < 0 or t >= len(self.levels):
            return 0
        return len(self.levels[t])
    
    def has_vertex(self, cell: Cell, t: int) -> bool:
        """Check if a specific cell is reachable at a given timestep in the MDD.
        Args:
            cell: Grid cell to check for reachability.
            t: Timestep at which to check for the cell.
        Returns:
            True if the cell is in the MDD at timestep t, False otherwise.
        """
        return 0 <= t < len(self.levels) and cell in self.levels[t]
    
    def has_edge(self, u: Cell, v: Cell, t: int) -> bool:
        """Check if there is a valid move from cell u to cell v between timesteps t and t+1 in the MDD.
        Args:
            u: Starting cell of the move at timestep t.
            v: Ending cell of the move at timestep t+1.
            t: Timestep of the starting cell u.
        Returns:
            True if there is an edge from u at time t to v at time t+1 in the MDD, False otherwise. 
        """
        return (u, v) in self.edges.get(t, set())
    


def _build_constraints_tables(
        agent_id: int,
        constraints: list[VertexConstraint | EdgeConstraint] | None,
) -> tuple[dict[int, set[Cell]], dict[int, set[tuple[Cell, Cell]]]]:
    """Convert a list of vertex and edge constraints into lookup tables for quick access during MDD construction.

    Args:
        agent_id: The ID of the agent for which to build the constraint tables.
        constraints: List of VertexConstraint and EdgeConstraint objects relevant to the agent.
    Returns:
        A tuple containing two dictionaries:
        - Vertex constraints table: Maps timestep to a set of forbidden cells.
        - Edge constraints table: Maps timestep to a set of forbidden edges (cell pairs).
    """
    vertex_table: dict[int, set[Cell]] = {}
    edge_table: dict[int, set[tuple[Cell, Cell]]] = {}

    if not constraints:
        return vertex_table, edge_table
    
    for c in constraints:
        if c.agent != agent_id:
            continue
        if isinstance(c, VertexConstraint):
            bucket = vertex_table.setdefault(c.time, set())
            bucket.add(c.cell)
        elif isinstance(c, EdgeConstraint):
            bucket = edge_table.setdefault(c.time, set())
            bucket.add(c.edge)
        else:
            raise ValueError(f"Unknown constraint type: {c}")
        
    return vertex_table, edge_table


def _violates_vertex(cell: Cell, time: int, vertex_table: dict[int, set[Cell]]) -> bool:
    """Check if occupying a cell at a specific time violates any vertex constraints.

    Args:
        cell: The grid cell being considered.
        time: The timestep at which the cell would be occupied.
        vertex_table: A dictionary mapping timesteps to sets of forbidden cells.
    Returns:
        True if the cell is forbidden at the given time, False otherwise.
    """
    return cell in vertex_table.get(time, set())


def _violates_edge(
        edge: tuple[Cell, Cell],
        time: int,
        edge_table: dict[int, set[tuple[Cell, Cell]]],
) -> bool:
    """Check if moving along an edge at a specific time violates any edge constraints.

    Args:
        edge: The edge (pair of cells) being considered.
        time: The timestep at which the edge would be traversed.
        edge_table: A dictionary mapping timesteps to sets of forbidden edges.
    Returns:
        True if the edge is forbidden at the given time, False otherwise.
    """
    return edge in edge_table.get(time, set())


def _heuristic(cell: Cell, goal: Cell, heuristic_type: str = "manhattan") -> float:
    """Compute heuristic distance from a cell to the goal based on the specified heuristic type.

    Args:
        cell: The current grid cell.
        goal: The goal grid cell.
        heuristic_type: The type of heuristic to compute ("manhattan" or "euclidean").
    Returns:
        The computed heuristic distance from the cell to the goal.
    """
    if heuristic_type == "manhattan":
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
    elif heuristic_type == "euclidean":
        dx = cell[0] - goal[0]
        dy = cell[1] - goal[1]
        return (dx * dx + dy * dy) ** 0.5
    raise ValueError(f"Unsupported heuristic: {heuristic_type}")


def build_mdd(
        *,
        world,
        robot,
        constraints: list[VertexConstraint | EdgeConstraint] | None = None,
        optimal_cost: int,
        heuristic_type: str = "manhattan",
) -> MDD:
    """Construct an MDD for a single agent given the world, robot, constraints, and optimal path cost.

    This function performs a breadth-first search from the start state, exploring all paths that
    have a total cost equal to the optimal cost. It uses the provided constraints to prune invalid
    states and transitions. The resulting MDD captures all optimal paths from the start to the goal
    that respect the constraints.

    Assumes unit-time moves on the grid, including wait.

    Args:
        world: GridWorld instance representing the environment.
        robot: Robot instance with start and goal states.
        constraints: List of vertex and edge constraints to consider during MDD construction.
        optimal_cost: The known optimal path cost from start to goal for this agent.
        heuristic_type: The type of heuristic to use for guiding the search ("manhattan" or "euclidean").
    Returns:
        An MDD object representing all optimal paths for the agent under the given constraints.
    """
    start: Cell = tuple(int(v) for v in robot.start[:2])
    goal: Cell = tuple(int(v) for v in robot.goal[:2])

    if world.is_occupied_xy(*start):
        raise ValueError(f"Start {start} is occupied")
    if world.is_occupied_xy(*goal):
        raise ValueError(f"Goal {goal} is occupied")
    
    vertex_table, edge_table = _build_constraints_tables(robot.id, constraints)

    if _violates_vertex(start, 0, vertex_table):
        raise ValueError(f"Start {start} violates vertex constraints at time 0")
    
    max_t = int(optimal_cost)
    moves = get_grid_moves(world.connectivity) + [(0, 0, 1.0)]

    reachable: dict[int, set[Cell]] = defaultdict(set)
    forward_edges: dict[int, set[tuple[Cell, Cell]]] = defaultdict(set)

    q: deque[TimedCell] = deque()
    q.append((start, 0))
    reachable[0].add(start)

    while q:
        cell, t = q.popleft()

        if t == max_t:
            continue

        # admissible pruning: if even optimistic remaining distance exceeds cost, stop
        if t + _heuristic(cell, goal, heuristic_type) > optimal_cost:
            continue

        cx, cy = cell
        for dx, dy, step_cost in moves:
            if step_cost != 1.0:
                continue  # skip non-unit moves for MDD construction

            nx, ny = cx + dx, cy + dy
            nxt = (nx, ny)
            nt = t + 1

            if not world.in_bounds_xy(nx, ny):
                continue
            if world.is_occupied_xy(nx, ny):
                continue
            if _violates_vertex(nxt, nt, vertex_table):
                continue
            if _violates_edge((cell, nxt), t, edge_table):
                continue
            if nt + _heuristic(nxt, goal, heuristic_type) > optimal_cost:
                continue

            if nxt not in reachable[nt]:
                reachable[nt].add(nxt)
                q.append((nxt, nt))

            forward_edges[t].add((cell, nxt))

    if goal not in reachable[max_t]:
        raise ValueError(f"No optimal-cost path of cost {optimal_cost} exists for agent {robot.id}")
    
    # backward prune: keep only states/edges that can reach goal at level C
    valid: dict[int, set[Cell]] = defaultdict(set)
    valid[max_t].add(goal)

    for t in range(max_t - 1, -1, -1):
        for u, v in forward_edges.get(t, set()):
            if v in valid[t + 1]:
                valid[t].add(u)

    mdd_levels: list[set[Cell]] = []
    mdd_edges: dict[int, set[tuple[Cell, Cell]]] = defaultdict(set)

    for t in range(max_t + 1):
        mdd_levels.append(set(valid.get(t, set())))

    for t, edges in forward_edges.items():
        for u, v in edges:
            if u in valid[t] and v in valid[t + 1]:
                mdd_edges[t].add((u, v))

    return MDD(cost=max_t, levels=mdd_levels, edges=dict(mdd_edges))