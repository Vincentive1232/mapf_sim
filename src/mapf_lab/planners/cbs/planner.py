from __future__ import annotations

"""Conflict-Based Search (CBS) high-level planner implementation."""

import copy
import heapq

from rich import print

from mapf_lab.core.constraints import Constraint, split_conflict_to_constraints
from mapf_lab.core.conflicts import detect_first_conflict
from mapf_lab.core.paths import DiscretePath
from mapf_lab.core.solution import MultiAgentSolution
from mapf_lab.planners.cbs.ct_node import CTNode
from mapf_lab.planners.low_level.astar import GridAStarPlanner


class CBSPlanner:
    """High-level CBS planner coordinating single-agent replanning.

    The planner maintains a constraint tree (CT). Each node stores a complete
    set of agent paths and associated constraints. On conflict, the node is
    split into children by adding one constraint per conflicting agent.
    """

    def __init__(
        self,
        low_level: GridAStarPlanner,
        max_ct_nodes: int = 10000,
        debug: bool = True,
    ) -> None:
        """Initialize the CBS planner.

        Args:
            low_level: Low-level single-agent planner used for replanning.
            max_ct_nodes: Maximum number of CT nodes to expand before stopping.
            debug: Whether to print debug logs during search.
        """
        self.low_level = low_level
        self.max_ct_nodes = max_ct_nodes
        self.debug = debug
        self._node_id = 0

    def _next_id(self) -> int:
        """Return a new unique CT node id."""
        self._node_id += 1
        return self._node_id

    def _compute_cost(self, paths: dict[int, DiscretePath], objective: str = "soc") -> float:
        """Compute objective value for a set of agent paths.

        Args:
            paths: Mapping from agent id to planned path.
            objective: Cost objective, either ``"soc"`` or ``"makespan"``.

        Returns:
            Objective value for the given path set.

        Raises:
            ValueError: If the objective is unsupported.
        """
        solution = MultiAgentSolution(paths=paths)
        if objective == "soc":
            return solution.cost_soc()
        if objective == "makespan":
            return solution.cost_makespan()
        raise ValueError(f"Unsupported objective: {objective}")

    def _constraint_signature(self, constraints: list[Constraint]) -> frozenset[Constraint]:
        """Build a hashable signature for deduplicating constraint sets."""
        return frozenset(constraints)

    def _build_root(self, world, robots, objective: str) -> CTNode | None:
        """Construct the root CT node by planning each agent independently.

        Returns ``None`` if any agent cannot be planned in the unconstrained
        root problem.
        """
        paths: dict[int, DiscretePath] = {}

        for robot in robots:
            result = self.low_level.solve(world, robot, constraints=[])
            if not result.success:
                if self.debug:
                    print(f"[red]Root planning failed for robot {robot.id}:[/red] {result.message}")
                return None
            paths[robot.id] = DiscretePath(states=result.path)

        conflict = detect_first_conflict(paths)
        cost = self._compute_cost(paths, objective=objective)
        return CTNode(
            priority=(cost, 0, self._next_id()),
            cost=cost,
            constraints=[],
            paths=paths,
            conflict=conflict,
            id=self._node_id,
            depth=0,
        )

    def _replan_one_agent(self, world, robots_by_id, constraints: list[Constraint], old_paths, agent_id: int):
        """Replan a single agent under the provided constraint set.

        Returns:
            Updated path dictionary on success, otherwise ``None``.
        """
        new_paths = copy.deepcopy(old_paths)
        robot = robots_by_id[agent_id]

        result = self.low_level.solve(world, robot, constraints=constraints)
        if not result.success:
            if self.debug:
                print(
                    f"[yellow]Replan failed[/yellow] for agent {agent_id} under constraints {constraints}: {result.message}"
                )
            return None

        new_paths[agent_id] = DiscretePath(states=result.path)
        return new_paths

    def solve(self, world, robots, objective: str = "soc") -> MultiAgentSolution | None:
        """Run CBS and return a conflict-free multi-agent solution.

        Args:
            world: Runtime world object used by low-level planning.
            robots: List of robot instances to plan.
            objective: High-level objective, ``"soc"`` or ``"makespan"``.

        Returns:
            A conflict-free solution if found, otherwise ``None``.
        """
        robots_by_id = {r.id: r for r in robots}

        root = self._build_root(world, robots, objective=objective)
        if root is None:
            return None

        open_list: list[CTNode] = []
        heapq.heappush(open_list, root)

        visited: set[frozenset[Constraint]] = set()
        visited.add(self._constraint_signature(root.constraints))

        expanded_ct = 0

        while open_list and expanded_ct < self.max_ct_nodes:
            node = heapq.heappop(open_list)
            expanded_ct += 1

            if self.debug:
                print(
                    f"\n[cyan]Expand CT node[/cyan] id={node.id}, depth={node.depth}, "
                    f"cost={node.cost}, constraints={len(node.constraints)}, conflict={node.conflict}"
                )

            if node.conflict is None:
                if self.debug:
                    print(f"[green]CBS solved[/green] after expanding {expanded_ct} CT nodes")
                return MultiAgentSolution(paths=node.paths)

            c1, c2 = split_conflict_to_constraints(node.conflict)

            for new_constraint in (c1, c2):
                child_constraints: list[Constraint] = list(node.constraints) + [new_constraint]
                sig = self._constraint_signature(child_constraints)

                if sig in visited:
                    if self.debug:
                        print(f"[dim]Skip duplicate constraint set:[/dim] {child_constraints}")
                    continue

                replanned_paths = self._replan_one_agent(
                    world=world,
                    robots_by_id=robots_by_id,
                    constraints=child_constraints,
                    old_paths=node.paths,
                    agent_id=new_constraint.agent,
                )
                if replanned_paths is None:
                    continue

                child_conflict = detect_first_conflict(replanned_paths)
                child_cost = self._compute_cost(replanned_paths, objective=objective)
                child_id = self._next_id()

                child = CTNode(
                    priority=(child_cost, node.depth + 1, child_id),
                    cost=child_cost,
                    constraints=child_constraints,
                    paths=replanned_paths,
                    conflict=child_conflict,
                    id=child_id,
                    depth=node.depth + 1,
                )

                visited.add(sig)
                heapq.heappush(open_list, child)

                if self.debug:
                    print(
                        f"[blue] Push child[/blue] id={child.id}, depth={child.depth}, "
                        f"cost={child.cost}, constraint={new_constraint}, conflict={child.conflict}"
                    )

        if self.debug:
            print(f"[red]CBS terminated without solution[/red], expanded {expanded_ct} CT nodes")
        return None