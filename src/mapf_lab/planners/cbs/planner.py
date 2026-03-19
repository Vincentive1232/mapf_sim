from __future__ import annotations

"""Conflict-Based Search (CBS) high-level planner implementation."""

import copy
import heapq
import time

from rich import print

from mapf_lab.core.constraints import Constraint, split_conflict_to_constraints
from mapf_lab.core.conflicts import Conflict, detect_all_conflicts, detect_first_conflict
# from mapf_lab.planners.cbs.conflict_selection import select_conflict
from mapf_lab.core.paths import DiscretePath
from mapf_lab.core.solution import MultiAgentSolution
from mapf_lab.planners.cbs.ct_node import CTNode
from mapf_lab.planners.cbs.result import CBSResult
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
        timeout_sec: float | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the CBS planner.

        Args:
            low_level: Low-level single-agent planner used for replanning.
            max_ct_nodes: Maximum number of CT nodes to expand before stopping.
            timeout_sec: Maximum time allowed for planning in seconds.
            debug: Whether to print debug logs during search.
            node_id: Internal counter for assigning unique CT node ids.
        """
        self.low_level = low_level
        self.max_ct_nodes = max_ct_nodes
        self.timeout_sec = timeout_sec
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
        node_id = self._next_id()
        return CTNode(
            priority=(cost, 0, node_id),
            cost=cost,
            constraints=[],
            paths=paths,
            conflict=conflict,
            id=node_id,
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

    def _select_conflict(self, paths, world=None, robots_by_id=None, node_constraints=None):
        conflicts = detect_all_conflicts(paths)
        if not conflicts:
            return None, 0
        return min(conflicts, key=lambda c: c.time), 0     # (conflict, probe_count)

    def solve(self, world, robots, objective: str = "soc") -> CBSResult:
        """Run CBS and return a conflict-free multi-agent solution.

        Args:
            world: Runtime world object used by low-level planning.
            robots: List of robot instances to plan.
            objective: High-level objective, ``"soc"`` or ``"makespan"``.

        Returns:
            A conflict-free solution if found, otherwise ``None``.
        """
        t0 = time.perf_counter()
        robots_by_id = {r.id: r for r in robots}

        expanded_ct = 0
        generated_ct = 0
        duplicate_skipped = 0
        replans = 0
        replan_failures = 0
        cardinal_probes = 0
        best_cost_seen: float | None = None

        root = self._build_root(world, robots, objective=objective)
        if root is None:
            return CBSResult(
                status="root_infeasible",
                solution=None,
                objective=objective,
                expanded_ct=expanded_ct,
                generated_ct=generated_ct,
                duplicate_skipped=duplicate_skipped,
                replans=replans,
                replan_failures=replan_failures,
                cardinal_probes=cardinal_probes,
                wall_time_sec=time.perf_counter() - t0,
                best_cost_seen=best_cost_seen,
            )

        open_list: list[CTNode] = []
        heapq.heappush(open_list, root)
        generated_ct += 1
        best_cost_seen = root.cost

        visited: set[frozenset[Constraint]] = set()
        visited.add(self._constraint_signature(root.constraints))

        while open_list:
            if self.timeout_sec is not None and (time.perf_counter() - t0) >= self.timeout_sec:
                return CBSResult(
                    status="timeout",
                    solution=None,
                    objective=objective,
                    expanded_ct=expanded_ct,
                    generated_ct=generated_ct,
                    duplicate_skipped=duplicate_skipped,
                    replans=replans,
                    replan_failures=replan_failures,
                    cardinal_probes=cardinal_probes,
                    wall_time_sec=time.perf_counter() - t0,
                    best_cost_seen=best_cost_seen,
                )

            if expanded_ct >= self.max_ct_nodes:
                return CBSResult(
                    status="node_budget_exceeded",
                    solution=None,
                    objective=objective,
                    expanded_ct=expanded_ct,
                    generated_ct=generated_ct,
                    duplicate_skipped=duplicate_skipped,
                    replans=replans,
                    replan_failures=replan_failures,
                    cardinal_probes=cardinal_probes,
                    wall_time_sec=time.perf_counter() - t0,
                    best_cost_seen=best_cost_seen,
                )

            node = heapq.heappop(open_list)
            expanded_ct += 1
            best_cost_seen = node.cost if best_cost_seen is None else min(best_cost_seen, node.cost)

            if self.debug:
                print(
                    f"\n[cyan]Expand CT node[/cyan] id={node.id}, depth={node.depth}, "
                    f"cost={node.cost}, constraints={len(node.constraints)}, conflict={node.conflict}"
                )

            if node.conflict is None:
                return CBSResult(
                    status="success",
                    solution=MultiAgentSolution(paths=node.paths),
                    objective=objective,
                    expanded_ct=expanded_ct,
                    generated_ct=generated_ct,
                    duplicate_skipped=duplicate_skipped,
                    replans=replans,
                    replan_failures=replan_failures,
                    cardinal_probes=cardinal_probes,
                    wall_time_sec=time.perf_counter() - t0,
                    best_cost_seen=best_cost_seen,
                )
            
            conflict, probe_count = self._select_conflict(
                node.paths, 
                world=world,
                robots_by_id=robots_by_id,
                node_constraints=node.constraints,
            )
            cardinal_probes += probe_count
            if conflict is None:
                # No conflicts found, should not happen since node.conflict is not None
                return CBSResult(
                    status="success",
                    solution=MultiAgentSolution(paths=node.paths),
                    objective=objective,
                    expanded_ct=expanded_ct,
                    generated_ct=generated_ct,
                    duplicate_skipped=duplicate_skipped,
                    replans=replans,
                    replan_failures=replan_failures,
                    cardinal_probes=cardinal_probes,
                    wall_time_sec=time.perf_counter() - t0,
                    best_cost_seen=best_cost_seen,
                )
            c1, c2 = split_conflict_to_constraints(conflict)

            for new_constraint in (c1, c2):
                child_constraints: list[Constraint] = list(node.constraints) + [new_constraint]
                sig = self._constraint_signature(child_constraints)

                if sig in visited:
                    duplicate_skipped += 1
                    if self.debug:
                        print(f"[dim]Skip duplicate constraint set:[/dim] {child_constraints}")
                    continue
                
                replans += 1
                replanned_paths = self._replan_one_agent(
                    world=world,
                    robots_by_id=robots_by_id,
                    constraints=child_constraints,
                    old_paths=node.paths,
                    agent_id=new_constraint.agent,
                )
                if replanned_paths is None:
                    replan_failures += 1
                    continue

                child_conflict, _ = self._select_conflict(
                    replanned_paths,
                    world=world,
                    robots_by_id=robots_by_id,
                    node_constraints=child_constraints,
                )
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
                generated_ct += 1

                if self.debug:
                    print(
                        f"[blue] Push child[/blue] id={child.id}, depth={child.depth}, "
                        f"cost={child.cost}, constraint={new_constraint}, conflict={child.conflict}"
                    )

        return CBSResult(
            status="open_exhausted",
            solution=None,
            objective=objective,
            expanded_ct=expanded_ct,
            generated_ct=generated_ct,
            duplicate_skipped=duplicate_skipped,
            replans=replans,
            replan_failures=replan_failures,
            cardinal_probes=cardinal_probes,
            wall_time_sec=time.perf_counter() - t0,
            best_cost_seen=best_cost_seen,
        )