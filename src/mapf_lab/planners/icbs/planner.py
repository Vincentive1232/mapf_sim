from __future__ import annotations
import heapq
import time

from mapf_lab.core.constraints import Constraint, split_conflict_to_constraints
from mapf_lab.core.conflicts import detect_all_conflicts
from mapf_lab.core.solution import MultiAgentSolution
from mapf_lab.planners.cbs.ct_node import CTNode
from mapf_lab.planners.cbs.planner import CBSPlanner
from mapf_lab.planners.cbs.result import CBSResult
from mapf_lab.planners.icbs.bypass import BypassCandidate, choose_bypass_candidate
from mapf_lab.planners.icbs.cardinal import classify_conflict
from mapf_lab.planners.icbs.conflict_selection import select_classified_conflict
from mapf_lab.planners.icbs.mdd import build_mdd, MDD


class ICBSPlanner(CBSPlanner):
    def __init__(
        self,
        low_level,
        max_ct_nodes: int = 10000,
        timeout_sec: float | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            low_level=low_level,
            max_ct_nodes=max_ct_nodes,
            timeout_sec=timeout_sec,
            debug=debug,
        )
        self._mdd_cache: dict[tuple[int, frozenset[Constraint], int], MDD] = {}

        self.mdd_cache_hits = 0
        self.mdd_cache_misses = 0
        self.mdd_build_count = 0
        self.classified_conflicts_total = 0
        self.select_conflict_calls = 0

        self.time_detect_conflicts = 0.0
        self.time_get_mdd = 0.0
        self.time_classify = 0.0
        self.time_select_rank = 0.0
        self.time_low_level_replan = 0.0

        self.bypass_attempts = 0
        self.bypass_successes = 0

    def _classify_one_conflict(self, conflict, *, paths, world, robots_by_id, node_constraints):
        mdd_i = self._get_mdd(
            world=world,
            robot=robots_by_id[conflict.agent_i],
            node_constraints=node_constraints,
            optimal_cost=int(paths[conflict.agent_i].cost()),
        )
        mdd_j = self._get_mdd(
            world=world,
            robot=robots_by_id[conflict.agent_j],
            node_constraints=node_constraints,
            optimal_cost=int(paths[conflict.agent_j].cost()),
        )
        return classify_conflict(conflict=conflict, mdd_i=mdd_i, mdd_j=mdd_j)

    def _try_bypass_node(
        self,
        *,
        node: CTNode,
        node_conflict,
        node_conflicts: list,
        world,
        robots_by_id,
        objective: str,
        visited: set[frozenset[Constraint]],
    ) -> tuple[bool, int, int, int, int]:
        """Try ICBS bypass for one non-cardinal conflict.

        Returns:
            ``(applied, probe_count, generated_inc, replan_inc, replan_failures_inc)``.
        """
        classified = self._classify_one_conflict(
            node_conflict,
            paths=node.paths,
            world=world,
            robots_by_id=robots_by_id,
            node_constraints=node.constraints,
        )
        if classified.cardinality == "cardinal":
            return False, 0, 0, 0, 0

        self.bypass_attempts += 1

        candidates: list[BypassCandidate] = []
        probe_count = 0
        generated_inc = 0
        replan_inc = 0
        replan_failures_inc = 0

        c1, c2 = split_conflict_to_constraints(node_conflict)
        for new_constraint in (c1, c2):
            child_constraints: list[Constraint] = list(node.constraints) + [new_constraint]
            sig = self._constraint_signature(child_constraints)
            if sig in visited:
                continue

            replan_inc += 1
            t0 = time.perf_counter()
            replanned_paths = self._replan_one_agent(
                world=world,
                robots_by_id=robots_by_id,
                constraints=child_constraints,
                old_paths=node.paths,
                agent_id=new_constraint.agent,
            )
            self.time_low_level_replan += time.perf_counter() - t0
            if replanned_paths is None:
                replan_failures_inc += 1
                continue

            child_cost = self._compute_cost(replanned_paths, objective=objective)
            child_conflict, child_probe_count = self._select_conflict(
                replanned_paths,
                world=world,
                robots_by_id=robots_by_id,
                node_constraints=child_constraints,
            )
            probe_count += child_probe_count
            child_conflicts = detect_all_conflicts(replanned_paths)

            candidates.append(
                BypassCandidate(
                    constraints=child_constraints,
                    paths=replanned_paths,
                    conflict=child_conflict,
                    conflicts=child_conflicts,
                    cost=child_cost,
                )
            )

            visited.add(sig)
            generated_inc += 1

        chosen = choose_bypass_candidate(
            candidates,
            parent_cost=node.cost,
            parent_conflicts=node_conflicts,
        )
        if chosen is None:
            return False, probe_count, generated_inc, replan_inc, replan_failures_inc

        node.constraints = chosen.constraints
        node.paths = chosen.paths
        node.conflict = chosen.conflict
        node.cost = chosen.cost
        node.priority = (node.cost, node.depth, node.id)
        self.bypass_successes += 1
        return True, probe_count, generated_inc, replan_inc, replan_failures_inc

    def _constraints_for_agent(
        self,
        node_constraints: list[Constraint],
        agent_id: int,
    ) -> list[Constraint]:
        return [c for c in node_constraints if c.agent == agent_id]
    
    def _get_mdd_for_agent_constraints(
        self,
        *,
        world,
        robot,
        agent_constraints: list[Constraint],
        optimal_cost: int,
    ) -> MDD:
        key = (robot.id, frozenset(agent_constraints), int(optimal_cost))

        cached = self._mdd_cache.get(key)
        if cached is not None:
            self.mdd_cache_hits += 1
            return cached

        self.mdd_cache_misses += 1
        self.mdd_build_count += 1

        mdd = build_mdd(
            world=world,
            robot=robot,
            constraints=agent_constraints,
            optimal_cost=int(optimal_cost),
            heuristic_name=getattr(self.low_level, "heuristic_name", "manhattan"),
        )
        self._mdd_cache[key] = mdd
        return mdd
    
    def _get_mdd(
        self,
        *,
        world,
        robot,
        node_constraints: list[Constraint],
        optimal_cost: int,
    ) -> MDD:
        
        agent_constraints = self._constraints_for_agent(node_constraints, robot.id)
        key = (robot.id, frozenset(agent_constraints), int(optimal_cost))

        cached = self._mdd_cache.get(key)
        if cached is not None:
            self.mdd_cache_hits += 1
            return cached

        self.mdd_cache_misses += 1
        self.mdd_build_count += 1
        
        t1 = time.perf_counter()
        mdd = build_mdd(
            world=world,
            robot=robot,
            constraints=agent_constraints,
            optimal_cost=int(optimal_cost),
            heuristic_type=getattr(self.low_level, "heuristic_type", "manhattan"),
        )

        self._mdd_cache[key] = mdd

        self.time_get_mdd += time.perf_counter() - t1
        return mdd

    def _select_conflict(self, paths, world=None, robots_by_id=None, node_constraints=None):
        t1 = time.perf_counter()
        conflicts = detect_all_conflicts(paths)
        self.time_detect_conflicts += time.perf_counter() - t1
        if not conflicts:
            return None, 0
        
        conflicts = sorted(conflicts, key=lambda c: c.time)
        conflicts = conflicts[:8]

        involved_agents = set()
        for conf in conflicts:
            involved_agents.add(conf.agent_i)
            involved_agents.add(conf.agent_j)

        mdds: dict[int, MDD] = {}
        for agent_id in involved_agents:
            robot = robots_by_id[agent_id]
            optimal_cost = int(paths[agent_id].cost())
            mdds[agent_id] = self._get_mdd(
                world=world,
                robot=robot,
                node_constraints=node_constraints or [],
                optimal_cost=optimal_cost,
            )

        t1 = time.perf_counter()
        classified = [
            classify_conflict(
                conflict=conf,
                mdd_i=mdds[conf.agent_i],
                mdd_j=mdds[conf.agent_j],
            )
            for conf in conflicts
        ]
        self.time_classify += time.perf_counter() - t1

        t2 = time.perf_counter()
        chosen = select_classified_conflict(classified)
        self.time_select_rank += time.perf_counter() - t2

        self.select_conflict_calls += 1
        self.classified_conflicts_total += len(conflicts)

        return chosen, 0

    def _reset_algorithm_counters(self) -> None:
        # Reset per-run ICBS diagnostics so result metrics are scoped to one solve.
        self._mdd_cache.clear()
        self.mdd_cache_hits = 0
        self.mdd_cache_misses = 0
        self.mdd_build_count = 0
        self.classified_conflicts_total = 0
        self.select_conflict_calls = 0
        self.time_detect_conflicts = 0.0
        self.time_get_mdd = 0.0
        self.time_classify = 0.0
        self.time_select_rank = 0.0
        self.time_low_level_replan = 0.0
        self.bypass_attempts = 0
        self.bypass_successes = 0

    def _collect_algorithm_metrics(self) -> dict[str, int | float | str | bool | None]:
        avg_conflicts = self.classified_conflicts_total / max(1, self.select_conflict_calls)
        return {
            "mdd_cache_hits": self.mdd_cache_hits,
            "mdd_cache_misses": self.mdd_cache_misses,
            "mdd_build_count": self.mdd_build_count,
            "select_conflict_calls": self.select_conflict_calls,
            "classified_conflicts_total": self.classified_conflicts_total,
            "avg_conflicts_per_select": avg_conflicts,
            "time_detect_conflicts": self.time_detect_conflicts,
            "time_get_mdd": self.time_get_mdd,
            "time_classify": self.time_classify,
            "time_select_rank": self.time_select_rank,
            "time_low_level_replan": self.time_low_level_replan,
            "bypass_attempts": self.bypass_attempts,
            "bypass_successes": self.bypass_successes,
        }

    def solve(self, world, robots, objective: str = "soc") -> CBSResult:
        t0 = time.perf_counter()
        self._reset_algorithm_counters()
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
            return self._make_result(
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

        root_conflict, probe_count = self._select_conflict(
            root.paths,
            world=world,
            robots_by_id=robots_by_id,
            node_constraints=root.constraints,
        )
        cardinal_probes += probe_count
        root.conflict = root_conflict

        open_list: list[CTNode] = []
        heapq.heappush(open_list, root)
        generated_ct += 1
        best_cost_seen = root.cost

        visited: set[frozenset[Constraint]] = set()
        visited.add(self._constraint_signature(root.constraints))

        while open_list:
            if self.timeout_sec is not None and (time.perf_counter() - t0) >= self.timeout_sec:
                return self._make_result(
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
                return self._make_result(
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
                return self._make_result(
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

            # Bypass loop: keep updating the current node while we can reduce
            # conflicts without increasing the node cost.
            while node.conflict is not None:
                node_conflicts = detect_all_conflicts(node.paths)
                applied, probe_inc, gen_inc, replan_inc, replan_fail_inc = self._try_bypass_node(
                    node=node,
                    node_conflict=node.conflict,
                    node_conflicts=node_conflicts,
                    world=world,
                    robots_by_id=robots_by_id,
                    objective=objective,
                    visited=visited,
                )
                cardinal_probes += probe_inc
                generated_ct += gen_inc
                replans += replan_inc
                replan_failures += replan_fail_inc

                if not applied:
                    break

                if node.conflict is None:
                    return self._make_result(
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

            if node.conflict is None:
                continue

            c1, c2 = split_conflict_to_constraints(node.conflict)
            for new_constraint in (c1, c2):
                child_constraints: list[Constraint] = list(node.constraints) + [new_constraint]
                sig = self._constraint_signature(child_constraints)

                if sig in visited:
                    duplicate_skipped += 1
                    if self.debug:
                        print(f"[dim]Skip duplicate constraint set:[/dim] {child_constraints}")
                    continue

                replans += 1
                t_replan = time.perf_counter()
                replanned_paths = self._replan_one_agent(
                    world=world,
                    robots_by_id=robots_by_id,
                    constraints=child_constraints,
                    old_paths=node.paths,
                    agent_id=new_constraint.agent,
                )
                self.time_low_level_replan += time.perf_counter() - t_replan

                if replanned_paths is None:
                    replan_failures += 1
                    continue

                child_conflict, probe_count = self._select_conflict(
                    replanned_paths,
                    world=world,
                    robots_by_id=robots_by_id,
                    node_constraints=child_constraints,
                )
                cardinal_probes += probe_count

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

        return self._make_result(
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