from __future__ import annotations
import time

from mapf_lab.core.constraints import Constraint
from mapf_lab.core.conflicts import detect_all_conflicts
from mapf_lab.planners.cbs.planner import CBSPlanner
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
        }

    def solve(self, world, robots, objective: str = "soc") -> CBSResult:
        return super().solve(world, robots, objective)