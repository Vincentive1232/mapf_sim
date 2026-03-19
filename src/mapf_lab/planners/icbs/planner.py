from __future__ import annotations

from mapf_lab.core.conflicts import detect_all_conflicts
from mapf_lab.planners.cbs.planner import CBSPlanner
from mapf_lab.planners.icbs.cardinal import classify_conflict
from mapf_lab.planners.icbs.conflict_selection import select_classified_conflict


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

    def _select_conflict(self, paths, world=None, robots_by_id=None, node_constraints=None):
        conflicts = detect_all_conflicts(paths)
        if not conflicts:
            return None, 0

        classified = [
            classify_conflict(
                conflict=conf,
                node_constraints=node_constraints or [],
                node_paths=paths,
                world=world,
                robots_by_id=robots_by_id,
                low_level=self.low_level,
            )
            for conf in conflicts
        ]
        probe_count = 2 * len(conflicts)

        if self.debug:
            debug_view = [
                {
                    "time": cc.conflict.time,
                    "type": cc.conflict.kind,
                    "cardinality": cc.cardinality,
                    "increased_i": cc.increased_i,
                    "increased_j": cc.increased_j,
                }
                for cc in classified
            ]
            print("Classified conflicts:", debug_view)

        return select_classified_conflict(
            classified,
            mode="cardinal_then_earliest",
        ), probe_count