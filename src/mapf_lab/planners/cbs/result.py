from __future__ import annotations

"""Result container for a complete CBS run."""

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from mapf_lab.core.solution import MultiAgentSolution


# High-level termination reason reported by the CBS search.
CBSStatus = Literal[
    "success",
    "root_infeasible",
    "timeout",
    "node_budget_exceeded",
    "open_exhausted",
]


@dataclass
class CBSResult:
    """Aggregated outcome and diagnostics for one CBS solve invocation.

    Attributes:
        status: Terminal search status.
        solution: Conflict-free multi-agent solution when ``status == "success"``.
        objective: High-level objective used to rank CT nodes.
        expanded_ct: Number of CT nodes popped from the open list.
        generated_ct: Number of CT nodes inserted into the open list.
        duplicate_skipped: Number of child nodes skipped due to duplicate constraints.
        replans: Number of low-level replanning attempts triggered by conflicts.
        replan_failures: Number of replans that returned no feasible path.
        wall_time_sec: End-to-end runtime of the CBS solve call.
        best_cost_seen: Lowest CT-node cost observed during the search.
    """
    status: CBSStatus
    solution: MultiAgentSolution | None
    objective: str

    expanded_ct: int
    generated_ct: int
    duplicate_skipped: int
    replans: int
    replan_failures: int
    cardinal_probes: int

    wall_time_sec: float
    best_cost_seen: float | None = None
    extra_metrics: dict[str, int | float | str | bool | None] = field(default_factory=dict)

    def success(self) -> bool:
        """Return ``True`` only for a valid solution-bearing success result."""
        return self.status == "success" and self.solution is not None

    def to_dict(self) -> dict:
        """Serialize the result into plain Python types for logging or export."""
        return {
            "status": self.status,
            "objective": self.objective,
            "expanded_ct": self.expanded_ct,
            "generated_ct": self.generated_ct,
            "duplicate_skipped": self.duplicate_skipped,
            "replans": self.replans,
            "replan_failures": self.replan_failures,
            "cardinal_probes": self.cardinal_probes,
            "wall_time_sec": self.wall_time_sec,
            "best_cost_seen": self.best_cost_seen,
            "extra_metrics": self.extra_metrics,
            "solution": None if self.solution is None else self.solution.to_dict(),
        }